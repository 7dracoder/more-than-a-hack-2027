"""Prepare licensed external COCO 2017 still images for confidence calibration.

No BAR-ATH input, video or possession label is read. Annotations are CC BY 4.0;
only images explicitly licensed CC BY or CC BY-SA are selected. All licensed
non-crowd sports-ball images and three deterministic negatives per positive
form a frozen image-level development/validation split. No match is requested.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlparse
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[3]
BUCKET = 'https://s3.amazonaws.com/images.cocodataset.org'
ZIP_URL = BUCKET + '/annotations/annotations_trainval2017.zip'
ZIP_SIZE = 252907541
ZIP_ETAG = '"f4bbac642086de4f52a3fdda2de5fa2c"'


def sha256(data):
    return hashlib.sha256(data).hexdigest()


class RangeReader(io.RawIOBase):
    """Bounded verified-HTTPS reads of the publisher's unchanged ZIP object."""
    def __init__(self, session):
        self.session, self.position, self.bytes_read = session, 0, 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        target = offset if whence == 0 else self.position + offset if whence == 1 else ZIP_SIZE + offset
        if not 0 <= target <= ZIP_SIZE:
            raise ValueError('Invalid archive seek')
        self.position = target
        return target

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = ZIP_SIZE - self.position if size < 0 else min(size, ZIP_SIZE - self.position)
        if size == 0:
            return b''
        if size > 64 * 1024 * 1024:
            raise ValueError('Refusing an unexpectedly large archive read')
        end = self.position + size - 1
        with self.session.get(ZIP_URL, headers={'Range': f'bytes={self.position}-{end}', 'If-Match': ZIP_ETAG},
                              timeout=(15, 45)) as response:
            response.raise_for_status()
            expected = f'bytes {self.position}-{end}/{ZIP_SIZE}'
            if response.status_code != 206 or response.headers.get('Content-Range') != expected:
                raise ValueError('Server did not honor the bounded range')
            if response.headers.get('ETag') != ZIP_ETAG or len(response.content) != size:
                raise ValueError('Publisher archive changed or was truncated')
            data = response.content
        self.position += size
        self.bytes_read += size
        return data


def allowed_license(license_record):
    url = urlparse(license_record.get('url', ''))
    return (url.hostname == 'creativecommons.org' and
            bool(re.match(r'^/licenses/(by|by-sa)/[0-9.]+/?$', url.path)))


def image_url(image):
    name = image['file_name']
    if not re.fullmatch(r'[0-9]{12}\.jpg', name):
        raise ValueError('Invalid COCO image filename')
    source = urlparse(image['coco_url'])
    if source.hostname != 'images.cocodataset.org' or source.path != '/val2017/' + name:
        raise ValueError('Unexpected image source')
    return BUCKET + source.path


def stable_key(image):
    return sha256(('coreorigin-coco-calibration-v1:' + str(image['id'])).encode())


def choose_images(annotation):
    licenses = {x['id']: x for x in annotation['licenses'] if allowed_license(x)}
    categories = {x['name']: x['id'] for x in annotation['categories']}
    ball_category = categories['sports ball']
    balls, crowds = {}, set()
    for box in annotation['annotations']:
        if box['category_id'] != ball_category:
            continue
        if box.get('iscrowd', 0):
            crowds.add(box['image_id'])
        else:
            balls.setdefault(box['image_id'], []).append(box)
    eligible = [x for x in annotation['images'] if x['license'] in licenses and x['id'] not in crowds]
    positive = sorted([x for x in eligible if x['id'] in balls], key=stable_key)
    negative = sorted([x for x in eligible if x['id'] not in balls], key=stable_key)[:3 * len(positive)]
    if not positive or not negative:
        raise ValueError('Need licensed positive and negative external images')
    selected = sorted(positive + negative, key=lambda x: x['id'])
    records = []
    for image in selected:
        record = dict(image, license_record=licenses[image['license']],
                      ball_boxes=[x['bbox'] for x in balls.get(image['id'], [])],
                      split='validation' if int(stable_key(image)[:8], 16) % 5 == 0 else 'development')
        # The external dataset predates the competition match by years.
        if record.get('date_captured') and int(record['date_captured'][:4]) >= 2024:
            raise ValueError('Unexpected post-2023 image in COCO 2017')
        record['download_url'] = image_url(record)
        records.append(record)
    return records


def download_image(record, folder):
    from PIL import Image
    path = folder / record['file_name']
    if not path.is_file():
        with requests.get(record['download_url'], timeout=(15, 45)) as response:
            response.raise_for_status()
            data = response.content
            etag = response.headers.get('ETag', '').strip('"')
        if re.fullmatch('[0-9a-f]{32}', etag) and hashlib.md5(data).hexdigest() != etag:
            raise ValueError('Image does not match publisher ETag')
        part = path.with_suffix('.jpg.partial')
        part.write_bytes(data)
        with Image.open(part) as image:
            if image.format != 'JPEG' or image.size != (record['width'], record['height']):
                raise ValueError('External image format/dimensions differ from annotations')
            image.verify()
        part.replace(path)
    data = path.read_bytes()
    with Image.open(path) as image:
        if image.size != (record['width'], record['height']):
            raise ValueError('Cached external image has incorrect dimensions')
        image.verify()
    return dict(record, sha256=sha256(data), bytes=len(data))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'data/external/coco2017-calibration')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    annotation_file = args.out / 'instances_val2017.json'
    fetched = 0
    if not annotation_file.is_file():
        with requests.Session() as session:
            source = RangeReader(session)
            with zipfile.ZipFile(source) as archive:
                data = archive.read('annotations/instances_val2017.json')  # verifies member CRC
            fetched = source.bytes_read
        annotation_file.write_bytes(data)
    annotation = json.loads(annotation_file.read_text(encoding='utf-8'))
    records = choose_images(annotation)
    protocol = dict(source=ZIP_URL, archive_etag=ZIP_ETAG, annotation_sha256=sha256(annotation_file.read_bytes()),
                    annotations_license='CC BY 4.0, COCO Consortium',
                    image_policy='Only explicit CC BY / CC BY-SA images; retain Flickr URL and license for attribution',
                    selection='All eligible non-crowd sports-ball images, three deterministic negatives per positive',
                    split='SHA256 image-id split, validation remainder zero modulo five',
                    provided_match_data_used=False, source_sha256=sha256(Path(__file__).read_bytes()))
    selection = args.out / 'frozen_selection.json'
    frozen = dict(protocol=protocol, images=records)
    if selection.is_file() and json.loads(selection.read_text(encoding='utf-8')) != frozen:
        raise ValueError('External calibration selection changed; use a separate protocol directory')
    selection.write_text(json.dumps(frozen, indent=2), encoding='utf-8')
    folder = args.out / 'images'; folder.mkdir(exist_ok=True)
    downloaded = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for item in pool.map(lambda record: download_image(record, folder), records):
            downloaded.append(item)
            if len(downloaded) % 25 == 0:
                print(f'{len(downloaded)}/{len(records)} external images verified', flush=True)
    (args.out / 'manifest.json').write_text(json.dumps(dict(protocol=protocol, images=downloaded), indent=2), encoding='utf-8')
    print(json.dumps(dict(images=len(downloaded), positive_images=sum(bool(x['ball_boxes']) for x in downloaded),
                         validation_images=sum(x['split']=='validation' for x in downloaded),
                         annotation_archive_bytes_fetched=fetched, provided_match_data_used=False), indent=2))


if __name__ == '__main__':
    main()
