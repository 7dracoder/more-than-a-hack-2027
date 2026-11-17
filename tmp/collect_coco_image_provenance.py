"""Read only the image-record prefix of official COCO annotation archives."""
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import zipfile
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from scripts.research.video_ball.prepare_coco_calibration import RangeReader,ZIP_URL,ZIP_ETAG
import requests
out=root/'data/external/openimages-ball-review-20261115';out.mkdir(parents=True,exist_ok=True)

def image_records(stream):
    decoder=json.JSONDecoder();buffer=''
    while True:
        buffer+=stream.read(1024*1024)
        marker=re.search(r'"images"\s*:\s*\[',buffer)
        if marker:buffer=buffer[marker.end():];break
        if len(buffer)>8*1024*1024:raise ValueError('Image-record prefix missing')
    while True:
        buffer=buffer.lstrip(' \r\n\t,')
        if buffer.startswith(']'):return
        try:
            record,end=decoder.raw_decode(buffer)
        except json.JSONDecodeError:
            chunk=stream.read(1024*1024)
            if not chunk:raise ValueError('Truncated image records')
            buffer+=chunk;continue
        if not isinstance(record,dict) or not {'id','file_name','flickr_url'}<=record.keys():raise ValueError('Unexpected image record')
        yield record;buffer=buffer[end:]

train_path=out/'coco_train_image_records.json'
if train_path.exists():raise ValueError('Preserve acquired provenance')
with requests.Session() as session:
    reader=RangeReader(session)
    with zipfile.ZipFile(reader) as archive:
        info=archive.getinfo('annotations/instances_train2017.json')
        with archive.open(info) as binary,io.TextIOWrapper(binary,encoding='utf-8') as stream:records=list(image_records(stream))
    downloaded=reader.bytes_read
assert len(records)==118287
train_path.write_text(json.dumps(records,separators=(',',':')),encoding='utf-8')
validation_path=root/'data/external/coco2017-calibration/instances_val2017.json'
validation=json.loads(validation_path.read_text(encoding='utf-8'))['images']
def photo_id(url):
    found=re.search(r'/([0-9]{5,})_[0-9a-f]+',url)
    return found.group(1) if found else None
ids=sorted({key for r in records+validation if (key:=photo_id(r['flickr_url'])) is not None})
(out/'coco_photo_ids.json').write_text(json.dumps(ids))
receipt=dict(source=ZIP_URL,etag=ZIP_ETAG,archive_member=info.filename,member_crc32=info.CRC,member_uncompressed_bytes=info.file_size,network_bytes_read=downloaded,train_image_records=len(records),validation_image_records=len(validation),unique_photo_ids=len(ids),train_metadata_sha256=hashlib.sha256(train_path.read_bytes()).hexdigest(),validation_annotations_sha256=hashlib.sha256(validation_path.read_bytes()).hexdigest(),provided_match_used=False,images_downloaded=False)
(out/'coco_provenance.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
