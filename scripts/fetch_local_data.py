"""Download official external matches and the previously checksum-pinned tactical inputs.

All downloads stay in ignored data/. Never uploads match data. Existing verified
files are reused; interrupted downloads cannot masquerade as complete inputs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
ARTICLE = 'https://api.figshare.com/v2/articles/28196177'


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def download(item, folder):
    path = folder / item['name']
    expected = item['computed_md5']
    if path.exists() and path.stat().st_size == item['size'] and digest(path, 'md5') == expected:
        print('Verified existing ' + path.name, flush=True)
        return dict(name=path.name, sha256=digest(path), md5=expected, bytes=path.stat().st_size)
    part = path.with_suffix(path.suffix + '.partial')
    for attempt in range(3):
        try:
            with requests.get(item['download_url'], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with part.open('wb') as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
            if part.stat().st_size != item['size'] or digest(part, 'md5') != expected:
                raise ValueError('Official size/MD5 mismatch: ' + path.name)
            part.replace(path)
            print('Downloaded and verified ' + path.name, flush=True)
            return dict(name=path.name, sha256=digest(path), md5=expected, bytes=path.stat().st_size)
        except (requests.RequestException, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def external():
    folder = ROOT / 'data/external/sportec'
    folder.mkdir(parents=True, exist_ok=True)
    response = requests.get(ARTICLE, timeout=60)
    response.raise_for_status()
    record = response.json()
    (folder / 'article.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
    with ThreadPoolExecutor(max_workers=3) as pool:
        outputs = list(pool.map(lambda item: download(item, folder), record['files']))
    manifest = dict(source=ARTICLE, title=record['title'], files=outputs,
                    purpose='External training/validation only; no BAR-ATH labels.')
    (folder / 'download_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')


def competition():
    import gdown
    notebook = json.loads((ROOT / 'notebooks/BlindSpot_v031_Colab.ipynb').read_text(encoding='utf-8'))
    # Parse the two literal dictionaries; never execute notebook cells.
    import ast
    values = {}
    for cell in notebook['cells']:
        for line in ''.join(cell.get('source', [])).splitlines():
            for name in ('ASSETS', 'EXPECTED'):
                if line.startswith(name + ' = '):
                    values[name] = ast.literal_eval(line.split(' = ', 1)[1])
    folder = ROOT / 'data/tactical'
    folder.mkdir(parents=True, exist_ok=True)
    records = []
    for name, file_id in values['ASSETS'].items():
        path = folder / name
        expected = values['EXPECTED'][name]
        if not path.exists() or digest(path) != expected:
            part = path.with_suffix(path.suffix + '.partial')
            result = gdown.download(id=file_id, output=str(part), quiet=False)
            if result is None or digest(part) != expected:
                raise ValueError('Tactical download unavailable or SHA256 mismatch: ' + name)
            part.replace(path)
        records.append(dict(name=name, bytes=path.stat().st_size, sha256=expected))
        print('Verified tactical input ' + name, flush=True)
    (folder / 'download_manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--competition', action='store_true')
    args = parser.parse_args()
    if not (args.external or args.competition):
        parser.error('Select --external and/or --competition')
    if args.external:
        external()
    if args.competition:
        competition()
