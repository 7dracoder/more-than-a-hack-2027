"""Pin and verify Metrica's public sample files; never reads challenge labels."""
import hashlib
import json
from pathlib import Path
import sys
import requests

ROOT = Path(__file__).resolve().parents[1]
API = 'https://api.github.com/repos/metrica-sports/sample-data'
RAW = 'https://raw.githubusercontent.com/metrica-sports/sample-data/'


def blob_hash(path, size):
    digest = hashlib.sha1(('blob '+str(size)+'\0').encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    folder = ROOT/'data/external/metrica-raw'; folder.mkdir(parents=True, exist_ok=True)
    session = requests.Session(); session.headers['User-Agent'] = 'Reproducible-football-data-audit'
    inventory = folder/'inventory.json'
    if inventory.exists():
        record = json.loads(inventory.read_text())
    else:
        response = session.get(API+'/commits/master', timeout=60); response.raise_for_status(); commit = response.json()
        response = session.get(API+'/git/trees/'+commit['commit']['tree']['sha'], params={'recursive': '1'}, timeout=60)
        response.raise_for_status(); tree = response.json()
        if tree.get('truncated'):
            raise ValueError('Incomplete public file inventory')
        record = dict(repository='https://github.com/metrica-sports/sample-data', commit=commit['sha'],
                      commit_date=commit['commit']['committer']['date'], tree_sha=tree['sha'],
                      files=[f for f in tree['tree'] if f['type'] == 'blob'],
                      credit='Metrica Sports', terms='Public sample use and acknowledgement described in preserved README; no invented SPDX license.')
        if record['commit_date'] >= '2024-01-01':
            raise ValueError('Recheck match provenance: public sample release is not older than challenge season')
        inventory.write_text(json.dumps(record, indent=2)+'\n')
    roles = folder/'roles.json'
    expected = dict(training=['METRICA1', 'METRICA2'], held_out=['METRICA3'],
                    held_out_policy='No target parsing, optimization or model selection on METRICA3 until the additional-data candidate is frozen. Prior public schema-only inspection is documented; no prediction score was opened.',
                    label_kind='event_derived_partial_not_dense_ground_truth')
    if roles.exists() and json.loads(roles.read_text()) != expected:
        raise ValueError('Preserve the predeclared match split')
    roles.write_text(json.dumps(expected, indent=2)+'\n')
    outputs = []
    for item in record['files']:
        path = folder/item['path']; size = item['size']
        if size > 100_000_000 or '..' in Path(item['path']).parts:
            raise ValueError('Unexpected public asset size/path')
        path.parent.mkdir(parents=True, exist_ok=True)
        if not (path.exists() and path.stat().st_size == size and blob_hash(path, size) == item['sha']):
            partial = path.with_suffix(path.suffix+'.partial')
            with session.get(RAW+record['commit']+'/'+item['path'], stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with partial.open('wb') as stream:
                    for block in response.iter_content(1048576):
                        stream.write(block)
            if partial.stat().st_size != size or blob_hash(partial, size) != item['sha']:
                raise ValueError('Public Git blob verification failed: '+item['path'])
            partial.replace(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        outputs.append(dict(path=item['path'], bytes=size, git_blob_sha1=item['sha'], sha256=digest))
        print(json.dumps(outputs[-1]), flush=True)
    (folder/'download_manifest.json').write_text(json.dumps(dict(commit=record['commit'],
         commit_date=record['commit_date'], files=outputs, no_challenge_labels_read=True), indent=2)+'\n')


if __name__ == '__main__':
    main()
