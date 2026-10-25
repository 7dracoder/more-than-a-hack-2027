"""Fetch pinned public SkillCorner inputs with Git-blob and LFS checksum verification."""
import argparse,hashlib,json,re,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
COMMIT='4340d274572876239c154c90bc507a9b3250a656'


def response(url):
    return urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'local-possession-research'}),timeout=60)


def blob(url,entry,limit):
    with response(url) as stream:data=stream.read(limit+1)
    if len(data)>limit:raise ValueError('Public source exceeds declared size limit')
    digest=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
    if digest!=entry['sha']:raise ValueError('Pinned Git blob differs')
    return data


def sha(path):
    with open(path,'rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matches',nargs='*');parser.add_argument('--all',action='store_true')
    args=parser.parse_args()
    if not args.all and not args.matches:parser.error('Choose --all or --matches')
    root=ROOT/'data/external/skillcorner-raw';root.mkdir(parents=True,exist_ok=True)
    with response(f'https://api.github.com/repos/SkillCorner/opendata/git/trees/{COMMIT}?recursive=1') as stream:
        tree=json.loads(stream.read(2_000_000))
    if tree.get('truncated'):raise ValueError('Incomplete source inventory')
    entries={row['path']:row for row in tree['tree'] if row['type']=='blob'}
    base=f'https://raw.githubusercontent.com/SkillCorner/opendata/{COMMIT}/'
    for path in ('LICENSE','README.md','data/matches.json'):
        data=blob(base+path,entries[path],1_000_000)
        (root/Path(path).name).write_bytes(data)
    available=sorted(path.split('/')[2] for path in entries if path.startswith('data/matches/') and path.endswith('_tracking_extrapolated.jsonl'))
    matches=available if args.all else args.matches
    if len(matches)!=len(set(matches)) or not set(matches)<=set(available):raise ValueError('Unknown or duplicate match')
    (root/'inventory.json').write_text(json.dumps(dict(repository='https://github.com/SkillCorner/opendata',commit=COMMIT,matches=available),indent=2))
    for match in matches:
        directory=root/match;directory.mkdir(exist_ok=True)
        prefix=f'data/matches/{match}/{match}_'
        metadata=blob(base+prefix+'match.json',entries[prefix+'match.json'],1_000_000)
        target=directory/'metadata.json'
        if target.exists() and target.read_bytes()!=metadata:raise ValueError('Existing metadata differs')
        target.write_bytes(metadata)
        pointer=blob(base+prefix+'tracking_extrapolated.jsonl',entries[prefix+'tracking_extrapolated.jsonl'],1024).decode()
        digest=re.search(r'oid sha256:([a-f0-9]{64})',pointer);size=re.search(r'size (\d+)',pointer)
        if not digest or not size:raise ValueError('Expected public Git LFS pointer')
        digest=digest.group(1);size=int(size.group(1))
        if size>200_000_000:raise ValueError('Match exceeds download bound')
        tracking=directory/'tracking.jsonl'
        inspected=ROOT/('data/external/skillcorner-inspection/'+match+'_tracking.jsonl')
        if not tracking.exists() and inspected.exists() and sha(inspected)==digest:shutil.copyfile(inspected,tracking)
        if not tracking.exists():
            partial=tracking.with_suffix('.partial');total=0;actual=hashlib.sha256()
            with response(f'https://media.githubusercontent.com/media/SkillCorner/opendata/{COMMIT}/'+prefix+'tracking_extrapolated.jsonl') as stream,partial.open('wb') as output:
                while block:=stream.read(1_048_576):
                    total+=len(block)
                    if total>size:raise ValueError('Download exceeds pinned size')
                    output.write(block);actual.update(block)
            if total!=size or actual.hexdigest()!=digest:raise ValueError('Downloaded LFS data differs')
            partial.replace(tracking)
        if tracking.stat().st_size!=size or sha(tracking)!=digest:raise ValueError('Existing tracking differs')
        record=dict(repository='https://github.com/SkillCorner/opendata',commit=COMMIT,match=match,
            tracking_sha256=digest,tracking_bytes=size,metadata_sha256=sha(target),license_sha256=sha(root/'LICENSE'),
            label_kind='Provider H/A weak targets; null stays unknown; no D labels inferred',trained=False)
        (directory/'source.json').write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(match=match,tracking_bytes=size,sha256=digest,verified=True)),flush=True)


if __name__=='__main__':main()
