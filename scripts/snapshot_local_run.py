"""Archive inspectable source/configuration without match assets or model files."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.data import sha256


def verify_source_archive(run,fixed):
    """Validate immutable historical bytes, even after the working copy evolves."""
    target=run/'source.zip'
    manifest=json.loads((run/'source_manifest.json').read_text())
    if sha256(target)!=manifest['source_zip_sha256']:
        raise ValueError('Recorded source archive bytes differ')
    with zipfile.ZipFile(target) as archive:
        names=archive.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(manifest['files']) or archive.testzip() is not None:
            raise ValueError('Source archive contents or CRC differ')
        for path,expected in manifest['files'].items():
            if hashlib.sha256(archive.read(path)).hexdigest()!=expected:
                raise ValueError('Archived file differs from source manifest: '+path)
        for path,expected in fixed.get('sources',{}).items():
            if path not in names or hashlib.sha256(archive.read(path)).hexdigest()!=expected:
                raise ValueError('Archived source differs from the run protocol: '+path)
    return dict(preserved_existing_archive=True,source_archive_verified=True,zip_sha256=manifest['source_zip_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--recover-from',type=Path,nargs='*',default=[],help='Recover changed required sources only when archived bytes match the recorded hash')
    args = parser.parse_args()
    if not args.run.is_dir():
        raise ValueError('Run must already exist')
    target = args.run/'source.zip'
    fixed = json.loads((args.run/'fixed_protocol.json').read_text())
    if target.exists():
        print(json.dumps(verify_source_archive(args.run,fixed)),flush=True)
        return
    files = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True).splitlines()
    allowed = {'.py', '.ps1', '.json', '.md', '.toml'}
    selected = sorted(set(p for p in files if (Path(p).suffix in allowed or Path(p).name.startswith('requirements'))
                          and not p.startswith(('data/', '.venv/', '.git/')) and (ROOT/p).is_file()))
    hashes = {p: sha256(ROOT/p) for p in selected}
    mismatches = [p for p, expected in fixed.get('sources', {}).items() if hashes.get(p) != expected]
    recovered={}; overrides={}
    for path in list(mismatches):
        if path not in selected:
            continue
        for source_zip in args.recover_from:
            with zipfile.ZipFile(source_zip) as previous:
                if path not in previous.namelist():
                    continue
                data=previous.read(path)
            if hashlib.sha256(data).hexdigest()==fixed['sources'][path]:
                overrides[path]=data; hashes[path]=fixed['sources'][path]
                recovered[path]=dict(archive=str(source_zip),archive_sha256=sha256(source_zip))
                mismatches.remove(path)
                break
    if mismatches:
        raise ValueError('Current sources differ from recorded run: '+str(mismatches))
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in selected:
            if path in overrides:
                archive.writestr(path,overrides[path])
            else:
                archive.write(ROOT/path, path)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Source ZIP CRC check failed')
    (args.run/'source_manifest.json').write_text(json.dumps(dict(files=hashes, source_zip_sha256=sha256(target),
                       file_count=len(hashes),restricted_data_included=False,recovered_sources=recovered), indent=2))
    print(json.dumps(dict(files=len(hashes), zip_sha256=sha256(target))), flush=True)


if __name__ == '__main__':
    main()
