"""Track opened external test cohorts across candidate output directories."""
import hashlib
import json
from pathlib import Path


def reserve_test_cohort(registry_path, cache_hashes, checkpoint_sha256, run):
    path=Path(registry_path)
    registry=json.loads(path.read_text()) if path.exists() else {}
    key=hashlib.sha256(json.dumps(cache_hashes,sort_keys=True).encode()).hexdigest()
    # Even a subset of an opened cohort is no longer a fresh test set.
    opened={match for record in registry.values() for match in record['cache_hashes']}
    overlap=opened.intersection(cache_hashes)
    if overlap:
        raise ValueError('External test matches already opened: '+', '.join(sorted(overlap))+
                         '. Preserve the original result; use new matches for independent testing.')
    registry[key]=dict(cache_hashes=cache_hashes,checkpoint_sha256=checkpoint_sha256,run=str(run),status='opened')
    part=path.with_suffix('.partial.json')
    part.write_text(json.dumps(registry,indent=2)); part.replace(path)
    return key
