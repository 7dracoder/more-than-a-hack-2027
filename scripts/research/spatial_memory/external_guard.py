"""External-study isolation with explicit read-only derivative input files."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def restrict_external(out, allowed_reads=()):
    data = (ROOT / 'data').resolve()
    external = (data / 'external').resolve()
    truth = (ROOT / 'evaluation/gt').resolve()
    out = Path(out).resolve()
    allowed = {Path(p).resolve() for p in allowed_reads}
    if not out.is_relative_to(data) or out.is_relative_to(external):
        raise ValueError('External study output must be a separate ignored directory')
    denied = []

    def guard(event, values):
        if event != 'open' or not values or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        mode = values[1] if len(values) > 1 else ''
        flags = values[2] if len(values) > 2 else 0
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
        forbidden = path.is_relative_to(truth)
        if path.is_relative_to(data) and not path.is_relative_to(out):
            forbidden |= writing or (not path.is_relative_to(external) and path not in allowed)
        if forbidden:
            denied.append(str(path))
            raise PermissionError('External study cannot access supplied assets/predictions/truth or mutate its inputs')

    sys.addaudithook(guard)
    for probe in (truth / '__probe__.csv', data / 'tactical/__probe__.txt'):
        try:
            open(probe, 'rb')
            raise AssertionError('External isolation failed')
        except PermissionError:
            pass
    denied.clear()
    return denied
