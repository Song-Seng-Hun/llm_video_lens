"""One explicit workspace boundary for source data and generated artifacts."""
import hashlib
import json
import os
from pathlib import Path

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / 'workspace'
_ROOT = _DEFAULT_ROOT


def configure(root=None):
    global _ROOT
    candidate = Path(root) if root is not None else _DEFAULT_ROOT
    candidate = Path(os.path.abspath(candidate.expanduser()))
    _no_symlinks(candidate)
    _ROOT = candidate.resolve()
    return _ROOT


def workspace_root():
    return _ROOT


def _no_symlinks(path):
    for part in [path, *path.parents]:
        if part.is_symlink():
            raise ValueError('symlink paths are not allowed')


def confined(value, *, exists=False, new=False, max_bytes=None):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = _ROOT / path
    path = Path(os.path.abspath(path))
    _no_symlinks(path)
    if not path.resolve().is_relative_to(_ROOT):
        raise ValueError('path escapes selected workspace')
    if exists and not path.is_file():
        raise ValueError('workspace regular input file required')
    if new and path.exists():
        raise ValueError('refusing to overwrite existing artifact')
    if max_bytes is not None and path.exists() and path.stat().st_size > max_bytes:
        raise ValueError('input exceeds byte limit')
    return path


_SHA = {}


def _state(path):
    st = os.stat(path)
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def sha(path):
    """SHA-256 of a file. A file of 1 MiB or more keeps its digest while its state is unchanged (any write moves ctime),
    since every step re-checks the source (clip B: about 450 times a look)."""
    key = _state(path)
    if key in _SHA:
        return _SHA[key]
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            h.update(block)
    if key[2] >= 1048576 and _state(path) == key:
        _SHA[key] = h.hexdigest()
    return h.hexdigest()


def load(value, max_bytes=16*1024*1024):
    path = confined(value, exists=True, max_bytes=max_bytes)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=reject), path


def write(value, text):
    path = confined(value, new=True)
    # The user chooses the workspace. Only output parents within it are created.
    path.parent.mkdir(parents=True, exist_ok=True)
    _no_symlinks(path)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(text)
    return path


def write_json(value, packet):
    return write(value, json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
