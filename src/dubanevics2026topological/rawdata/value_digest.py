"""Value-level digests of the raw reference files.

A rebuilt ``data/raw/`` file may differ in bytes (attribute order, number formatting,
Parquet writer version) while carrying exactly the same values. These digests hash the
values only, so equal values give equal digests whatever the bytes are:

* GraphML: the node and edge attributes, nodes and edges sorted, numbers to 12 significant digits;
* peak CSV: the sorted set of peak sequences;
* Parquet: the schema and every row, sorted by all columns.

A file that cannot be read as its kind (a corrupt or foreign file) gets the digest of
its bytes under a ``bytes:`` prefix, which never equals a value digest, so such a file
is always reported as a mismatch rather than raising from inside a stage.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import polars as pl

from .check_raw import read_graphml

PEAK_COLUMN = "dna"
_BATCH_ROWS = 500_000


def bytes_sha256(path: Path) -> str:
    """SHA-256 of the bytes of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _scalar(value: object) -> str:
    if isinstance(value, float):
        return "nan" if value != value else format(value, ".12g")
    return str(value)


def _attributes(attributes: dict[str, object]) -> str:
    return ";".join(f"{key}={_scalar(attributes[key])}" for key in sorted(attributes))


def graphml_value_digest(path: Path) -> str:
    """Digest of the nodes and edges (with attributes) of a GraphML file."""
    nodes, edges = read_graphml(Path(path))
    digest = hashlib.sha256(b"graphml-values-v1\n")
    for name in sorted(nodes):
        digest.update(f"N\t{name}\t{_attributes(nodes[name])}\n".encode())
    lines = sorted(
        f"E\t{source}\t{target}\t{_attributes(dict(attrs))}\t{count}"
        for (source, target, attrs), count in edges.items()
    )
    for line in lines:
        digest.update((line + "\n").encode())
    return digest.hexdigest()


def peaks_value_digest(path: Path) -> str:
    """Digest of the sorted set of peak sequences of a peak CSV."""
    peaks = pl.read_csv(Path(path), infer_schema=False)[PEAK_COLUMN].to_list()
    digest = hashlib.sha256(b"peaks-values-v1\n")
    for peak in sorted(set(peaks)):
        digest.update(f"{peak}\n".encode())
    return digest.hexdigest()


def parquet_value_digest(path: Path) -> str:
    """Digest of the schema and the rows (sorted by every column) of a Parquet file."""
    frame = pl.read_parquet(Path(path))
    digest = hashlib.sha256(b"parquet-values-v1\n")
    digest.update(repr(list(frame.schema.items())).encode())
    frame = frame.sort(frame.columns, nulls_last=True)
    for offset in range(0, frame.height, _BATCH_ROWS):
        digest.update(frame.slice(offset, _BATCH_ROWS).write_csv().encode())
    return digest.hexdigest()


_DIGESTS = {
    ".graphml": graphml_value_digest,
    ".csv": peaks_value_digest,
    ".parquet": parquet_value_digest,
}


_MEMO: dict[tuple[str, int, int], str] = {}


def value_digest(path: Path) -> str:
    """The value digest of ``path`` by its suffix, or ``bytes:<sha256>`` when it cannot be read.

    Remembered per process for an unchanged file (path, size, mtime): a GraphML parse
    takes seconds and a stage asks for the digest more than once.
    """
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if key not in _MEMO:
        _MEMO[key] = _value_digest(path)
    return _MEMO[key]


def _value_digest(path: Path) -> str:
    function = _DIGESTS.get(path.suffix.lower())
    if function is not None:
        try:
            return function(path)
        except Exception:  # noqa: BLE001 - any unreadable file is a mismatch, not a crash
            pass
    return "bytes:" + bytes_sha256(path)
