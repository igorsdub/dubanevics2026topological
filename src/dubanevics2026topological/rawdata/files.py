"""Checksums, resumable downloads, zip-member extraction and the data config.

Standard library only. Network access goes through an injectable ``opener`` so that
tests never touch the network.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import tomllib
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_DATA_CONFIG = "FITNESSTREE_DATA_CONFIG"
CHUNK = 1 << 20

Opener = Callable[[urllib.request.Request], Any]


class Md5Mismatch(RuntimeError):
    """A file's md5 differs from the expected value."""


def load_config(path: Path | None = None) -> dict[str, Any]:
    """Read config/data.toml (or $FITNESSTREE_DATA_CONFIG, or ``path``)."""
    if path is None:
        env = os.environ.get(ENV_DATA_CONFIG)
        path = Path(env) if env else REPO_ROOT / "config" / "data.toml"
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def md5_file(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def matches_md5(path: Path, expected: str) -> bool:
    """True when ``path`` exists and its md5 equals ``expected``."""
    return path.is_file() and md5_file(path) == expected.lower()


def require_md5(path: Path, expected: str) -> None:
    actual = md5_file(path)
    if actual != expected.lower():
        raise Md5Mismatch(f"{path}: md5 {actual}, expected {expected}")


def parse_md5_lines(text: str) -> list[tuple[str, str]]:
    """Parse ``md5sum`` / ``md5 -r`` lines into (md5, path); a leading ``./`` is dropped."""
    entries = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rest = line.partition(" ")
        rest = rest.strip().lstrip("*")
        entries.append((digest.lower(), rest.removeprefix("./")))
    return entries


def default_opener(request: urllib.request.Request) -> Any:
    return urllib.request.urlopen(request, timeout=60)


def fetch_json(url: str, opener: Opener = default_opener) -> Any:
    with opener(urllib.request.Request(url)) as response:
        return json.loads(response.read())


@dataclass(frozen=True)
class RemoteFile:
    key: str
    size: int
    md5: str
    url: str


def record_files(
    record: str | int, opener: Opener = default_opener
) -> dict[str, RemoteFile]:
    """File list of a Zenodo record from its API: key -> RemoteFile (md5 from the API)."""
    data = fetch_json(f"https://zenodo.org/api/records/{record}", opener)
    out = {}
    for entry in data["files"]:
        checksum = entry["checksum"]
        algorithm, _, value = checksum.partition(":")
        if algorithm != "md5":
            raise ValueError(f"{entry['key']}: unexpected checksum {checksum!r}")
        out[entry["key"]] = RemoteFile(
            entry["key"], int(entry["size"]), value, entry["links"]["self"]
        )
    return out


def _open_range(url: str, offset: int, opener: Opener) -> Any:
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    return opener(urllib.request.Request(url, headers=headers))


def download(
    url: str,
    dest: Path,
    expected_md5: str,
    opener: Opener = default_opener,
    progress: Callable[[int], None] | None = None,
) -> str:
    """Download ``url`` to ``dest`` and verify the md5. Returns ``"cached"`` or ``"downloaded"``.

    Skips when ``dest`` already matches. Resumes a partial ``dest + '.part'`` with a Range
    request (restarting when the server ignores it). A bad download is deleted and raises.
    """
    if matches_md5(dest, expected_md5):
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    offset = part.stat().st_size if part.exists() else 0
    try:
        response = _open_range(url, offset, opener)
    except urllib.error.HTTPError as error:
        if error.code != 416 or not offset:
            raise
        # Range not satisfiable: the partial file is already whole or stale.
        if matches_md5(part, expected_md5):
            os.replace(part, dest)
            return "downloaded"
        part.unlink()
        offset = 0
        response = _open_range(url, 0, opener)
    with response:
        status = getattr(response, "status", 200)
        if offset and status != 206:
            offset = 0
        with open(part, "ab" if offset else "wb") as out:
            while block := response.read(CHUNK):
                out.write(block)
                if progress:
                    progress(len(block))
    actual = md5_file(part)
    if actual != expected_md5.lower():
        part.unlink()
        raise Md5Mismatch(f"{url}: md5 {actual}, expected {expected_md5}")
    os.replace(part, dest)
    return "downloaded"


def _copy_verified(src: BinaryIO, dest: Path, expected_md5: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp")
    digest = hashlib.md5(usedforsecurity=False)
    with open(tmp, "wb") as out:
        while block := src.read(CHUNK):
            digest.update(block)
            out.write(block)
    if digest.hexdigest() != expected_md5.lower():
        tmp.unlink()
        raise Md5Mismatch(f"{dest}: md5 {digest.hexdigest()}, expected {expected_md5}")
    os.replace(tmp, dest)


def extract_members(
    zip_path: Path, members: Iterable[tuple[str, Path, str]]
) -> dict[Path, str]:
    """Stream the named zip members to their destinations, verifying each md5.

    ``members`` holds (path inside the zip, destination, expected md5). A destination that
    already matches is skipped. Returns destination -> ``"cached"`` or ``"extracted"``.
    """
    status = {}
    with zipfile.ZipFile(zip_path) as archive:
        names = set(archive.namelist())
        for in_zip, dest, expected in members:
            if matches_md5(dest, expected):
                status[dest] = "cached"
                continue
            if in_zip not in names:
                raise KeyError(f"{in_zip} is not in {zip_path.name}")
            with archive.open(in_zip) as handle:
                _copy_verified(handle, dest, expected)
            status[dest] = "extracted"
    return status


MAX_UNPACK_FILES = 100_000
MAX_UNPACK_BYTES = 64 << 30


def unpack_zip(
    zip_path: Path,
    dest: Path,
    *,
    max_files: int = MAX_UNPACK_FILES,
    max_bytes: int = MAX_UNPACK_BYTES,
) -> list[Path]:
    """Extract a whole zip under ``dest``, refusing members that escape it.

    A zip with more than ``max_files`` members, or whose members add up to more than
    ``max_bytes`` (by their declared sizes, and again by the bytes actually written),
    is refused before or during extraction.
    """
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    written = []
    with zipfile.ZipFile(zip_path) as archive:
        members = archive.infolist()
        if len(members) > max_files:
            raise ValueError(f"{zip_path.name} has {len(members)} members, over {max_files}")
        declared = sum(info.file_size for info in members)
        if declared > max_bytes:
            raise ValueError(f"{zip_path.name} unpacks to {declared} bytes, over {max_bytes}")
        total = 0
        for info in members:
            target = (dest / info.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"unsafe zip member {info.filename!r}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as handle, open(target, "wb") as out:
                while block := handle.read(CHUNK):
                    total += len(block)
                    if total > max_bytes:
                        raise ValueError(
                            f"{zip_path.name} wrote more than {max_bytes} bytes; stopped"
                        )
                    out.write(block)
            written.append(target)
    return written


def update_manifest(store: Path, files: Iterable[Path]) -> Path:
    """Add or refresh ``md5  ./relative/path`` lines in the store's MANIFEST.md5.

    Existing lines for other files are kept, so a larger local store is never clobbered.
    """
    manifest = store / "MANIFEST.md5"
    lines = {}
    if manifest.exists():
        for digest, path in parse_md5_lines(manifest.read_text()):
            lines[path] = digest
    for file in files:
        lines[file.resolve().relative_to(store.resolve()).as_posix()] = md5_file(file)
    manifest.write_text("".join(f"{lines[path]} ./{path}\n" for path in sorted(lines)))
    return manifest


def route_destination(name: str, destinations: list[dict[str, str]]) -> str:
    """First destination whose glob matches ``name``."""
    for rule in destinations:
        if fnmatch.fnmatch(name, rule["match"]):
            return rule["dest"]
    return "data/processed"
