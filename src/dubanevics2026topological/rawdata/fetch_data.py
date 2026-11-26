"""``just fetch-data``: download our own Zenodo record and unpack it into the repository.

The record id is ``[ours] zenodo_record`` in config/data.toml (empty until the DOI is
minted). Each file's md5 comes from the record API. Archives are unpacked, other files
copied, to the destination chosen by the first matching ``[[ours.destinations]]`` glob.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

from dubanevics2026topological.rawdata import files as fs
from dubanevics2026topological.rawdata.papkou_fetch import store_path


def resolve_destination(template: str, store: Path | None, repo_root: Path) -> Path:
    if "{store}" in template:
        if store is None:
            raise SystemExit(
                "A destination needs the Papkou store: set PAPKOU2023RUGGED_DATA."
            )
        template = template.replace("{store}", str(store))
    path = Path(template).expanduser()
    return path if path.is_absolute() else repo_root / path


def check_record_key(name: str) -> str:
    """A record file key is a plain file name: no directory part, no ``..``, not hidden."""
    if not name or "/" in name or "\\" in name or ".." in name or name.startswith("."):
        raise SystemExit(f"Refusing the record file key {name!r}: not a plain file name.")
    return name


def fetch_data(
    config: dict[str, Any],
    *,
    store: Path | None,
    repo_root: Path = fs.REPO_ROOT,
    cache_dir: Path | None = None,
    opener: fs.Opener = fs.default_opener,
    log=print,
) -> int:
    record = str(config["ours"].get("zenodo_record", "")).strip()
    if not record:
        raise SystemExit(
            "No Zenodo record to fetch: [ours] zenodo_record in config/data.toml is empty "
            "(the DOI is not minted yet). Set it to the record id and rerun."
        )
    cache = cache_dir or repo_root / "data" / "interim" / "zenodo-downloads" / record
    destinations = config["ours"]["destinations"]
    remote = fs.record_files(record, opener)
    for name, item in remote.items():
        check_record_key(name)
        archive = cache / name
        state = fs.download(item.url, archive, item.md5, opener)
        log(f"{name}: {state}, md5 {item.md5}")
        dest = resolve_destination(
            fs.route_destination(name, destinations), store, repo_root
        )
        if name.endswith(".zip"):
            written = fs.unpack_zip(archive, dest)
            log(f"  unpacked {len(written)} files into {dest}")
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(archive, dest / name)
            log(f"  copied to {dest / name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store", help="Papkou store (default $PAPKOU2023RUGGED_DATA)."
    )
    args = parser.parse_args(argv)
    config = fs.load_config()
    try:
        store = store_path(config, args.store)
    except SystemExit:
        store = None
    return fetch_data(config, store=store)


if __name__ == "__main__":
    sys.exit(main())
