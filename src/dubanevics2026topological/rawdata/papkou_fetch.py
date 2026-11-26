"""``just fetch-papkou2023``: fill the Papkou et al. 2023 store from Zenodo.

Downloads the record's ``code_analyses_figures.zip`` (resumable, md5-checked), extracts only
the files named in config/data.toml into the store layout under ``$PAPKOU2023RUGGED_DATA``,
verifies every md5 and refreshes the store's MANIFEST.md5. The authors' ampseq-nextflow
scripts are not in that record; they come from the authors' pipeline archive (Zenodo record
8229097, CC-BY-4.0), or from a local checkout when one is given (it takes precedence).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

from dubanevics2026topological.rawdata import files as fs


def store_path(config: dict[str, Any], override: str | None = None) -> Path:
    cfg = config["papkou2023"]
    value = override or os.environ.get(cfg["store_env"])
    if not value:
        raise SystemExit(
            f"Set {cfg['store_env']} to the Papkou store directory (or pass --store)."
        )
    return Path(value).expanduser()


def member_targets(config: dict[str, Any], store: Path) -> list[tuple[str, Path, str]]:
    cfg = config["papkou2023"]
    root = store / cfg["zip_root_in_store"]
    return [(m["in_zip"], root / m["in_zip"], m["md5"]) for m in cfg["members"]]


def _set_aside(dest: Path) -> Path:
    """Move a differing existing file to ``<name>.orig`` (``.orig.1`` ... if taken)."""
    target = dest.with_name(dest.name + ".orig")
    n = 0
    while target.exists():
        n += 1
        target = dest.with_name(f"{dest.name}.orig.{n}")
    os.replace(dest, target)
    return target


def _fetch_scripts_zip(
    config: dict[str, Any], store: Path, opener: fs.Opener, log
) -> Path:
    """Download the authors' pipeline archive (Zenodo) into the store's downloads/."""
    cfg = config["papkou2023"]["authors_scripts"]
    remote = fs.record_files(cfg["record"], opener)[cfg["zip_name"]]
    if (remote.md5, remote.size) != (cfg["zip_md5"], cfg["zip_size"]):
        raise SystemExit(
            f"Zenodo lists {cfg['zip_name']} as {remote.size} B md5 {remote.md5}; "
            f"config/data.toml expects {cfg['zip_size']} B md5 {cfg['zip_md5']}."
        )
    zip_path = store / "downloads" / cfg["zip_name"]
    state = fs.download(remote.url, zip_path, remote.md5, opener)
    log(f"{cfg['zip_name']} (record {cfg['record']}): {state}, md5 {remote.md5}")
    return zip_path


def copy_authors_scripts(
    config: dict[str, Any],
    store: Path,
    source: Path | None,
    repo_root: Path = fs.REPO_ROOT,
    opener: fs.Opener = fs.default_opener,
    log=print,
) -> tuple[list[Path], list[str]]:
    """Place the authors' scripts in the store, verifying each md5.

    A script already in the store with the pinned md5 is kept. Otherwise it comes from the
    local checkout ``source`` when that has it, else from the authors' pipeline archive on
    Zenodo (downloaded once, md5 from the API checked against the config). An existing store
    file with a different md5 is moved to ``<name>.orig`` first, never overwritten.

    Returns (verified destinations, relative paths still missing or unverified).
    """
    cfg = config["papkou2023"]["authors_scripts"]
    entries = fs.parse_md5_lines((repo_root / cfg["md5_file"]).read_text())
    dest_root = store / cfg["store_dir"]
    done: list[Path] = []
    missing: list[str] = []
    archive: zipfile.ZipFile | None = None
    try:
        for digest, rel in entries:
            dest = dest_root / rel
            if fs.matches_md5(dest, digest):
                done.append(dest)
                continue
            origin = source / rel if source else None
            from_local = origin is not None and origin.is_file()
            if not from_local and archive is None:
                archive = zipfile.ZipFile(
                    _fetch_scripts_zip(config, store, opener, log)
                )
            member = cfg["zip_prefix"] + rel
            if from_local:
                fs.require_md5(origin, digest)
                where = f"local checkout {source}"
            elif member in archive.namelist():
                where = f"Zenodo record {cfg['record']}"
            else:
                missing.append(rel)
                continue
            if dest.exists():
                old = _set_aside(dest)
                log(
                    f"{rel}: store copy differs from the pinned md5; moved to "
                    f"{old.relative_to(store)}"
                )
            if from_local:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(origin, dest)
            else:
                with archive.open(member) as handle:
                    fs._copy_verified(handle, dest, digest)
            log(f"{rel}: written from {where}, md5 {digest}")
            done.append(dest)
    finally:
        if archive is not None:
            archive.close()
    return done, missing


def fetch(
    config: dict[str, Any],
    store: Path,
    *,
    authors_source: Path | None = None,
    keep_zip: bool = True,
    repo_root: Path = fs.REPO_ROOT,
    opener: fs.Opener = fs.default_opener,
    log=print,
) -> int:
    cfg = config["papkou2023"]
    targets = member_targets(config, store)
    zip_path = store / "downloads" / cfg["zip_name"]
    store.mkdir(parents=True, exist_ok=True)

    if all(fs.matches_md5(dest, md5) for _, dest, md5 in targets):
        log("All four store members already present with matching md5; no download.")
        extracted = [dest for _, dest, _ in targets]
    else:
        remote = fs.record_files(cfg["record"], opener)[cfg["zip_name"]]
        if (remote.md5, remote.size) != (cfg["zip_md5"], cfg["zip_size"]):
            raise SystemExit(
                f"Zenodo lists {cfg['zip_name']} as {remote.size} B md5 {remote.md5}; "
                f"config/data.toml expects {cfg['zip_size']} B md5 {cfg['zip_md5']}."
            )
        state = fs.download(remote.url, zip_path, remote.md5, opener)
        log(f"{cfg['zip_name']}: {state}, md5 {remote.md5}")
        status = fs.extract_members(zip_path, targets)
        for dest, how in status.items():
            log(f"{how:9s} {dest.relative_to(store)}")
        extracted = list(status)
        if not keep_zip:
            zip_path.unlink()

    scripts, missing = copy_authors_scripts(
        config, store, authors_source, repo_root, opener, log
    )
    log(
        f"Authors' scripts verified in the store: {len(scripts)} of {len(scripts) + len(missing)}."
    )
    if missing:
        log(
            f"WARNING: {len(missing)} authors' scripts could not be placed. Missing: "
            + ", ".join(missing)
        )
    manifest = fs.update_manifest(store, [*extracted, *scripts])
    log(f"Updated {manifest}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--store", help="Store directory (default $PAPKOU2023RUGGED_DATA)."
    )
    parser.add_argument(
        "--authors-scripts-from",
        help="Local ampseq-nextflow checkout (overrides the Zenodo archive).",
    )
    parser.add_argument(
        "--delete-zip", action="store_true", help="Remove the zip afterwards."
    )
    args = parser.parse_args(argv)
    config = fs.load_config()
    source = args.authors_scripts_from or os.environ.get(
        config["papkou2023"]["authors_scripts"]["source_env"]
    )
    return fetch(
        config,
        store_path(config, args.store),
        authors_source=Path(source).expanduser() if source else None,
        keep_zip=not args.delete_zip,
    )


if __name__ == "__main__":
    sys.exit(main())
