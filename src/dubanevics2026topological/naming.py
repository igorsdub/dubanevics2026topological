"""Copy canonical renders to manuscript-numbered names.

Rendering writes each figure once under its stable stem in
``output/figures/canonical/`` (and its titled twin in
``output/figures/canonical/titled/``). The namer reads the manuscript map
(``config/manuscript.toml``) and copies those files, never re-rendering:

- a numbered entry goes to ``<section>/fig<number>_<desc>.png`` (``main`` or
  ``supplementary``) and its titled twin to ``titled/<section>/``;
- an external entry (a Google Slides panel) is read from
  ``external/<desc>.png`` and goes to the same places;
- every canonical render no numbered entry claims goes to ``exploratory/``
  under its canonical stem (titled twins to ``titled/exploratory/``).

Files the namer wrote earlier and no longer produces are removed from those
output directories, so a renumbering leaves no stale names; ``canonical/`` and
``external/`` are only read. ``manifest.csv`` records every copy with its
SHA-256. A missing render or external export is reported, not fatal, so the
numbering stays complete while figures are still being made.

Tables follow the same scheme: ``[[table]]`` entries of the manuscript map copy
``output/tables/canonical/<source>.<ext>`` (csv, html, md and the json sidecar)
to ``output/tables/<section>/tab<number>_<desc>.<ext>`` and write
``output/tables/manifest.csv``. There is no exploratory shelf for tables.
"""

from __future__ import annotations

import csv
import hashlib
import re
import shutil
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

SECTIONS = ("main", "supplementary", "exploratory")
OUTPUT_DIRECTORIES = (
    "main",
    "supplementary",
    "exploratory",
    "titled/main",
    "titled/supplementary",
    "titled/exploratory",
)
MANIFEST_COLUMNS = ("number", "section", "desc", "source", "variant", "path", "sha256")

_DESC = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)*$")
_NUMBER = {"main": re.compile(r"^[1-9][0-9]*$"), "supplementary": re.compile(r"^S[1-9][0-9]*$")}


@dataclass(frozen=True)
class Entry:
    """One figure of the manuscript map."""

    section: str
    desc: str
    number: str | None = None
    source: str | None = None
    external: bool = False
    status: str | None = None

    @property
    def name(self) -> str | None:
        """File stem in the numbered directories, or None while unnumbered."""
        if self.number is None:
            return None
        return f"fig{self.number}_{self.desc}"


TABLE_EXTENSIONS = ("csv", "html", "md", "json")
TABLE_SECTIONS = ("main", "supplementary")


@dataclass(frozen=True)
class TableEntry:
    """One table of the manuscript map."""

    section: str
    desc: str
    number: str
    source: str | None = None
    status: str | None = None

    @property
    def name(self) -> str:
        """File stem in the numbered directories."""
        return f"tab{self.number}_{self.desc}"


@dataclass
class NamingResult:
    """What one run of the namer wrote and what it could not find."""

    rows: list[dict[str, str]] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    planned: list[str] = field(default_factory=list)
    removed: list[Path] = field(default_factory=list)


def load_manuscript_map(path: Path) -> list[Entry]:
    """Read and validate the figure entries of the manuscript map."""
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    entries = [Entry(**raw) for raw in document.get("figure", [])]
    _validate(entries)
    return entries


def load_table_map(path: Path) -> list[TableEntry]:
    """Read and validate the table entries of the manuscript map."""
    with path.open("rb") as stream:
        document = tomllib.load(stream)
    entries = [TableEntry(**raw) for raw in document.get("table", [])]
    _validate_tables(entries)
    return entries


def _validate_tables(entries: list[TableEntry]) -> None:
    numbers: set[str] = set()
    sources: set[str] = set()
    for entry in entries:
        label = entry.desc
        if entry.section not in TABLE_SECTIONS:
            raise ValueError(f"{label}: unknown table section {entry.section!r}")
        if not _DESC.match(entry.desc):
            raise ValueError(f"{label}: desc must be lowercase snake_case")
        pattern = _NUMBER[entry.section]
        if not pattern.match(entry.number):
            raise ValueError(f"{label}: number {entry.number!r} does not fit section {entry.section!r}")
        if entry.number in numbers:
            raise ValueError(f"{label}: number {entry.number} is used twice")
        numbers.add(entry.number)
        if entry.source is None and entry.status != "planned":
            raise ValueError(f"{label}: needs a source or status = 'planned'")
        if entry.source is not None:
            if entry.source in sources:
                raise ValueError(f"{label}: source {entry.source} is mapped twice")
            sources.add(entry.source)


def _validate(entries: list[Entry]) -> None:
    numbers: set[str] = set()
    sources: set[str] = set()
    for entry in entries:
        label = entry.desc
        if entry.section not in SECTIONS:
            raise ValueError(f"{label}: unknown section {entry.section!r}")
        if not _DESC.match(entry.desc):
            raise ValueError(f"{label}: desc must be lowercase snake_case")
        if entry.number is not None:
            pattern = _NUMBER.get(entry.section)
            if pattern is None or not pattern.match(entry.number):
                raise ValueError(f"{label}: number {entry.number!r} does not fit section {entry.section!r}")
            if entry.number in numbers:
                raise ValueError(f"{label}: number {entry.number} is used twice")
            numbers.add(entry.number)
        elif entry.section == "main":
            raise ValueError(f"{label}: a main figure needs a number")
        if entry.external and entry.source is not None:
            raise ValueError(f"{label}: an external figure has no source")
        if not entry.external and entry.source is None and entry.status != "planned":
            raise ValueError(f"{label}: needs a source, external = true or status = 'planned'")
        if entry.source is not None:
            if entry.source in sources:
                raise ValueError(f"{label}: source {entry.source} is mapped twice")
            sources.add(entry.source)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy(source: Path, target: Path, written: set[Path]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    written.add(target.resolve())


def name_outputs(figures_root: Path, entries: list[Entry]) -> NamingResult:
    """Copy canonical renders to their numbered and exploratory names."""
    canonical = figures_root / "canonical"
    figures_root.mkdir(parents=True, exist_ok=True)
    result = NamingResult()
    written: set[Path] = set()

    def record(entry_number: str, section: str, desc: str, source: str, variant: str, target: Path) -> None:
        result.rows.append(
            {
                "number": entry_number,
                "section": section,
                "desc": desc,
                "source": source,
                "variant": variant,
                "path": target.relative_to(figures_root).as_posix(),
                "sha256": _sha256(target),
            }
        )

    claimed: set[str] = set()
    for entry in entries:
        if entry.name is None:
            continue
        if entry.status == "planned":
            result.planned.append(entry.name)
            continue
        if entry.external:
            origin = figures_root / "external" / f"{entry.desc}.png"
            if not origin.is_file():
                result.missing.append(origin.relative_to(figures_root).as_posix())
                continue
            variants = [("titleless", origin), ("titled", origin)]
            source_label = f"external/{entry.desc}.png"
        else:
            claimed.add(entry.source)
            variants = [
                ("titleless", canonical / f"{entry.source}.png"),
                ("titled", canonical / "titled" / f"{entry.source}.png"),
            ]
            source_label = entry.source
        for variant, origin in variants:
            prefix = Path("titled") if variant == "titled" else Path()
            target = figures_root / prefix / entry.section / f"{entry.name}.png"
            if not origin.is_file():
                result.missing.append(origin.relative_to(figures_root).as_posix())
                continue
            _copy(origin, target, written)
            record(entry.number, entry.section, entry.desc, source_label, variant, target)

    for variant, directory in (("titleless", canonical), ("titled", canonical / "titled")):
        if not directory.is_dir():
            continue
        prefix = Path("titled") if variant == "titled" else Path()
        for origin in sorted(directory.glob("*.png")):
            if origin.stem in claimed:
                continue
            target = figures_root / prefix / "exploratory" / origin.name
            _copy(origin, target, written)
            record("", "exploratory", "", origin.stem, variant, target)

    for relative in OUTPUT_DIRECTORIES:
        directory = figures_root / relative
        if not directory.is_dir():
            continue
        for stale in sorted(directory.glob("*.png")):
            if stale.resolve() not in written:
                stale.unlink()
                result.removed.append(stale)

    with (figures_root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(result.rows)
    return result


def name_tables(tables_root: Path, entries: list[TableEntry]) -> NamingResult:
    """Copy canonical table exports to their numbered names."""
    canonical = tables_root / "canonical"
    tables_root.mkdir(parents=True, exist_ok=True)
    result = NamingResult()
    written: set[Path] = set()
    for entry in entries:
        if entry.status == "planned":
            result.planned.append(entry.name)
            continue
        for extension in TABLE_EXTENSIONS:
            origin = canonical / f"{entry.source}.{extension}"
            if not origin.is_file():
                result.missing.append(origin.relative_to(tables_root).as_posix())
                continue
            target = tables_root / entry.section / f"{entry.name}.{extension}"
            _copy(origin, target, written)
            result.rows.append(
                {
                    "number": entry.number,
                    "section": entry.section,
                    "desc": entry.desc,
                    "source": str(entry.source),
                    "variant": extension,
                    "path": target.relative_to(tables_root).as_posix(),
                    "sha256": _sha256(target),
                }
            )
    for section in TABLE_SECTIONS:
        directory = tables_root / section
        if not directory.is_dir():
            continue
        for stale in sorted(directory.glob("tab*")):
            if stale.is_file() and stale.suffix[1:] in TABLE_EXTENSIONS and stale.resolve() not in written:
                stale.unlink()
                result.removed.append(stale)
    with (tables_root / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(result.rows)
    return result


def main() -> None:
    """Run the namer on the repository's output tree and report gaps."""
    root = Path(__file__).resolve().parents[2]
    entries = load_manuscript_map(root / "config" / "manuscript.toml")
    result = name_outputs(root / "output" / "figures", entries)
    print(f"wrote {len(result.rows)} files, removed {len(result.removed)} stale files")
    for name in result.planned:
        print(f"planned, not built: {name}")
    for path in result.missing:
        print(f"warning: missing: {path}")
    table_result = name_tables(root / "output" / "tables", load_table_map(root / "config" / "manuscript.toml"))
    print(f"wrote {len(table_result.rows)} table files, removed {len(table_result.removed)} stale files")
    for name in table_result.planned:
        print(f"planned, not built: {name}")
    for path in table_result.missing:
        print(f"warning: missing: tables/{path}")


if __name__ == "__main__":
    main()
