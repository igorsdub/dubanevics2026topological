"""Version primitives of ``analysis/checkpoint.py`` (ADR 0013, Phase 3, step 3.1.2).

A checkpoint slot ``<slot>`` becomes a relative link to one immutable version
directory ``<slot>.versions/<name>``. These tests pin the primitives that name,
digest, point at and list versions. Nothing publishes through them yet, so every
test builds its version directories by hand.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from dubanevics2026topological.analysis import checkpoint, produce_or_load

FINGERPRINT = "3f9c2a1b" + "0" * 56
OTHER_FINGERPRINT = "91be04d2" + "1" * 56
DIGEST = "7d0e4c55" + "a" * 56
OTHER_DIGEST = "44e2f9a1" + "b" * 56


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _digest_of(lines: list[str]) -> str:
    text = "".join(sorted(lines))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _make_version(link: Path, name: str) -> Path:
    """Create ``<slot>.versions/<name>`` beside ``link`` and return it."""
    directory = link.with_name(link.name + checkpoint.VERSIONS_SUFFIX) / name
    directory.mkdir(parents=True)
    (directory / "data.txt").write_text(name, encoding="utf-8")
    return directory


def _name(fingerprint: str = FINGERPRINT, digest: str = DIGEST, savename=None) -> str:
    return checkpoint.version_name(fingerprint, digest, savename)


# --- constants and the shared ignore list ---------------------------------


def test_versions_suffix_is_dot_versions() -> None:
    assert checkpoint.VERSIONS_SUFFIX == ".versions"


def test_ignored_names_live_in_checkpoint_and_the_helper_re_exports_them() -> None:
    assert checkpoint.IGNORED_NAMES == frozenset({".DS_Store"})
    assert produce_or_load.IGNORED_NAMES is checkpoint.IGNORED_NAMES


# --- fingerprint_tag ------------------------------------------------------


def test_fingerprint_tag_of_a_64_hex_string_is_its_prefix() -> None:
    assert checkpoint.fingerprint_tag(FINGERPRINT) == "3f9c2a1b"
    assert checkpoint.fingerprint_tag(FINGERPRINT, 12) == FINGERPRINT[:12]
    assert checkpoint.fingerprint_tag(FINGERPRINT, 64) == FINGERPRINT


def test_fingerprint_tag_of_a_dict_hashes_its_canonical_json() -> None:
    value = {"b": [1, 2], "a": {"z": None, "y": 0.5}}
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"))
    full = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert checkpoint.fingerprint_tag(value) == full[:8]
    assert checkpoint.fingerprint_tag(value, 64) == full


def test_fingerprint_tag_of_a_dict_ignores_key_order() -> None:
    first = {"a": 1, "b": {"c": 2, "d": 3}}
    second = {"b": {"d": 3, "c": 2}, "a": 1}
    assert checkpoint.fingerprint_tag(first) == checkpoint.fingerprint_tag(second)
    assert checkpoint.fingerprint_tag(first) != checkpoint.fingerprint_tag({"a": 2})


@pytest.mark.parametrize("text", ["3F9C2A1B" + "0" * 56, "g" * 64, "abc", "a" * 63])
def test_fingerprint_tag_hashes_a_string_that_is_not_64_lowercase_hex(
    text: str,
) -> None:
    expected = hashlib.sha256(json.dumps(text).encode("utf-8")).hexdigest()
    assert checkpoint.fingerprint_tag(text) == expected[:8]


def test_fingerprint_tag_of_a_64_hex_string_is_not_its_own_hash() -> None:
    assert checkpoint.fingerprint_tag(FINGERPRINT) != checkpoint.fingerprint_tag(
        {"fingerprint": FINGERPRINT}
    )


@pytest.mark.parametrize("length", [0, -1, 65])
def test_fingerprint_tag_rejects_a_length_outside_1_to_64(length: int) -> None:
    with pytest.raises(ValueError, match="length"):
        checkpoint.fingerprint_tag(FINGERPRINT, length)


def test_fingerprint_tag_refuses_a_missing_fingerprint() -> None:
    with pytest.raises(ValueError, match="fingerprint"):
        checkpoint.fingerprint_tag(None)


def test_fingerprint_tag_refuses_data_that_is_not_json() -> None:
    with pytest.raises((TypeError, ValueError)):
        checkpoint.fingerprint_tag({"x": float("nan")})
    with pytest.raises((TypeError, ValueError)):
        checkpoint.fingerprint_tag({"x": object()})


# --- version_name and parse_version_name ---------------------------------


def test_version_name_without_a_savename_is_key_and_digest() -> None:
    assert checkpoint.version_name(FINGERPRINT, DIGEST) == "3f9c2a1b-7d0e4c55"


def test_version_name_puts_the_savename_first() -> None:
    name = checkpoint.version_name(FINGERPRINT, DIGEST, "draws=10000_seed=20261019")
    assert name == "draws=10000_seed=20261019-3f9c2a1b-7d0e4c55"


def test_an_empty_savename_means_none() -> None:
    assert checkpoint.version_name(FINGERPRINT, DIGEST, "") == "3f9c2a1b-7d0e4c55"


def test_version_name_takes_the_key_tag_of_a_dict_fingerprint() -> None:
    fingerprint = {"generation": "g1"}
    tag = checkpoint.fingerprint_tag(fingerprint)
    assert checkpoint.version_name(fingerprint, DIGEST) == f"{tag}-7d0e4c55"


@pytest.mark.parametrize(
    "savename",
    [
        None,
        "landscape=full_tree=split",
        "draws=10000_seed=20261019",
        "q=-0.284_cutoff=1e-06",  # '-' and '_' inside the savename
        "study_clamped=true",
        "label=a%20b_x=none",
        "a.b+c=1",
        "x-",  # a trailing '-' still round-trips
        "x--y",
    ],
)
def test_name_and_parse_round_trip(savename: str | None) -> None:
    name = checkpoint.version_name(FINGERPRINT, DIGEST, savename)
    parsed = checkpoint.parse_version_name(name)
    assert parsed == checkpoint.VersionName(savename, "3f9c2a1b", "7d0e4c55")


def test_a_savename_that_looks_like_hex_is_still_a_savename() -> None:
    name = f"abcdef01-{_name()}"
    assert checkpoint.parse_version_name(name) == checkpoint.VersionName(
        "abcdef01", "3f9c2a1b", "7d0e4c55"
    )


def test_full_form_writes_both_tags_in_64_hex_and_round_trips() -> None:
    name = checkpoint.version_name(FINGERPRINT, DIGEST, "draws=10", full=True)
    assert name == f"draws=10-{FINGERPRINT}-{DIGEST}"
    parsed = checkpoint.parse_version_name(name)
    assert parsed == checkpoint.VersionName("draws=10", FINGERPRINT, DIGEST)
    short = checkpoint.parse_version_name(_name(savename="draws=10"))
    assert (parsed.key, parsed.digest) != (short.key, short.digest)
    assert parsed.key.startswith(short.key) and parsed.digest.startswith(short.digest)


def test_full_form_of_a_dict_fingerprint_is_its_full_canonical_digest() -> None:
    fingerprint = {"generation": "g1"}
    full = checkpoint.fingerprint_tag(fingerprint, 64)
    assert len(full) == 64
    name = checkpoint.version_name(fingerprint, DIGEST, full=True)
    assert name == f"{full}-{DIGEST}"


@pytest.mark.parametrize(
    "digest", ["", "abc", "A" * 64, "g" * 64, "a" * 63, "a" * 65, None, 7]
)
def test_version_name_rejects_a_digest_that_is_not_64_lowercase_hex(digest) -> None:
    with pytest.raises(ValueError, match="digest"):
        checkpoint.version_name(FINGERPRINT, digest)


@pytest.mark.parametrize(
    "savename",
    [".hidden", "a/b", "a b", "a\\b", "..", "-x", "x\ny", "é", "a:b", "x" * 300],
)
def test_version_name_rejects_an_unsafe_or_hidden_savename(savename: str) -> None:
    with pytest.raises(ValueError, match="savename"):
        checkpoint.version_name(FINGERPRINT, DIGEST, savename)


@pytest.mark.parametrize(
    "name",
    [
        "",
        "3f9c2a1b",  # one field
        "3f9c2a1b-7d0e4c55-",  # empty digest
        "-3f9c2a1b-7d0e4c55",  # empty savename
        "3f9c2a1b-7d0e4c5",  # short digest
        "3F9C2A1B-7D0E4C55",  # uppercase
        "3f9c2a1b-7d0e4c5g",  # not hex
        f"3f9c2a1b-{DIGEST}",  # mixed tag lengths
        f"{FINGERPRINT}-7d0e4c55",  # mixed tag lengths
        ".draws=1-3f9c2a1b-7d0e4c55",  # hidden
        ".3f9c2a1b-7d0e4c55.unusable-abc",
        "draws=1/x-3f9c2a1b-7d0e4c55",
        "draws=1 -3f9c2a1b-7d0e4c55",
        "junk",
    ],
)
def test_parse_version_name_rejects_names_outside_the_grammar(name: str) -> None:
    with pytest.raises(ValueError):
        checkpoint.parse_version_name(name)


# --- tree_digest ----------------------------------------------------------


def test_tree_digest_follows_the_documented_line_format(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"alpha")
    (tmp_path / "sub").mkdir()
    binary = bytes([0, 1])
    (tmp_path / "sub" / "b.bin").write_bytes(binary)
    lines = [
        f"{_sha(b'alpha')}  a.txt\n",
        f"{_sha(binary)}  sub/b.bin\n",
    ]
    assert checkpoint.tree_digest(tmp_path) == _digest_of(lines)


def test_tree_digest_of_an_empty_directory_is_the_digest_of_nothing(
    tmp_path: Path,
) -> None:
    assert checkpoint.tree_digest(tmp_path) == hashlib.sha256(b"").hexdigest()


def test_tree_digest_is_independent_of_creation_order(tmp_path: Path) -> None:
    files = {"a.txt": b"1", "z.txt": b"2", "d/m.txt": b"3", "d/e/f.txt": b"4"}
    first, second = tmp_path / "first", tmp_path / "second"
    for root, order in ((first, list(files)), (second, list(reversed(files)))):
        for relative in order:
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(files[relative])
    assert checkpoint.tree_digest(first) == checkpoint.tree_digest(second)


def test_tree_digest_changes_with_content_and_with_name(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"one")
    before = checkpoint.tree_digest(tmp_path)
    (tmp_path / "a.txt").write_bytes(b"two")
    changed_content = checkpoint.tree_digest(tmp_path)
    (tmp_path / "a.txt").rename(tmp_path / "b.txt")
    changed_name = checkpoint.tree_digest(tmp_path)
    assert len({before, changed_content, changed_name}) == 3


def test_tree_digest_does_not_confuse_a_path_with_its_content(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "ab").write_bytes(b"c")
    (second / "a").write_bytes(b"bc")
    assert checkpoint.tree_digest(first) != checkpoint.tree_digest(second)


def test_tree_digest_skips_ignored_names_at_any_depth(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_bytes(b"x")
    (tmp_path / "sub" / "b.txt").write_bytes(b"y")
    before = checkpoint.tree_digest(tmp_path)
    for name in checkpoint.IGNORED_NAMES:
        (tmp_path / name).write_bytes(b"finder")
        (tmp_path / "sub" / name).write_bytes(b"finder")
    assert checkpoint.tree_digest(tmp_path) == before


def test_tree_digest_records_a_symlink_without_following_it(tmp_path: Path) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"target bytes")
    root = tmp_path / "root"
    root.mkdir()
    (root / "link").symlink_to(outside)
    recorded = checkpoint.tree_digest(root)

    outside.write_bytes(b"other target bytes")  # the link is not followed
    assert checkpoint.tree_digest(root) == recorded

    other = tmp_path / "other.txt"
    other.write_bytes(b"target bytes")
    (root / "link").unlink()
    (root / "link").symlink_to(other)  # same bytes, other target string
    assert checkpoint.tree_digest(root) != recorded


def test_a_symlink_is_not_the_same_entry_as_a_file_with_its_content(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"bytes")
    linked, copied = tmp_path / "linked", tmp_path / "copied"
    linked.mkdir()
    copied.mkdir()
    (linked / "x").symlink_to(outside)
    (copied / "x").write_bytes(b"bytes")
    assert checkpoint.tree_digest(linked) != checkpoint.tree_digest(copied)


def test_tree_digest_does_not_descend_into_a_linked_directory(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "inner.txt").write_bytes(b"1")
    root = tmp_path / "root"
    root.mkdir()
    (root / "dirlink").symlink_to(outside)
    recorded = checkpoint.tree_digest(root)
    (outside / "inner.txt").write_bytes(b"2")
    (outside / "extra.txt").write_bytes(b"3")
    assert checkpoint.tree_digest(root) == recorded
    assert recorded != hashlib.sha256(b"").hexdigest()  # the link itself counts


def test_tree_digest_accepts_a_dangling_symlink(tmp_path: Path) -> None:
    (tmp_path / "dangling").symlink_to(tmp_path / "missing")
    assert checkpoint.tree_digest(tmp_path) != hashlib.sha256(b"").hexdigest()


def test_tree_digest_honours_exclude_by_exact_relative_path(tmp_path: Path) -> None:
    (tmp_path / "data.txt").write_bytes(b"data")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "provenance.json").write_bytes(b"nested")
    without_record = checkpoint.tree_digest(tmp_path)
    (tmp_path / "provenance.json").write_bytes(b"record")

    assert checkpoint.tree_digest(tmp_path) != without_record
    assert (
        checkpoint.tree_digest(tmp_path, exclude=("provenance.json",)) == without_record
    )
    # A nested file with the same name is a different path and stays counted.
    assert checkpoint.tree_digest(
        tmp_path, exclude={"provenance.json", "sub/provenance.json"}
    ) != checkpoint.tree_digest(tmp_path, exclude=("provenance.json",))


def test_tree_digest_rejects_a_bare_string_for_exclude(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="exclude"):
        checkpoint.tree_digest(tmp_path, exclude="provenance.json")


def test_tree_digest_needs_a_directory(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        checkpoint.tree_digest(tmp_path / "missing")
    (tmp_path / "file").write_bytes(b"x")
    with pytest.raises(OSError):
        checkpoint.tree_digest(tmp_path / "file")


def test_tree_digest_cannot_be_forged_by_a_file_name_holding_a_line(
    tmp_path: Path,
) -> None:
    """A name with a newline must not read as two entries."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    digest = _sha(b"x")
    (first / f"a\n{digest}  b").write_bytes(b"x")
    (second / "a").write_bytes(b"x")
    (second / "b").write_bytes(b"x")
    assert checkpoint.tree_digest(first) != checkpoint.tree_digest(second)


def test_tree_digest_matches_the_artefact_map_of_a_record(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"alpha")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"beta")
    (tmp_path / "provenance.json").write_bytes(b"{}")
    artefacts = {"a.txt": _sha(b"alpha"), "sub/b.bin": _sha(b"beta")}
    lines = [f"{digest}  {relative}\n" for relative, digest in artefacts.items()]
    assert checkpoint.tree_digest(tmp_path, exclude=("provenance.json",)) == _digest_of(
        lines
    )


# --- switch_link ----------------------------------------------------------


def test_switch_link_writes_a_relative_link_to_the_version(tmp_path: Path) -> None:
    link = tmp_path / "checkpoints" / "analysis"
    version = _make_version(link, _name())
    checkpoint.switch_link(link, version)
    assert link.is_symlink()
    assert os.readlink(link) == f"analysis.versions/{_name()}"
    assert (link / "data.txt").read_text(encoding="utf-8") == _name()
    assert link.resolve() == version.resolve()


def test_switch_link_repoints_an_existing_link(tmp_path: Path) -> None:
    link = tmp_path / "analysis"
    first = _make_version(link, _name())
    second = _make_version(link, _name(OTHER_FINGERPRINT, OTHER_DIGEST))
    checkpoint.switch_link(link, first)
    checkpoint.switch_link(link, second)
    assert os.readlink(link) == f"analysis.versions/{second.name}"
    checkpoint.switch_link(link, first)
    assert os.readlink(link) == f"analysis.versions/{first.name}"
    assert first.is_dir() and second.is_dir()  # nothing was removed


def test_switch_link_accepts_a_relative_link_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    link = Path("analysis")
    version = _make_version(link, _name())
    checkpoint.switch_link(link, version)
    assert os.readlink(link) == f"analysis.versions/{version.name}"


def test_switch_link_is_a_no_op_when_the_target_is_equal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    version = _make_version(link, _name())
    checkpoint.switch_link(link, version)
    before = os.lstat(link)

    calls: list[str] = []
    for name in ("symlink", "replace", "rename", "unlink", "remove"):
        original = getattr(os, name)

        def spy(*args, _name=name, _original=original, **kwargs):
            calls.append(_name)
            return _original(*args, **kwargs)

        monkeypatch.setattr(checkpoint.os, name, spy)
    checkpoint.switch_link(link, version)

    assert calls == []
    after = os.lstat(link)
    assert (after.st_ino, after.st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_switch_link_refuses_a_real_directory_and_leaves_it_alone(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    link.mkdir()
    (link / "legacy.txt").write_text("legacy", encoding="utf-8")
    version = _make_version(link, _name())
    with pytest.raises(RuntimeError, match="real directory|not a link|adopt"):
        checkpoint.switch_link(link, version)
    assert not link.is_symlink()
    assert (link / "legacy.txt").read_text(encoding="utf-8") == "legacy"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "analysis",
        "analysis.versions",
    ]


def test_switch_link_refuses_a_regular_file_at_the_link_path(tmp_path: Path) -> None:
    link = tmp_path / "analysis"
    link.write_text("not a link", encoding="utf-8")
    version = _make_version(link, _name())
    with pytest.raises(RuntimeError):
        checkpoint.switch_link(link, version)
    assert link.read_text(encoding="utf-8") == "not a link"


@pytest.mark.parametrize(
    "where",
    [
        "other.versions/{name}",  # another slot's versions directory
        "analysis/{name}",  # not a versions directory
        "analysis.versions/sub/{name}",  # one level too deep
        "{name}",  # no versions directory at all
        "analysis.versions/junk",  # not in the name grammar
        "analysis.versions/.{name}",  # hidden
    ],
)
def test_switch_link_refuses_a_target_outside_its_own_versions_directory(
    tmp_path: Path, where: str
) -> None:
    link = tmp_path / "analysis"
    target = tmp_path / where.format(name=_name())
    target.mkdir(parents=True)
    with pytest.raises(ValueError):
        checkpoint.switch_link(link, target)
    assert not link.is_symlink()


def test_switch_link_refuses_an_existing_link_outside_its_versions_directory(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link.symlink_to("elsewhere")
    version = _make_version(link, _name())
    with pytest.raises(ValueError):
        checkpoint.switch_link(link, version)
    assert os.readlink(link) == "elsewhere"
    assert elsewhere.is_dir()


def test_switch_link_refuses_a_version_that_does_not_exist(tmp_path: Path) -> None:
    link = tmp_path / "analysis"
    missing = link.with_name("analysis.versions") / _name()
    with pytest.raises(OSError):
        checkpoint.switch_link(link, missing)
    assert not os.path.lexists(link)


def test_switch_link_never_leaves_the_link_missing_while_switching(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    versions = [
        _make_version(link, name)
        for name in (_name(), _name(OTHER_FINGERPRINT, OTHER_DIGEST))
    ]
    checkpoint.switch_link(link, versions[0])

    stop, started = threading.Event(), threading.Event()
    problems: list[str] = []
    reads = 0

    def watch() -> None:
        nonlocal reads
        while not stop.is_set():
            try:
                present = os.path.lexists(link)
                current = checkpoint.current_version(link)
                # A reader resolves once and reads from the version path.
                text = (current / "data.txt").read_text(encoding="utf-8")
            except (OSError, ValueError, TypeError) as error:
                problems.append(repr(error))
                continue
            reads += 1
            started.set()
            if not present or current not in versions or text != current.name:
                problems.append(f"{present=} {current=}")

    watcher = threading.Thread(target=watch)
    watcher.start()
    try:
        assert started.wait(timeout=10)
        for index in range(200):
            checkpoint.switch_link(link, versions[(index + 1) % 2])
    finally:
        stop.set()
        watcher.join()

    assert problems == []
    assert reads > 0


def test_switch_link_replaces_the_link_and_never_unlinks_it_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    first = _make_version(link, _name())
    second = _make_version(link, _name(OTHER_FINGERPRINT, OTHER_DIGEST))
    checkpoint.switch_link(link, first)

    replaced: list[tuple[Path, Path]] = []
    removed: list[Path] = []
    original_replace = os.replace

    def spy_replace(source, target, *args, **kwargs):
        replaced.append((Path(source), Path(target)))
        return original_replace(source, target, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "replace", spy_replace)
    for name in ("unlink", "remove", "rmdir"):
        original = getattr(os, name)

        def spy_remove(path, *args, _original=original, **kwargs):
            removed.append(Path(path))
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(checkpoint.os, name, spy_remove)

    checkpoint.switch_link(link, second)

    assert len(replaced) == 1
    source, target = replaced[0]
    assert target == link
    assert source.parent == link.parent and source.name.startswith(".analysis.")
    assert link not in removed
    assert not source.exists()  # the temporary name was consumed by the replace


def test_switch_link_syncs_the_parent_directory_after_the_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    version = _make_version(link, _name())
    events: list[str] = []
    original_replace, original_fsync = os.replace, os.fsync

    def replace(*args, **kwargs):
        events.append("replace")
        return original_replace(*args, **kwargs)

    def fsync(descriptor):
        events.append("fsync")
        return original_fsync(descriptor)

    monkeypatch.setattr(checkpoint.os, "replace", replace)
    monkeypatch.setattr(checkpoint.os, "fsync", fsync)
    checkpoint.switch_link(link, version)
    assert events == ["replace", "fsync"]


def test_a_failed_replace_keeps_the_old_link_and_leaves_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    first = _make_version(link, _name())
    second = _make_version(link, _name(OTHER_FINGERPRINT, OTHER_DIGEST))
    checkpoint.switch_link(link, first)

    def fail(*args, **kwargs):
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint.os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        checkpoint.switch_link(link, second)

    assert os.readlink(link) == f"analysis.versions/{first.name}"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "analysis",
        "analysis.versions",
    ]


# --- current_version ------------------------------------------------------


def test_current_version_returns_the_lexical_path(tmp_path: Path) -> None:
    link = tmp_path / "checkpoints" / "analysis"
    version = _make_version(link, _name())
    checkpoint.switch_link(link, version)
    assert checkpoint.current_version(link) == version


def test_current_version_does_not_resolve_a_linked_parent(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real)
    link = alias / "analysis"
    version = _make_version(link, _name())
    checkpoint.switch_link(link, version)

    found = checkpoint.current_version(link)
    assert found == alias / "analysis.versions" / version.name
    assert real not in found.parents


def test_current_version_of_a_missing_link_is_none(tmp_path: Path) -> None:
    assert checkpoint.current_version(tmp_path / "analysis") is None


def test_current_version_of_a_dangling_link_is_none(tmp_path: Path) -> None:
    link = tmp_path / "analysis"
    link.symlink_to(f"analysis.versions/{_name()}")
    assert os.path.lexists(link) and not link.exists()
    assert checkpoint.current_version(link) is None


def test_current_version_is_none_when_the_target_is_not_a_directory(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    versions = tmp_path / "analysis.versions"
    versions.mkdir()
    (versions / _name()).write_text("a file", encoding="utf-8")
    link.symlink_to(f"analysis.versions/{_name()}")
    assert checkpoint.current_version(link) is None


@pytest.mark.parametrize(
    "target",
    [
        "/abs/analysis.versions/{name}",  # absolute
        "../analysis.versions/{name}",  # leaves the directory
        "analysis.versions/../analysis.versions/{name}",  # contains '..'
        "analysis.versions/{name}/..",  # '..' at the end
        "analysis.versions/.{name}",  # hidden
        "analysis.versions/{name}.unusable-0123",  # quarantined, not grammar
        "analysis.versions/junk",  # not in the grammar
        "other.versions/{name}",  # another slot's versions
        "analysis/{name}",  # not a versions directory
        "analysis.versions/sub/{name}",  # too deep
        "analysis.versions",  # the directory itself
        "{name}",  # no versions directory
    ],
)
def test_current_version_refuses_a_target_outside_the_grammar(
    tmp_path: Path, target: str
) -> None:
    link = tmp_path / "analysis"
    link.symlink_to(target.format(name=_name()))
    with pytest.raises(ValueError):
        checkpoint.current_version(link)


def test_current_version_refuses_a_refused_target_even_when_it_exists(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    elsewhere = tmp_path / "elsewhere" / _name()
    elsewhere.mkdir(parents=True)
    link.symlink_to(elsewhere)  # absolute, and it exists
    with pytest.raises(ValueError):
        checkpoint.current_version(link)


def test_current_version_refuses_a_real_directory_at_the_link_path(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    link.mkdir()
    with pytest.raises(ValueError, match="not a link|real directory"):
        checkpoint.current_version(link)


def test_current_version_refuses_a_version_that_is_itself_a_link(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    real = tmp_path / "real"
    real.mkdir()
    versions = tmp_path / "analysis.versions"
    versions.mkdir()
    (versions / _name()).symlink_to(real)
    link.symlink_to(f"analysis.versions/{_name()}")
    with pytest.raises(ValueError):
        checkpoint.current_version(link)


def test_current_version_reads_what_switch_link_wrote(tmp_path: Path) -> None:
    link = tmp_path / "analysis"
    first = _make_version(link, _name(savename="draws=1_seed=2"))
    second = _make_version(link, _name(OTHER_FINGERPRINT, OTHER_DIGEST))
    for version in (first, second, first):
        checkpoint.switch_link(link, version)
        assert checkpoint.current_version(link) == version


def test_current_version_retries_a_transient_einval_while_a_link_is_replaced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    version = _make_version(link, _name())
    checkpoint.switch_link(link, version)
    original = os.readlink
    calls: list[int] = []

    def flaky(path, *args, **kwargs):
        calls.append(1)
        if len(calls) <= 3:
            raise OSError(errno.EINVAL, "Invalid argument", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(checkpoint.os, "readlink", flaky)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda _seconds: None)
    assert checkpoint.current_version(link) == version
    assert len(calls) == 4


def test_current_version_gives_up_on_a_persistent_einval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    checkpoint.switch_link(link, _make_version(link, _name()))
    calls: list[int] = []

    def broken(path, *args, **kwargs):
        calls.append(1)
        raise OSError(errno.EINVAL, "Invalid argument", str(path))

    monkeypatch.setattr(checkpoint.os, "readlink", broken)
    monkeypatch.setattr(checkpoint.time, "sleep", lambda _seconds: None)
    with pytest.raises(OSError) as raised:
        checkpoint.current_version(link)
    assert raised.value.errno == errno.EINVAL
    assert 1 < len(calls) <= 50


def test_current_version_does_not_retry_any_other_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    link = tmp_path / "analysis"
    checkpoint.switch_link(link, _make_version(link, _name()))
    calls: list[int] = []

    def denied(path, *args, **kwargs):
        calls.append(1)
        raise PermissionError(errno.EACCES, "Permission denied", str(path))

    monkeypatch.setattr(checkpoint.os, "readlink", denied)
    with pytest.raises(PermissionError):
        checkpoint.current_version(link)
    assert calls == [1]


# --- list_versions --------------------------------------------------------


def test_list_versions_of_a_missing_directory_is_empty(tmp_path: Path) -> None:
    assert checkpoint.list_versions(tmp_path / "analysis") == []


def test_list_versions_returns_visible_grammar_directories_sorted(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    names = [
        _name(savename="landscape=full"),
        _name(OTHER_FINGERPRINT, OTHER_DIGEST),
        _name(),
    ]
    for name in names:
        _make_version(link, name)
    assert checkpoint.list_versions(link) == [
        link.with_name("analysis.versions") / name for name in sorted(names)
    ]


def test_list_versions_skips_hidden_quarantined_and_foreign_entries(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    good = _make_version(link, _name())
    versions = good.parent
    (versions / f".{_name()}.unusable-0123456789abcdef").mkdir()  # quarantined
    (versions / ".hidden").mkdir()
    (versions / ".DS_Store").write_bytes(b"finder")
    (versions / "junk").mkdir()  # not in the grammar
    (versions / f"{_name(OTHER_FINGERPRINT, OTHER_DIGEST)}.partial").mkdir()
    (versions / _name(OTHER_FINGERPRINT, OTHER_DIGEST)).write_text(  # a file
        "not a directory", encoding="utf-8"
    )
    outside = tmp_path / "outside"
    outside.mkdir()
    (versions / _name("c" * 64, "d" * 64)).symlink_to(outside)  # a linked version
    assert checkpoint.list_versions(link) == [good]


def test_list_versions_does_not_list_the_link_or_another_slots_versions(
    tmp_path: Path,
) -> None:
    link = tmp_path / "analysis"
    mine = _make_version(link, _name())
    _make_version(tmp_path / "other", _name(OTHER_FINGERPRINT, OTHER_DIGEST))
    checkpoint.switch_link(link, mine)
    assert checkpoint.list_versions(link) == [mine]
