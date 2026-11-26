"""The public-tree export rewrites package addresses and drops private paths."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "private" / "make_public_tree.sh"
BASH = "/bin/bash" if Path("/bin/bash").exists() else "bash"


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    if not SCRIPT.is_file():
        pytest.skip("tools/private/make_public_tree.sh is absent (public tree)")
    inside = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    if inside.returncode != 0 or Path(inside.stdout.strip()).resolve() != ROOT:
        pytest.skip("not the root of a git checkout")
    dest = tmp_path_factory.mktemp("public") / "tree"
    env = {**os.environ, "MAKE_PUBLIC_TREE_SKIP_LOCK": "1"}
    result = subprocess.run(
        [BASH, str(SCRIPT), str(dest)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return dest


def test_pyproject_names_public_package_repos(exported):
    text = (exported / "pyproject.toml").read_text()
    assert "oist/barseqfit.git" in text
    assert "oist/topographer.git" in text


@pytest.mark.parametrize("name", ["README.md", "CITATION.cff"])
def test_user_facing_files_have_no_dev_addresses(exported, name):
    text = (exported / name).read_text()
    assert "-dev" not in text
    # Built from pieces so the exported copy of this file passes the guard.
    assert "igorsdub/" + "barseqfit" not in text
    assert "igorsdub/" + "topographer" not in text


def test_agent_guidance_files_are_present(exported):
    assert (exported / "AGENTS.md").is_file()
    assert (exported / "CLAUDE.md").read_text() == "@AGENTS.md\n"


def test_private_paths_are_absent(exported):
    assert not (exported / "tools" / "private").exists()
    assert not (exported / "docs" / "adr").exists()


def test_guard_fails_when_lock_is_not_regenerated(exported, tmp_path):
    """Non-skip path: a no-op lock command leaves -dev addresses in uv.lock."""
    dest = tmp_path / "tree"
    env = {**os.environ, "MAKE_PUBLIC_TREE_LOCK_CMD": "true"}
    env.pop("MAKE_PUBLIC_TREE_SKIP_LOCK", None)
    result = subprocess.run(
        [BASH, str(SCRIPT), str(dest)],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "uv.lock" in result.stderr + result.stdout


def test_non_empty_destination_is_refused(exported, tmp_path):
    dest = tmp_path / "tree"
    dest.mkdir()
    (dest / "keep.txt").write_text("x")
    result = subprocess.run(
        [BASH, str(SCRIPT), str(dest)],
        cwd=ROOT,
        env={**os.environ, "MAKE_PUBLIC_TREE_SKIP_LOCK": "1"},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "not empty" in result.stderr
    assert (dest / "keep.txt").exists()


def test_public_gitattributes_does_not_normalise(exported):
    text = (exported / ".gitattributes").read_text()
    assert "* -text" in text.splitlines()
    assert "text=auto" not in text
    assert "eol=" not in text


def test_manifest_is_byte_identical_and_git_keeps_crlf(exported, tmp_path):
    rel = "output/tables/manifest.csv"
    shown = subprocess.run(
        ["git", "-C", str(ROOT), "show", f"HEAD:{rel}"],
        capture_output=True,
    )
    assert shown.returncode == 0, shown.stderr
    assert (exported / rel).read_bytes() == shown.stdout
    # Work on a copy so the shared export stays free of a .git directory.
    copy = tmp_path / "copy"
    shutil.copytree(exported, copy)
    env = {
        **os.environ,
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
    }
    for cmd in (["init", "-q"], ["add", "-A"]):
        done = subprocess.run(
            ["git", "-C", str(copy), *cmd], env=env, capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr
    eol = subprocess.run(
        ["git", "-C", str(copy), "ls-files", "--eol", rel],
        env=env,
        capture_output=True,
        text=True,
    )
    assert eol.stdout.split()[0] == "i/crlf", eol.stdout


def test_assistant_name_guard_fails_on_leaked_branch_name(exported, tmp_path):
    """Second guard: a branch-style name in an exported text file is refused."""
    # Built from pieces so the exported copy of this file passes the guard.
    leak = "claude" + "/some-branch"
    lock = tmp_path / "lock.sh"
    lock.write_text(
        "perl -pi -e 's/-dev//g; s#igorsdub/#oist/#g' uv.lock\n"
        f"echo '{leak}' > leaked.md\n"
    )
    env = {**os.environ, "MAKE_PUBLIC_TREE_LOCK_CMD": f"/bin/sh {lock}"}
    env.pop("MAKE_PUBLIC_TREE_SKIP_LOCK", None)
    result = subprocess.run(
        [BASH, str(SCRIPT), str(tmp_path / "tree")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "leaked.md" in result.stderr
    assert "vendor names" in result.stderr
