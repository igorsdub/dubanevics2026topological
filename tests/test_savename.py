"""``savename``: readable, parameter-named labels for checkpoint versions (ADR 0013).

``savename(settings, name_keys, *, execution_keys=())`` is a pure function: it reads
the named settings out of a stage's declared settings and writes them as one label
such as ``landscape=full_tree=split``. The label is for people. It is never part of
a fingerprint and never decides what a version holds (``test_produce_or_load.py``
checks the first, ``test_checkpoint_versions.py`` the second).
"""

from __future__ import annotations

import copy
import itertools
import math
import random

import pytest

from dubanevics2026topological.analysis.checkpoint import (
    parse_version_name,
    version_name,
)

EXECUTION = ("n_jobs", "chunk_size")


def savename(*args, **kwargs):
    """Call ``produce_or_load.savename``, imported when a test runs.

    The verb is not implemented yet (ADR 0013, D2). Importing it here, not at the top
    of the module, keeps collection working: each test fails on the missing name.
    """
    from dubanevics2026topological.analysis.produce_or_load import savename as verb

    return verb(*args, **kwargs)


def test_the_label_is_the_sorted_alias_value_pairs_joined_by_underscores() -> None:
    settings = {"tree": {"kind": "split"}, "dataset": {"landscape": "full"}}
    name_keys = {"tree": "tree.kind", "landscape": "dataset.landscape"}
    assert savename(settings, name_keys) == "landscape=full_tree=split"


def test_pairs_are_sorted_as_whole_strings() -> None:
    settings = {"v": 1}
    # ``a1=1`` sorts before ``a=1`` because the digit '1' precedes '=' in ASCII;
    # the rule is the sorted pair strings, whatever the aliases look like.
    name_keys = {"b": "v", "a": "v", "a1": "v", "ab": "v"}
    assert savename(settings, name_keys) == "a1=1_a=1_ab=1_b=1"
    assert savename(settings, name_keys) == "_".join(
        sorted(f"{alias}=1" for alias in name_keys)
    )


def test_the_label_does_not_depend_on_dict_order() -> None:
    settings = {"a": {"x": 1, "y": 2.5}, "b": {"z": "s", "w": True}, "c": None}
    name_keys = {"p": "a.x", "q": "a.y", "r": "b.z", "s": "b.w", "t": "c"}
    expected = savename(settings, name_keys)
    generator = random.Random(7)
    for _ in range(25):
        shuffled_keys = list(name_keys.items())
        generator.shuffle(shuffled_keys)
        inner_a = list(settings["a"].items())
        inner_b = list(settings["b"].items())
        generator.shuffle(inner_a)
        generator.shuffle(inner_b)
        outer = [("a", dict(inner_a)), ("b", dict(inner_b)), ("c", None)]
        generator.shuffle(outer)
        assert savename(dict(outer), dict(shuffled_keys)) == expected


def test_no_aliases_give_an_empty_label_and_read_nothing() -> None:
    assert savename({"a": 1}, {}) == ""
    # A stage without aliases never fails on the shape of its settings.
    assert savename({"a": {1, 2}}, {}) == ""


def test_a_dotted_path_walks_nested_settings() -> None:
    settings = {"scientific": {"config": {"n_replicates": 10000, "seed": 7}}}
    name_keys = {
        "draws": "scientific.config.n_replicates",
        "seed": "scientific.config.seed",
    }
    assert savename(settings, name_keys) == "draws=10000_seed=7"


def test_two_aliases_may_name_the_same_setting() -> None:
    assert savename({"v": 3}, {"a": "v", "b": "v"}) == "a=3_b=3"


def test_the_settings_are_not_modified() -> None:
    settings = {"a": {"n_jobs": 4, "x": 1}, "b": [1, 2]}
    before = copy.deepcopy(settings)
    savename(settings, {"x": "a.x"}, execution_keys=EXECUTION)
    assert settings == before


@pytest.mark.parametrize(
    ("value", "written"),
    [
        (True, "true"),
        (False, "false"),
        (0, "0"),
        (10000, "10000"),
        (-7, "-7"),
        (10**30, "1" + "0" * 30),
        (None, "none"),
        # Floats: three significant digits, written with ``repr``.
        (-0.283918357340333, "-0.284"),
        (0.05, "0.05"),
        (0.95, "0.95"),
        (1e-06, "1e-06"),
        (0.123456, "0.123"),
        (0.0005, "0.0005"),
        (100.0, "100.0"),
        (3.0, "3.0"),
        (1234.5, "1230.0"),
        (1e22, "1e+22"),
        (-0.0, "-0.0"),
        # Strings: bytes outside [A-Za-z0-9._+-] are percent-encoded.
        ("split", "split"),
        ("study_clamped", "study_clamped"),
        ("A-1.2_3+4", "A-1.2_3+4"),
        ("a b", "a%20b"),
        ("a=b", "a%3Db"),
        ("a/b", "a%2Fb"),
        ("~", "%7E"),
        ("%", "%25"),
        ("é", "%C3%A9"),
        ("", ""),
    ],
)
def test_value_encodings(value: object, written: str) -> None:
    assert savename({"v": value}, {"x": "v"}) == f"x={written}"


def test_a_bool_is_not_written_as_an_int() -> None:
    assert savename({"v": True}, {"x": "v"}) == "x=true"
    assert savename({"v": 1}, {"x": "v"}) == "x=1"


def test_an_int_and_a_float_of_equal_value_are_written_differently() -> None:
    assert savename({"v": 3}, {"x": "v"}) != savename({"v": 3.0}, {"x": "v"})


def test_rounding_to_three_digits_is_a_label_not_an_identity() -> None:
    first = savename({"v": 0.2839}, {"x": "v"})
    second = savename({"v": 0.28391}, {"x": "v"})
    assert first == second == "x=0.284"


@pytest.mark.parametrize(
    "value",
    [
        True,
        None,
        12345,
        -0.283918357340333,
        1e-06,
        "a b=c/d é",
        "study_clamped",
        "-",
        "",
    ],
)
def test_a_label_always_fits_the_version_name_grammar(value: object) -> None:
    label = savename({"v": value, "w": 2}, {"alpha": "v", "beta": "w"})
    name = version_name("0" * 64, "1" * 64, label)
    parsed = parse_version_name(name)
    assert parsed.savename == label
    assert parsed.key == "00000000" and parsed.digest == "11111111"


def test_execution_keys_are_stripped_before_the_path_is_read() -> None:
    settings = {"n_jobs": 8, "run": {"chunk_size": 50, "draws": 10}}
    name_keys = {"draws": "run.draws"}
    assert savename(settings, name_keys, execution_keys=EXECUTION) == "draws=10"
    assert savename(settings, name_keys) == "draws=10"


@pytest.mark.parametrize("path", ["n_jobs", "run.chunk_size", "run.chunk_size.deeper"])
def test_an_alias_that_crosses_an_execution_key_is_refused(path: str) -> None:
    settings = {"n_jobs": 8, "run": {"chunk_size": {"deeper": 1}}}
    with pytest.raises(ValueError, match="execution"):
        savename(settings, {"x": path}, execution_keys=EXECUTION)


def test_the_same_path_is_fine_when_it_is_not_an_execution_key() -> None:
    assert savename({"n_jobs": 8}, {"x": "n_jobs"}) == "x=8"


@pytest.mark.parametrize("path", ["missing", "a.missing", "a.x.deeper", "a.x.y.z"])
def test_a_missing_path_raises_key_error(path: str) -> None:
    with pytest.raises(KeyError):
        savename({"a": {"x": 1}}, {"k": path})


def test_a_path_through_a_list_is_missing_not_indexed() -> None:
    with pytest.raises(KeyError):
        savename({"a": [1, 2]}, {"k": "a.0"})


@pytest.mark.parametrize("leaf", [[1, 2], (1, 2), [], {"x": 1}, {}])
def test_a_list_or_dict_leaf_raises_type_error(leaf: object) -> None:
    with pytest.raises(TypeError, match="scalar"):
        savename({"a": leaf}, {"k": "a"})


@pytest.mark.parametrize("leaf", [math.nan, math.inf, -math.inf])
def test_a_non_finite_float_is_refused(leaf: float) -> None:
    with pytest.raises(ValueError):
        savename({"a": leaf}, {"k": "a"})


@pytest.mark.parametrize("leaf", [{1, 2}, b"bytes", object(), 1 + 2j])
def test_a_value_that_is_not_plain_json_data_is_refused(leaf: object) -> None:
    with pytest.raises(TypeError):
        savename({"a": leaf}, {"k": "a"})


def test_the_settings_must_have_string_keys() -> None:
    with pytest.raises(TypeError):
        savename({1: "x"}, {"k": "a"})


@pytest.mark.parametrize(
    "alias", ["", "Upper", "1abc", "has-dash", "has_underscore", "a b", "a=b", "é"]
)
def test_an_alias_must_match_lowercase_letter_then_letters_or_digits(
    alias: str,
) -> None:
    with pytest.raises(ValueError, match="alias"):
        savename({"v": 1}, {alias: "v"})


@pytest.mark.parametrize("alias", ["a", "q", "draws", "a1", "x9y"])
def test_good_aliases_are_accepted(alias: str) -> None:
    assert savename({"v": 1}, {alias: "v"}) == f"{alias}=1"


@pytest.mark.parametrize("path", ["", ".", "a.", ".a", "a..b"])
def test_a_malformed_path_is_refused(path: str) -> None:
    with pytest.raises(ValueError, match="path"):
        savename({"a": {"b": 1}}, {"k": path})


@pytest.mark.parametrize(
    "name_keys",
    [
        "landscape",
        b"landscape",
        [("a", "v")],
        ("a", "v"),
        {1: "v"},
        {"a": 1},
        {"a": None},
        {"a": ["v"]},
    ],
)
def test_name_keys_must_be_a_mapping_of_alias_to_path(name_keys: object) -> None:
    with pytest.raises(TypeError, match="name_keys"):
        savename({"v": 1}, name_keys)  # type: ignore[arg-type]


@pytest.mark.parametrize("bare", ["n_jobs", b"n_jobs", ""])
def test_a_bare_string_is_refused_for_execution_keys(bare: object) -> None:
    with pytest.raises(TypeError, match=r"execution_keys must be a collection"):
        savename({"v": 1}, {"a": "v"}, execution_keys=bare)  # type: ignore[arg-type]


def test_the_length_cap_is_one_hundred_characters() -> None:
    # The label is "a=" and the value, so 98 characters of value fill the cap.
    assert len(savename({"v": "x" * 98}, {"a": "v"})) == 100
    with pytest.raises(ValueError, match="100"):
        savename({"v": "x" * 99}, {"a": "v"})


def test_the_cap_counts_the_encoded_label_not_the_raw_value() -> None:
    # A space is written as "%20", three characters.
    assert len(savename({"v": " " * 32}, {"a": "v"})) == 2 + 96
    with pytest.raises(ValueError, match="100"):
        savename({"v": " " * 33}, {"a": "v"})


def test_the_cap_counts_joined_pairs() -> None:
    settings = {"v": "x" * 40}
    name_keys = {"a": "v", "b": "v", "c": "v"}
    with pytest.raises(ValueError, match="100"):
        savename(settings, name_keys)
    assert len(savename(settings, {"a": "v", "b": "v"})) == 2 * 42 + 1


def test_an_error_names_the_stage_when_the_caller_says_which() -> None:
    with pytest.raises(ValueError, match="'demo-stage'"):
        savename({"v": "x" * 200}, {"a": "v"}, stage="demo-stage")
    with pytest.raises(KeyError, match="demo-stage"):
        savename({}, {"a": "v"}, stage="demo-stage")
    with pytest.raises(TypeError, match="demo-stage"):
        savename({"v": [1]}, {"a": "v"}, stage="demo-stage")
    with pytest.raises(ValueError, match="demo-stage"):
        savename({"v": 1}, {"BAD": "v"}, stage="demo-stage")
    with pytest.raises(ValueError, match="demo-stage"):
        savename(
            {"n_jobs": 1}, {"a": "n_jobs"}, execution_keys=EXECUTION, stage="demo-stage"
        )


def test_every_error_names_the_alias_and_the_path() -> None:
    with pytest.raises(KeyError, match="landscape") as missing:
        savename({"a": 1}, {"landscape": "dataset.landscape"})
    assert "dataset.landscape" in str(missing.value)
    with pytest.raises(TypeError, match="landscape") as nested:
        savename({"dataset": {"landscape": [1]}}, {"landscape": "dataset.landscape"})
    assert "dataset.landscape" in str(nested.value)


def test_all_orderings_of_many_aliases_give_one_label() -> None:
    settings = {"p": 1, "q": 2, "r": 3, "s": 4}
    name_keys = {"w": "p", "x": "q", "y": "r", "z": "s"}
    labels = {
        savename(settings, dict(order))
        for order in itertools.permutations(name_keys.items())
    }
    assert labels == {"w=1_x=2_y=3_z=4"}
