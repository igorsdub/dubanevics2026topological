"""Title flag: publication headings drop, titled headings stay."""

import altair as alt

from dubanevics2026topological.figures._title_flag import figure_heading, without_headings
from dubanevics2026topological.figures.registry import FIGURE_SPECS
from dubanevics2026topological.output_paths import figure_path


def test_titled_keeps_heading_and_subtitle():
    heading = figure_heading("Peaks", ["10,000 draws"], titled=True)
    assert heading == {
        "text": "Peaks",
        "anchor": "start",
        "subtitle": ["10,000 draws"],
    }


def test_publication_variant_has_no_heading():
    assert figure_heading("Peaks", ["10,000 draws"], titled=False) is None


# -- the headingless variant of a finished chart --------------------------------------


def _panel(title=None):
    chart = (
        alt.Chart(alt.Data(values=[{"x": 0, "y": 0}, {"x": 1, "y": 1}]))
        .mark_point()
        .encode(x=alt.X("x:Q", title="Fitness"), y=alt.Y("y:Q", title="Count"))
    )
    return chart if title is None else chart.properties(title=title)


def test_without_headings_drops_only_the_root_heading_and_keeps_panel_and_row_titles():
    row = alt.hconcat(
        _panel(alt.TitleParams("Panel A", subtitle=["note"])), _panel("Panel B")
    ).properties(title=alt.TitleParams("Row heading", anchor="start"))
    figure = alt.vconcat(row, _panel()).properties(
        title=alt.TitleParams("Figure heading", anchor="start")
    )
    result, removed = without_headings(figure)
    spec = result.to_dict()
    assert removed == 1
    assert "title" not in spec
    first = spec["vconcat"][0]
    assert first["title"]["text"] == "Row heading"
    panels = first["hconcat"]
    assert panels[0]["title"]["text"] == "Panel A"
    assert panels[0]["title"]["subtitle"] == ["note"]
    assert panels[1]["title"] == "Panel B"
    # axis titles stay
    assert panels[0]["encoding"]["x"]["title"] == "Fitness"
    assert panels[0]["encoding"]["y"]["title"] == "Count"
    # the input is not changed: it is the titled twin
    assert figure.to_dict()["title"]["text"] == "Figure heading"


def test_without_headings_keeps_notes_at_the_bottom_left_and_right():
    note = alt.TitleParams("Gray: outside population span", orient="bottom")
    figure = alt.vconcat(_panel("Heading")).properties(title=note)
    result, removed = without_headings(figure)
    assert removed == 0
    assert result.to_dict()["title"]["text"] == "Gray: outside population span"


def test_without_headings_counts_zero_for_a_figure_without_headings():
    result, removed = without_headings(_panel())
    assert removed == 0
    assert result.to_dict() == _panel().to_dict()


def test_without_headings_never_raises():
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot copy")

    assert without_headings(Unclonable()) is None
    result, removed = without_headings(object())
    assert removed == 0


def test_every_registered_figure_has_an_untitled_and_a_titled_path(tmp_path):
    assert FIGURE_SPECS
    seen = set()
    for spec in FIGURE_SPECS:
        for variant in sorted(spec.variants):
            stem = spec.figure_id if variant == "annotated" else f"{spec.figure_id}_vanilla"
            plain = figure_path(tmp_path, stem)
            titled = figure_path(tmp_path, stem, titled=True)
            assert plain.parent == tmp_path / "figures" / "canonical"
            assert titled.parent == tmp_path / "figures" / "canonical" / "titled"
            assert plain.name == titled.name
            assert plain != titled
            seen.add((plain, titled))
    assert len(seen) >= len(FIGURE_SPECS)
