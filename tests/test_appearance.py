"""Persist figure appearance edits without losing shared settings."""


def test_saving_figure_appearance_preserves_shared_variants_and_styles(
    tmp_path, monkeypatch
):
    import tomllib

    from dubanevics2026topological.figures import appearance

    root = tmp_path
    module_path = root / "src/dubanevics2026topological/figures/appearance.py"
    module_path.parent.mkdir(parents=True)
    module_path.touch()
    config_dir = root / "config"
    config_dir.mkdir()
    config_path = config_dir / "figures.toml"
    config_path.write_text(
        'formats = ["svg", "png"]\n'
        'variants = ["vanilla", "annotated"]\n'
        "\n[rendering]\nfont_size = 15\n"
        "\n[figures.linked_tree_persistence]\n"
        'title = "Before"\n\n'
        "[figures.linked_tree_persistence.tree]\nwidth = 420\nheight = 250\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(appearance, "__file__", str(module_path))

    appearance.save_appearance_settings(
        "linked_tree_persistence", {}, {"title": "After"}
    )

    with config_path.open("rb") as stream:
        saved = tomllib.load(stream)
    assert saved["formats"] == ["svg", "png"]
    assert saved["variants"] == ["vanilla", "annotated"]
    assert saved["rendering"]["font_size"] == 15
    assert saved["figures"]["linked_tree_persistence"] == {
        "title": "After",
        "tree": {"width": 420, "height": 250},
    }


def test_preview_settings_merges_appearance_without_mutating_input():
    import copy

    from dubanevics2026topological.figures import preview_settings

    settings = {
        "png_dpi": 300,
        "figures": {
            "rendering": {"font_size": 15},
            "figures": {
                "linked_tree_persistence": {
                    "title": "Before",
                    "tree": {"width": 420, "height": 250},
                },
                "other": {"title": "Other"},
            },
        },
    }
    original = copy.deepcopy(settings)

    preview = preview_settings(
        settings,
        "linked_tree_persistence",
        {"tree": {"width": 500, "height": 250}},
    )

    assert settings == original
    assert preview["png_dpi"] == 300
    assert preview["figures"]["rendering"] == {"font_size": 15}
    assert preview["figures"]["figures"]["linked_tree_persistence"] == {
        "title": "Before",
        "tree": {"width": 500, "height": 250},
    }
    assert preview["figures"]["figures"]["other"] == {"title": "Other"}
    assert preview is not settings
    assert preview["figures"]["figures"] is not settings["figures"]["figures"]


def test_shipped_colorbar_and_hamming_settings():
    from dubanevics2026topological.analysis import load_settings

    figures = load_settings()["figures"]
    assert "count_legend_placement" not in figures["rendering"]
    assert figures["rendering"]["colorbar_top_inset"] == 8
    expected = {
        "full_tree_persistence": 0.9,
        "linked_tree_persistence": 0.9,
        "full_tree_histogram": 0.9,
        "published_tree_histogram": 0.9,
        "candidate_tree_histogram": 0.9,
        "study_tree_persistence_metrics": 0.9,
    }
    for figure_id, fraction in expected.items():
        assert figures["figures"][figure_id]["colorbar_x_fraction"] == fraction
    metrics = figures["figures"]["study_tree_persistence_metrics"]
    assert metrics["aa_hamming_color_scheme"] == "bluepurple"
