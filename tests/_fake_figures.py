"""Test helper: route the build through a fake ``render`` callable."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

import dubanevics2026topological.build as build_module
from dubanevics2026topological.figures import registry


def install_fake_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for the exported manuscript quantity registry.

    Figures that read labels from it need its export record for their dependency
    fingerprint and its rows at render time; neither is under test in the build
    fixtures, so a constant fingerprint and the sentinel rows replace both. The
    registry-currency check ``build_figures`` runs is stubbed too; it is covered
    in test_figure_quantities.py.
    """
    from _figure_labels import label_quantities

    monkeypatch.setattr(
        "dubanevics2026topological.figures._quantities.registry_fingerprint",
        lambda output: "registry-fingerprint",
    )
    monkeypatch.setattr(build_module, "require_current_registry", lambda settings: None)
    monkeypatch.setattr(
        "dubanevics2026topological.figures._quantities.load_manuscript_quantities",
        lambda output, **kwargs: label_quantities(),
    )


def install_fake_figures(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    """Replace the build's specs with ones whose render calls ``fake.render``.

    ``fake`` exposes ``FIGURE_IDS`` and ``render(payload, figure_id,
    settings, *, variant)`` as the old module-level fake did.
    """
    install_fake_registry(monkeypatch)
    monkeypatch.setattr(
        build_module, "active_bootstrap_band_identity", lambda checkpoint: "a" * 64, raising=False
    )
    monkeypatch.setattr(
        build_module, "selection_source", lambda settings: {"rule": "mixture_quantile"}
    )
    monkeypatch.setattr(
        "dubanevics2026topological.build.saved_figure_inputs.load_saved_figure_inputs",
        lambda context, selection_identity, include_band, **kwargs: {
            "mode": "legacy_mixture",
            "product_ids": {"selection": selection_identity},
        },
    )
    specs = []
    for spec in registry.FIGURE_SPECS:
        if spec.figure_id not in fake.FIGURE_IDS:
            continue

        def render(inputs, settings, variant, spec=spec):
            payload = registry.compose_payload(spec, inputs, settings)
            return fake.render(payload, spec.figure_id, settings, variant=variant)

        specs.append(replace(spec, render=render))
    monkeypatch.setattr(build_module, "FIGURE_SPECS", tuple(specs))


def write_renderer_sources(figure_dir: Path, text: str = "# renderer\n") -> None:
    """Write every file the figure fingerprints hash into ``figure_dir``.

    The set is derived from the registry (each figure's renderer file plus the
    shared modules), so fixtures cannot drift from the real fingerprint logic.
    """
    figure_dir.mkdir(parents=True, exist_ok=True)
    names = set(registry.SHARED_RENDERER_FILES)
    for spec in registry.FIGURE_SPECS:
        names.update(registry.renderer_files(spec))
    for name in sorted(names):
        (figure_dir / name).write_text(text, encoding="utf-8")
