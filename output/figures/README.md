# output/figures

Every figure of the manuscript and every exploratory render, by stable identifier and by manuscript number.

## What
| Directory | Content |
| :--- | :--- |
| `canonical/` | Every render once, under its stable stem; `canonical/titled/` holds the titled twin |
| `main/` | Main-text figures, `fig<N>_<desc>.png`, titleless |
| `supplementary/` | Supplementary figures, `figS<N>_<desc>.png`, titleless |
| `titled/` | The same names with figure titles, subtitles and panel headings, for browsing and slides |
| `exploratory/` | Renders the manuscript does not use, under their canonical stems |
| `external/` | Panels made outside this repository (Google Slides), exported by hand |
| `slides/` | Slide layouts written by the build |
| `manifest.csv` | Every numbered and exploratory copy: number, section, desc, source, variant, path, sha256 |

Git tracks only the canonical PNGs the manuscript uses (Figures 1, 3–6, 9 and S1–S14; see `canonical/README.md`) and the READMEs; Figures 2, 7 and 8 are external and supplied separately; the other directories and `canonical/titled/` are local.

The manuscript numbering lives only in `config/manuscript.toml`.

## How it is generated
Two steps. The render step (`build figures`) writes `canonical/` from saved stage products, one render record per figure and variant. The namer (`dubanevics2026topological.naming`) copies `canonical/` and `external/` into `main/`, `supplementary/`, `titled/` and `exploratory/`, removes copies it no longer produces, and writes `manifest.csv`. Renumbering is an edit of `config/manuscript.toml` and a rerun of the namer; nothing is re-rendered.

## Command
```
just figures              # render every manuscript figure from data/processed/, then name-outputs
just figures-exploratory  # render every figure outside the default selection (build figures --opt-in --keep-going), then name-outputs
just name-outputs         # namer alone
```

## Inputs
The folders of `data/processed/` (`just all` writes them, `just fetch-data` downloads them) and the stage outputs restored from it under `output/`; scientific settings from `config/analysis.toml`, presentation from `config/figures.toml`.

## Outputs
PNG only for now. File names use underscores; directory names are single words.

## Provenance
Render records (fingerprint of every input checkpoint) sit beside the canonical renders; `manifest.csv` holds the SHA-256 of every copy.

## Caveats
Figures 2, 7 and 8 come from Google Slides and are only as current as their last export. Every built manuscript figure is a default registry figure, so `just figures` renders all of them; Figure 6 is `peak_uncertainty`. Supplementary figures S1–S14 are numbered in `config/manuscript.toml`; S14 is `graph_codon_entropy`; renders it does not number are opt-in (`just figures-exploratory`) and stay in `exploratory/`.
