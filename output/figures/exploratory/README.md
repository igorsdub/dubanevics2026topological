# output/figures/exploratory

Renders the manuscript does not use, kept for analysis and comparison, under their canonical stems.

## What
Every render in `../canonical/` that no numbered entry of `config/manuscript.toml` claims, copied under its canonical stem (titled twins to `../titled/exploratory/`): parameter-grid cells (`tree_persistence__…`, `bootstrap__…` and the other band stems), the opt-in figures (`papkou_agreement`, `peak_rank*`, `peak_pair_*`, `graph_codon_entropy`, `candidate_fdr_leaf_swarm`, the other band leaf figures and so on) and supplementary figures that have no number yet. The files are untracked; only the numbered manuscript sources are tracked, in `../canonical/`.

## How it is generated
Copies made by the namer; see `../README.md`.

## Command
`just figures-exploratory` renders the opt-in figures into `../canonical/` and then runs the namer; `just name-outputs` is the namer alone. The namer removes only the `.png` files it no longer produces in this directory.

## Inputs
`../canonical/`.

## Outputs
`<canonical stem>.png`.

## Provenance
`../manifest.csv` (`section = exploratory`).

## Caveats
Content changes whenever the manuscript map gives a figure a number. Do not edit here.
