# output/figures/titled

The same figures as `../main/`, `../supplementary/` and `../exploratory/`, with their figure titles, subtitles and panel headings, for browsing and slides.

## What
`main/`, `supplementary/` and `exploratory/`, with the same file names as their titleless counterparts. External panels (figures 2 and 7) are the same file in both.

## How it is generated
The namer copies `../canonical/titled/`; see `../README.md`. Both variants come from one render call with `titled` set to `True` or `False`.

## Command
`just name-outputs`

## Inputs
`../canonical/titled/<source>.png`.

## Outputs
`<section>/<name>.png`.

## Provenance
`../manifest.csv`, `variant = titled`.

## Caveats
Not for the manuscript. Do not edit here: the namer overwrites and prunes this directory.
