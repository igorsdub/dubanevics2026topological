# output/figures/external

Panels made outside this repository and exported here by hand.

## What
| File | Manuscript | Made in |
| :--- | :--- | :--- |
| `workflow.png` | Figure 2: workflow and tree annotations | Google Slides |
| `network_bootstrap.png` | Figure 7: network bootstrap of persistence diagrams (after Fasy et al. 2014) | Google Slides |

## How it is generated
Export the slide as PNG and save it under the `desc` its entry has in `config/manuscript.toml`. The namer copies it to `../main/` and `../titled/main/`.

## Command
None in this repository; `just name-outputs` picks the export up and reports a missing one.

## Inputs
The Google Slides deck (link and slide number to be recorded here).

## Outputs
`<desc>.png`.

## Provenance
`../manifest.csv` holds the SHA-256 of each export; record the export date and slide revision here when replacing a file.

## Caveats
Not reproducible from code. An export goes stale whenever the slide changes.
