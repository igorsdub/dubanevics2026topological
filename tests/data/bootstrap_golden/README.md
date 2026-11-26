# Golden bootstrap tables

Output of the PREVIOUS Topographer's `bootstrap_terminal_cooccurrence` (the development revision
preceding `topographer v0.5.0`, before the chunked API and the peak-to-leaf
renames), on a 7-vertex path network (`DNA = "abcdefg"`, raw fitness
`[-0.1, 0.5, -0.2, 0.4, -0.15, 0.6, 0.1]`, se 0.05), 30 draws, seed 7, core threshold 0.95,
split tree, nominal hills `b = {b, a}`, `d = {d}`, `f = {f, g}`. `observed_*` is that
landscape, `flat_*` the same graph with every raw fitness 0.0 and the same reference tree.
Column names are the old ones (`peak_frequency`, `peak_count`); the test renames them.

`tests/test_bootstrap_chunks.py::test_the_chunked_path_reproduces_the_previous_topographers_tables`
runs the chunked path (chunk size 7) and requires the same tables, exactly.

Regenerate (isolated environment, never the project one):

```sh
uv run --no-project \
  --with "topographer[workflows] @ git+file://<topographer checkout>@<revision preceding v0.5.0>" \
  python tests/data/bootstrap_golden/generate.py
```
