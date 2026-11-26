# Local raw inputs

Source: Papkou et al. 2023, "A rugged yet easily navigable fitness landscape", Science,
doi:10.1126/science.adh3860. The data are Zenodo record 13837982
(doi:10.5281/zenodo.13837982), licensed CC-BY-4.0. These files are derived from that record
and are credited to Papkou et al.; the licence allows redistribution with that credit. The
files are ignored by Git (full size), and rebuilt from the record by the recipes below.

## Processed data

`data/processed/` (ignored by Git) holds every finished result; it is written by `just all`. No
`data/processed/` deposit exists yet, so `just fetch-data` (which would download it from the record
named by `zenodo_record` in `config/data.toml`, now empty) stays unavailable until one is published.
`just figures` renders from it and computes nothing; `just check-manifest` compares the folder with
`MANIFEST.tsv` (path, bytes, sha256).
The BarSeq counts the growth-rate stage reads come from barseqfit (`just barseq-export`).

## Rebuild

```sh
export PAPKOU2023RUGGED_DATA=/path/to/store   # the store barseqfit also reads
just fetch-papkou2023    # download the record's zip, extract only the files used, verify md5
just make-raw            # rebuild the files below; runs the shape check
```

`just fetch-papkou2023` extracts four files from `code_analyses_figures.zip` into
`$PAPKOU2023RUGGED_DATA/zenodo/in_data/` (`sequence_reads_counts.RDS`, `fitness_data_wt.rds`,
`fitness_diff_wald_test_data.RDS`, `graph_largest_component.rds`; paths and md5 sums in
`config/data.toml`). The 13 authors' scripts that barseqfit's `validate-papkou2023` reads are
not in that record. They come from the authors' pipeline repository, archived on Zenodo as
record 8229097 (DOI 10.5281/zenodo.8229097, CC-BY-4.0, `ampseq_nextflow-master.zip`, 72,704 B);
`just fetch-papkou2023` downloads that zip (resumable, md5 checked against the Zenodo API and
`config/data.toml`) and extracts only those 13 files into `$PAPKOU2023RUGGED_DATA/ampseq-nextflow/`,
each verified against `config/papkou2023_authors_scripts.md5`. A local checkout given with
`--authors-scripts-from` (or `PAPKOU2023_AMPSEQ_DIR`) overrides the download. A store script with a
different md5 is moved to `<name>.orig` before the verified copy is written. Credit: Papkou et al.
(Science 2023) for the data record and the authors of the pipeline archive, both CC-BY-4.0.

`just make-raw` reads the RDS files in R (`scripts/r/rds_to_interim.R`, needs the `igraph`
package), then transforms in polars. Rules:

- `papkou2023_fitness_wt.parquet`: from `fitness_data_wt.rds`; the WT row (`SV == GCCGATCTC`)
  is dropped; `dna_from` is `GCCGATCTC`, `dna_to` is `SV`, `s = m`, `se = m_se`,
  `pval = m_p.value`; row order is the RDS order.
- `papkou2023_fitness_diff.parquet`: from `fitness_diff_wald_test_data.RDS`; an edge with
  `m < 0` has `from` and `to` swapped; `s = |m|`; row order is the RDS order.
- `aa_from`, `aa_to`: standard-genetic-code translation codon by codon; stops TAA, TAG, TGA
  are `%`, `^`, `&`.
- `papkou2023_graph.graphml`: `igraph::write_graph(format = "graphml")` of
  `graph_largest_component.rds`.
- `papkou2023_peaks.csv`: one column `dna`; the nodes of that directed graph with out-degree 0,
  sorted.

## Expected shapes

The standing check run by `just make-raw` (counts live in `config/data.toml`, `[raw_shape]`):

| File | Check |
| --- | --- |
| `papkou2023_fitness_wt.parquet` | 261,332 rows; columns `dna_from, dna_to, aa_from, aa_to, s, se, pval` (four strings, three Float64) |
| `papkou2023_fitness_diff.parquet` | 3,518,696 rows; same columns and dtypes |
| `papkou2023_graph.graphml` | 135,178 nodes, 324,044 edges |
| `papkou2023_peaks.csv` | 514 peaks (stop-codon peaks included); column `dna` |

## One-off check against an existing copy

To confirm a rebuild equals a previous copy by value (not by bytes):

```sh
just make-raw data/interim/raw-rebuilt
just check-raw-against data/raw data/interim/raw-rebuilt
```

It compares the Parquet frames with `polars.testing.assert_frame_equal`, the peak sets, and
the GraphML nodes and edges with their attributes. A mismatch is a stop-and-report.

## Earlier copy

The copies in use before the rebuild came from a clean Topographer checkout from before
`topographer v0.5.0` (2026-10-03 and 2026-10-04):

| File | Size | SHA-256 |
| --- | ---: | --- |
| `papkou2023_fitness_wt.parquet` | 8.1 MB | `a3f3fd421b7ad7d05c7ae870c190f2d0b5a4e998bfffb3b8f76652a37882e4b4` |
| `papkou2023_fitness_diff.parquet` | 111 MB | `44fcd723d360a73df9187fc66e66d2d28d0b9b5985984d0be8f0d56e3c8ffd5d` |
| `papkou2023_graph.graphml` | 145 MB | `8a83925aec4aeb08eeeeb446e79ffaffad11efe81f4d912dc9be77d189204dad` |
| `papkou2023_peaks.csv` | 5.0 KB | `362726e64ce215d2f0dfffaefe43d7a824224de06c4083e937c07aaac57d01a5` |
