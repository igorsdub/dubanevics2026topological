# Graph descriptive statistics

| metric | Full | Papkou | Topographer | Topographer + FDR |
| --- | --- | --- | --- | --- |
| nodes | 261333 | 135178 | 102575 | 98131 |
| edges | 3518696 | 324044 | 243507 | 233444 |
| % functional | – | 13.3298317774 | 13.2732147209 | 13.1986833926 |
| mean degree (undirected) | 26.9288302664 | 4.79433043838 | 4.74788203753 | 4.75780334451 |
| mean in/out degree (directed uphill) | 13.4644151332 | 2.39716521919 | 2.37394101877 | 2.37890167225 |


Manuscript Table 1. Four graphs, one column each. Every cell is computed from
that column's network. An en dash means the quantity is undefined for that graph.
`summarize_graph` still computes fitness, threshold and functional-count metrics;
they are not rows of this table.

| Column | Graph |
| --- | --- |
| Full | Every sequence in the dataset. No functional threshold is applied. |
| Papkou | Papkou threshold, then the largest connected component of the edges that touch a vertex at or above that threshold. A vertex is functional when its raw fitness is at least the threshold. |
| Topographer | The Topographer point-estimate graph (median-reflected threshold) (`threshold=reflected_fdr=none`): threshold `T = m + (m − g)` from the Full-graph raw fitness, vertices with raw fitness at least `T` are functional, same edge and largest-component rule. No FDR classes. |
| Topographer + FDR | Built (Benjamini–Yekutieli, nominal q = 0.05) by loading variant `threshold=reflected_fdr=by_q=0.05` and selecting with `prepare_selected_landscape`. `% functional` follows that variant's saved `functional` class, not a relabeling of the Topographer column. |

## Rows

- **nodes**: vertices in the graph.
- **edges**: undirected edges in the graph.
- **% functional**: `100 * functional count / number of nodes`. Undefined for Full. For Papkou and Topographer, functional is raw fitness at or above the threshold. For Topographer + FDR it is the saved `functional` class. Uncertain BY vertices are not functional.
- **mean degree (undirected)**: `2 * number of edges / number of nodes`.
- **mean in/out degree (directed uphill)**: each undirected edge is oriented once, from the endpoint of lower raw fitness to the endpoint of higher raw fitness (equal fitness points from the lexicographically smaller DNA identity to the larger). Mean in-degree equals mean out-degree because every edge contributes one to each sum, so the row is `number of edges / number of nodes`.
