# Graph descriptive statistics

Four graphs, one column each. Every cell is computed from that column's network.
An en dash means the quantity is undefined for that graph.

| Column | Graph |
| --- | --- |
| Full | Every sequence in the dataset. No functional threshold is applied. |
| Published | Published functional threshold, then the largest connected component of the edges that touch a vertex at or above that threshold. A vertex is functional when its raw fitness is at least the threshold. |
| Candidate (median-reflected threshold, no FDR) | The median-reflected point-estimate graph (`threshold=reflected_fdr=none`): threshold `T = m + (m − g)` from the Full-graph raw fitness, vertices with raw fitness at least `T` are functional, same edge and largest-component rule. No FDR classes. |
| Candidate (BY q = 0.05) | Built by loading variant `threshold=reflected_fdr=by_q=0.05` and selecting with `prepare_selected_landscape`. Functional and non-functional counts follow that variant's saved classes, not a relabeling of the no-FDR Candidate. The reported threshold is the variant's threshold. |

## Rows

- **number of nodes**: vertices in the graph.
- **number of edges**: undirected edges in the graph.
- **minimum fitness**: smallest raw fitness among the vertices.
- **maximum fitness**: largest raw fitness among the vertices.
- **mean fitness**: arithmetic mean of the vertices' raw fitness.
- **median fitness**: median of the vertices' raw fitness (the mean of the two central values when the count is even).
- **functional threshold**: the threshold used to build the graph. Undefined for Full.
- **functional count**: vertices classed functional. For Published and the no-FDR Candidate that is raw fitness at or above the threshold. For the BY column it is the saved `functional` class. Undefined for Full.
- **non-functional count**: the remaining vertices under a threshold rule, or the saved `non_functional` class on the BY graph. Uncertain BY vertices are in neither count. Undefined for Full.
- **% functional**: `100 * functional count / number of nodes`. Undefined for Full.
- **mean degree (undirected)**: `2 * number of edges / number of nodes`.
- **mean in/out degree (in = out)**: each undirected edge is oriented once, from the endpoint of lower raw fitness to the endpoint of higher raw fitness (equal fitness points from the lexicographically smaller DNA identity to the larger). Mean in-degree equals mean out-degree because every edge contributes one to each sum, so the row is `number of edges / number of nodes`.
