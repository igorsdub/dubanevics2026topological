#!/usr/bin/env Rscript
# Read the three Papkou et al. 2023 RDS files and hand them to Python as plain files.
#
# Usage: Rscript scripts/r/rds_to_interim.R <in_data dir> <output dir>
#
# Writes into <output dir>:
#   fitness_wt.csv    SV, m, m_se, m_p.value        (RDS row order)
#   fitness_diff.csv  from, to, m, m_se, m_p.value  (RDS row order)
#   graph.graphml     igraph::write_graph(format = "graphml") of graph_largest_component.rds
#                     (as scripts/convert_rds_to_graphml.R in papkou2023rugged-article, b0310fa)
#   graph_nodes.csv   name, out_degree              (graph vertex order)
# Numbers are written with 17 significant digits so that doubles round-trip exactly;
# NA is written as an empty field.

suppressPackageStartupMessages(library(igraph))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("usage: rds_to_interim.R <in_data dir> <output dir>", call. = FALSE)
in_dir <- args[[1]]
out_dir <- args[[2]]
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

num <- function(x) ifelse(is.na(x), "", sprintf("%.17g", as.numeric(x)))
write_table <- function(df, file) {
  utils::write.table(df, file, sep = ",", row.names = FALSE, quote = TRUE,
                     na = "", qmethod = "double", fileEncoding = "UTF-8")
}
need <- function(df, cols, what) {
  absent <- setdiff(cols, names(df))
  if (length(absent)) stop(what, " lacks columns: ", paste(absent, collapse = ", "), call. = FALSE)
}

message("Reading fitness_data_wt.rds")
wt <- base::readRDS(file.path(in_dir, "fitness_data_wt.rds"))
need(wt, c("SV", "m", "m_se", "m_p.value"), "fitness_data_wt.rds")
write_table(data.frame(SV = as.character(wt$SV), m = num(wt$m), m_se = num(wt$m_se),
                       m_p.value = num(wt$m_p.value), stringsAsFactors = FALSE),
            file.path(out_dir, "fitness_wt.csv"))
rm(wt)

message("Reading fitness_diff_wald_test_data.RDS")
diff <- base::readRDS(file.path(in_dir, "fitness_diff_wald_test_data.RDS"))
need(diff, c("from", "to", "m", "m_se", "m_p.value"), "fitness_diff_wald_test_data.RDS")
write_table(data.frame(from = as.character(diff$from), to = as.character(diff$to),
                       m = num(diff$m), m_se = num(diff$m_se), m_p.value = num(diff$m_p.value),
                       stringsAsFactors = FALSE),
            file.path(out_dir, "fitness_diff.csv"))
rm(diff)

message("Reading graph_largest_component.rds")
G <- base::readRDS(file.path(in_dir, "graph_largest_component.rds"))
if (!igraph::is_directed(G)) stop("the Papkou graph is expected to be directed", call. = FALSE)
message("Saving graph to GraphML...")
igraph::write_graph(G, file = file.path(out_dir, "graph.graphml"), format = "graphml")
write_table(data.frame(name = igraph::V(G)$name,
                       out_degree = as.integer(igraph::degree(G, mode = "out")),
                       stringsAsFactors = FALSE),
            file.path(out_dir, "graph_nodes.csv"))
message("Done.")
