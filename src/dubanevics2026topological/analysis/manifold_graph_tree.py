"""Conceptual landscape, sampled graph, and split-tree sweep construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from topographer.graph import Graph, ScalarNetwork
from topographer.plotting import make_tree_layout
from topographer.topology import MergeTree, ScalarNetworkTopology

from . import load_settings

SCALAR_ATTR = "scalar"
_CRITICAL_EVENT_BY_NAME = {
    "left_peak": "maximum",
    "saddle": "split_saddle",
    "right_peak": "maximum",
    "global_minimum": "minimum",
}


@dataclass(frozen=True)
class SweepState:
    """Stable sample identities and connected components at one threshold."""

    threshold: float
    active_sample_ids: tuple[str, ...]
    components: tuple[tuple[str, ...], ...]
    entered_critical_ids: tuple[str, ...]

    @property
    def component_count(self) -> int:
        """Number of active connected components."""
        return len(self.components)


@dataclass(frozen=True)
class PersistenceEvent:
    """One entered peak with its birth and, once paired, death fitness.

    ``alive`` peaks have not yet merged into a higher peak's component and
    carry ``death=None``; the figure draws them at the current threshold.
    """

    peak_name: str
    birth: float
    death: float | None
    alive: bool
    pair_kind: str
    birth_sample_id: str
    death_sample_id: str | None


@dataclass(frozen=True)
class ManifoldGraphTreeConstruction:
    """Scientific data and public Topographer results for the illustration."""

    sample_table: pl.DataFrame
    edge_table: pl.DataFrame
    grid_table: pl.DataFrame
    thresholds: tuple[float, float, float, float, float]
    critical_ids: dict[str, str]
    graph: Graph
    tree: MergeTree
    layout: Any
    opening_threshold: float
    scientific_settings: dict[str, Any]

    def sweep_state(self, threshold: float) -> SweepState:
        """Return active sample identities and graph components at ``threshold``.

        The configured opening threshold is a sentinel: at exactly 1.0 all
        samples remain inactive, matching the source notebook's convention.
        Sample inclusion at every other threshold is scalar-inclusive.
        """
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if threshold == self.opening_threshold:
            active_ids: set[str] = set()
        else:
            active_ids = {
                row["sample_id"]
                for row in self.sample_table.iter_rows(named=True)
                if row["scalar"] >= threshold
            }

        adjacency: dict[str, set[str]] = {sample_id: set() for sample_id in active_ids}
        for source, target in self.edge_table.iter_rows():
            if source in active_ids and target in active_ids:
                adjacency[source].add(target)
                adjacency[target].add(source)

        components: list[tuple[str, ...]] = []
        remaining = set(active_ids)
        while remaining:
            seed = min(remaining)
            component = {seed}
            frontier = [seed]
            remaining.remove(seed)
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current] & remaining:
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
            components.append(tuple(sorted(component)))

        entered_critical_ids = tuple(
            sorted(
                sample_id
                for sample_id in active_ids
                if sample_id in self.critical_ids.values()
            )
        )
        return SweepState(
            threshold=threshold,
            active_sample_ids=tuple(sorted(active_ids)),
            components=tuple(sorted(components)),
            entered_critical_ids=entered_critical_ids,
        )

    def persistence_events(self, threshold: float) -> tuple[PersistenceEvent, ...]:
        """Return one event per visible peak at ``threshold``, birth descending.

        Pairs come from ``MergeTree.persistence()`` (elder-rule ordinary H0
        pairs plus the global pair closed at the global minimum, which Topographer
        labels ``pair_kind == "extended"``). A pair is visible once its birth is
        reached (``birth >= threshold``) and dead once its death is reached
        (``death >= threshold``, inclusive); otherwise the peak is alive and
        carries ``death=None``. The configured opening threshold is a
        sentinel that shows nothing, matching :meth:`sweep_state`.
        """
        threshold = float(threshold)
        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if threshold == self.opening_threshold:
            return ()
        name_by_id = {sample_id: name for name, sample_id in self.critical_ids.items()}
        pairs = self.tree.persistence().sort("birth_scalar", descending=True)
        events: list[PersistenceEvent] = []
        for pair in pairs.iter_rows(named=True):
            birth = float(pair["birth_scalar"])
            if birth < threshold:
                continue
            birth_id = str(pair["birth_label"])
            death = float(pair["death_scalar"])
            dead = death >= threshold
            events.append(
                PersistenceEvent(
                    peak_name=name_by_id.get(birth_id, birth_id),
                    birth=birth,
                    death=death if dead else None,
                    alive=not dead,
                    pair_kind=str(pair["pair_kind"]),
                    birth_sample_id=birth_id,
                    death_sample_id=str(pair["death_label"]) if dead else None,
                )
            )
        return tuple(events)


def scalar_value(x: Any, y: Any, *, settings: dict[str, Any] | None = None) -> Any:
    """Evaluate normalized conceptual fitness at one or more coordinates."""
    config = _settings(settings)
    raw = _raw_fitness(x, y, config)
    domain_min, domain_max = _normalization_bounds(config)
    return (raw - domain_min) / (domain_max - domain_min)


def build_manifold_graph_tree(
    settings: dict[str, Any] | None = None,
    *,
    horizontal_layout: str = "subtree_extremum",
) -> ManifoldGraphTreeConstruction:
    """Build the deterministic conceptual landscape, graph, split tree, and sweep."""
    config = _settings(settings)
    rows = int(config["grid_rows"])
    columns = int(config["grid_columns"])
    critical_indices = {
        str(key): int(value) for key, value in config["critical_sample_indices"].items()
    }
    sample_coordinates = _sample_coordinates(rows, columns)
    if max(critical_indices.values()) >= len(sample_coordinates):
        raise ValueError("critical sample index falls outside the configured lattice")

    left_peak = _stationary_point(config["left_gaussian_center"], config)
    saddle = _stationary_point((0.5, 0.5), config)
    right_peak = _stationary_point(config["right_gaussian_center"], config)
    sample_coordinates[critical_indices["left_peak"]] = left_peak
    sample_coordinates[critical_indices["saddle"]] = saddle
    sample_coordinates[critical_indices["right_peak"]] = right_peak
    sample_coordinates[critical_indices["global_minimum"]] = np.asarray(
        config["global_minimum_coordinate"], dtype=float
    )

    axis = np.linspace(0.0, 1.0, int(config["surface_resolution"]))
    x_grid, y_grid = np.meshgrid(axis, axis)
    grid_raw = _raw_fitness(x_grid, y_grid, config)
    maximum_raw = max(
        float(grid_raw.max()),
        float(_raw_fitness(right_peak[0], right_peak[1], config)),
    )
    minimum_raw = float(grid_raw.min())
    normalized_grid = (grid_raw - minimum_raw) / (maximum_raw - minimum_raw)
    sample_raw = _raw_fitness(
        sample_coordinates[:, 0], sample_coordinates[:, 1], config
    )
    sample_scalars = (sample_raw - minimum_raw) / (maximum_raw - minimum_raw)

    critical_index_to_name = {index: name for name, index in critical_indices.items()}
    sample_records = []
    for index, ((x, y), raw_value, scalar) in enumerate(
        zip(sample_coordinates, sample_raw, sample_scalars, strict=True)
    ):
        row, column = divmod(index, columns)
        sample_id = f"sample-r{row + 1}-c{column + 1}"
        critical_name = critical_index_to_name.get(index)
        sample_records.append(
            {
                "sample_id": sample_id,
                "x": float(x),
                "y": float(y),
                "raw_fitness": float(raw_value),
                SCALAR_ATTR: float(scalar),
                "critical_name": critical_name,
                "event": _CRITICAL_EVENT_BY_NAME.get(critical_name, "regular"),
            }
        )
    sample_table = pl.DataFrame(sample_records)
    edge_table = _grid_edges(rows, columns)
    grid_table = pl.DataFrame(
        {
            "x": x_grid.ravel(),
            "y": y_grid.ravel(),
            SCALAR_ATTR: normalized_grid.ravel(),
        }
    )

    graph = Graph.from_tables(
        sample_table,
        edge_table,
        vertex_id_col="sample_id",
        source_id_col="source",
        target_id_col="target",
        vertex_attribute_cols=("x", "y", "raw_fitness", SCALAR_ATTR, "event"),
    )
    tree = ScalarNetworkTopology(
        ScalarNetwork(graph, scalar_attr=SCALAR_ATTR)
    ).split_tree()
    layout = make_tree_layout(
        tree.augmented_graph,
        scalar_attr=SCALAR_ATTR,
        root_kind="minimum",
        node_attrs=("critical_name",),
        horizontal_layout=horizontal_layout,
    )
    critical_ids = {
        name: sample_table[critical_indices[name], "sample_id"]
        for name in critical_indices
    }
    saddle_threshold = sample_table[critical_indices["saddle"], SCALAR_ATTR]
    thresholds = (
        float(config["opening_threshold"]),
        float(config["first_peak_threshold"]),
        float(config["second_peak_threshold"]),
        float(saddle_threshold),
        float(config["complete_sweep_threshold"]),
    )
    _assert_vertical_trunk(
        layout,
        critical_ids["global_minimum"],
        critical_ids["saddle"],
    )
    return ManifoldGraphTreeConstruction(
        sample_table=sample_table,
        edge_table=edge_table,
        grid_table=grid_table,
        thresholds=thresholds,
        critical_ids=critical_ids,
        graph=graph,
        tree=tree,
        layout=layout,
        opening_threshold=float(config["opening_threshold"]),
        scientific_settings=dict(config),
    )


def _settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    if settings is None:
        settings = load_settings()["analysis"]["graph_tree_persistence"]
    elif "analysis" in settings:
        settings = settings["analysis"]["graph_tree_persistence"]
    elif "graph_tree_persistence" in settings:
        settings = settings["graph_tree_persistence"]
    return settings


def _sample_coordinates(rows: int, columns: int) -> np.ndarray:
    x_coordinates = (np.arange(columns) + 0.5) / columns
    y_coordinates = (np.arange(rows) + 0.5) / rows
    return np.asarray(
        [(x, y) for y in y_coordinates for x in x_coordinates], dtype=float
    )


def _raw_fitness(x: Any, y: Any, config: dict[str, Any]) -> Any:
    width = float(config["gaussian_width"])
    left_center = np.asarray(config["left_gaussian_center"], dtype=float)
    right_center = np.asarray(config["right_gaussian_center"], dtype=float)
    left = float(config["left_peak_amplitude"]) * np.exp(
        -((np.asarray(x) - left_center[0]) ** 2 + (np.asarray(y) - left_center[1]) ** 2)
        / (2 * width**2)
    )
    right = float(config["right_peak_amplitude"]) * np.exp(
        -(
            (np.asarray(x) - right_center[0]) ** 2
            + (np.asarray(y) - right_center[1]) ** 2
        )
        / (2 * width**2)
    )
    return left + right + float(config["fitness_baseline"])


def _gradient_and_hessian(point: np.ndarray, config: dict[str, Any]):
    centers = (
        (
            np.asarray(config["left_gaussian_center"], dtype=float),
            float(config["left_peak_amplitude"]),
        ),
        (
            np.asarray(config["right_gaussian_center"], dtype=float),
            float(config["right_peak_amplitude"]),
        ),
    )
    width = float(config["gaussian_width"])
    gradient = np.zeros(2)
    hessian = np.zeros((2, 2))
    for center, amplitude in centers:
        displacement = point - center
        value = amplitude * np.exp(-np.dot(displacement, displacement) / (2 * width**2))
        gradient -= value * displacement / width**2
        hessian += value * (
            np.outer(displacement, displacement) / width**4 - np.eye(2) / width**2
        )
    return gradient, hessian


def _stationary_point(seed: Any, config: dict[str, Any]) -> np.ndarray:
    point = np.asarray(seed, dtype=float)
    for _ in range(int(config["newton_iterations"])):
        gradient, hessian = _gradient_and_hessian(point, config)
        point = point - np.linalg.solve(hessian, gradient)
    return point


def _normalization_bounds(config: dict[str, Any]) -> tuple[float, float]:
    axis = np.linspace(0.0, 1.0, int(config["surface_resolution"]))
    x_grid, y_grid = np.meshgrid(axis, axis)
    raw_grid = _raw_fitness(x_grid, y_grid, config)
    right_peak = _stationary_point(config["right_gaussian_center"], config)
    return float(raw_grid.min()), max(
        float(raw_grid.max()),
        float(_raw_fitness(right_peak[0], right_peak[1], config)),
    )


def _grid_edges(rows: int, columns: int) -> pl.DataFrame:
    edges = []
    for row in range(rows):
        for column in range(columns):
            source = f"sample-r{row + 1}-c{column + 1}"
            if column + 1 < columns:
                edges.append(
                    {"source": source, "target": f"sample-r{row + 1}-c{column + 2}"}
                )
            if row + 1 < rows:
                edges.append(
                    {"source": source, "target": f"sample-r{row + 2}-c{column + 1}"}
                )
    return pl.DataFrame(edges)


def _assert_vertical_trunk(layout: Any, root_id: str, saddle_id: str) -> None:
    """Ensure the public tree layout keeps the root-to-saddle trunk vertical."""
    nodes = layout.node_table
    node_rows = {
        int(row["tree_node_index"]): row for row in nodes.iter_rows(named=True)
    }
    ids_to_indices = {
        row["vertex_id"]: int(row["tree_node_index"])
        for row in nodes.iter_rows(named=True)
    }
    root_index = ids_to_indices[root_id]
    saddle_index = ids_to_indices[saddle_id]
    adjacency = {index: set() for index in node_rows}
    for source, target in layout.edge_table.select(
        "source_tree_node_index", "target_tree_node_index"
    ).iter_rows():
        adjacency[int(source)].add(int(target))
        adjacency[int(target)].add(int(source))

    paths = [(root_index, [root_index])]
    visited = {root_index}
    trunk = None
    while paths:
        current, path = paths.pop(0)
        if current == saddle_index:
            trunk = path
            break
        for neighbor in adjacency[current] - visited:
            visited.add(neighbor)
            paths.append((neighbor, [*path, neighbor]))
    if trunk is None:
        raise ValueError("split-tree layout has no path from global minimum to saddle")
    x_values = [float(node_rows[index]["x"]) for index in trunk]
    if not np.allclose(x_values, x_values[0], rtol=0.0, atol=1e-12):
        raise ValueError("split-tree layout does not keep the shared trunk vertical")
