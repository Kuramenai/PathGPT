import json
import math
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import geopandas as gpd
import networkx as nx
import numpy as np
import variables
from termcolor import cprint
from tqdm import tqdm

from evaluate_single import (
    build_edge_id_to_name,
    build_edge_id_to_uvk,
    calculate_metrics,
    edge_ids_to_road_names,
    safe_float,
)
from generate_custom_dataset import apply_poi_aware_weights, load_graph
from utils import make_dir


EdgeId = int
OdPair = Tuple[int, int]
Subgraph = Dict[int, Iterable[int]]


def extract_json_object(raw_text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return {}


def extract_ranked_contexts(raw_text: str, max_context_id: int) -> List[int]:
    raw_rankings = extract_json_object(raw_text).get("ranked_contexts", [])
    if not isinstance(raw_rankings, list):
        return []

    ranked_contexts = []
    seen = set()
    for value in raw_rankings:
        try:
            context_id = int(value)
        except (TypeError, ValueError):
            continue
        if context_id < 1 or context_id > max_context_id or context_id in seen:
            continue
        ranked_contexts.append(context_id)
        seen.add(context_id)
    return ranked_contexts


def ranked_output_name() -> str:
    return (
        f"{variables.retrieval_type}_context_{variables.place_name}_top_"
        f"{variables.number_of_docs_to_retrieve}_rank_contexts"
    )


def retrieval_metadata_name() -> str:
    return (
        f"{variables.place_name}_prompts_{variables.retrieval_type}_top_"
        f"{variables.number_of_docs_to_retrieve}_rank_contexts.json"
    )


def parse_od_pair(raw_od_pair: Any) -> Optional[OdPair]:
    if not isinstance(raw_od_pair, (list, tuple)) or len(raw_od_pair) < 2:
        return None
    try:
        return int(raw_od_pair[0]), int(raw_od_pair[1])
    except (TypeError, ValueError):
        return None


def get_ground_truth_edges(path_collection: dict) -> List[int]:
    edge_path = path_collection.get(f"{variables.path_type}_path_edges", [])
    return [int(edge_id) for edge_id in edge_path]


def task_weight_column(path_type: str) -> str:
    return {
        "fastest": "travel_time",
        "shortest": "length",
        "fuel_efficient": "fuel_cost",
        "highway_free": "fuel_cost",
        "poi_aware": "touristic_value",
        "scenic": "touristic_value",
    }.get(path_type, "length")


def build_edge_weights(edges_df: gpd.GeoDataFrame, weight_column: str) -> Dict[int, float]:
    edge_weights = {}
    for edge_id, row in edges_df.iterrows():
        edge_weights[int(edge_id)] = safe_float(row.get(weight_column, 1.0))
    return edge_weights


def build_task_edge_weights(
    edges_df: gpd.GeoDataFrame,
    edge_id_to_uvk: Dict[int, tuple],
    graph: nx.MultiDiGraph,
    weight_column: str,
) -> Tuple[Dict[int, float], dict]:
    edge_weights: Dict[int, float] = {}
    graph_hits = 0
    shapefile_hits = 0
    fallback_hits = 0
    shapefile_has_column = weight_column in edges_df.columns

    for edge_id, row in edges_df.iterrows():
        edge_id = int(edge_id)
        weight: Optional[float] = None
        source = "fallback"

        uvk = edge_id_to_uvk.get(edge_id)
        if uvk is not None:
            edge_data = graph.get_edge_data(uvk[0], uvk[1], uvk[2], default={}) or {}
            if weight_column in edge_data:
                candidate = safe_float(edge_data[weight_column])
                if candidate > 0:
                    weight = candidate
                    source = "graph"

        if weight is None and shapefile_has_column:
            candidate = safe_float(row.get(weight_column))
            if candidate > 0:
                weight = candidate
                source = "shapefile"

        if weight is None:
            weight = safe_float(row.get("length", 1.0))
            source = "shapefile" if weight_column == "length" else "fallback"

        edge_weights[edge_id] = weight
        if source == "graph":
            graph_hits += 1
        elif source == "shapefile":
            shapefile_hits += 1
        else:
            fallback_hits += 1

    return edge_weights, {
        "weight_column": weight_column,
        "graph_hits": graph_hits,
        "shapefile_hits": shapefile_hits,
        "fallback_hits": fallback_hits,
        "total_edges": len(edge_weights),
    }


def print_weight_source(weight_meta: dict, path_type: str) -> None:
    cprint(
        f"Task weight column: {weight_meta['weight_column']} (path_type={path_type})",
        "cyan",
    )
    cprint(
        "Weight sources: "
        f"graph={weight_meta['graph_hits']}, "
        f"shapefile={weight_meta['shapefile_hits']}, "
        f"length_fallback={weight_meta['fallback_hits']} / "
        f"{weight_meta['total_edges']}",
        "cyan",
    )
    if weight_meta["fallback_hits"] > 0 and path_type != "shortest":
        cprint(
            f"WARNING: {weight_meta['fallback_hits']} edges fell back to length for "
            f"{path_type} graph search.",
            "yellow",
        )
    if weight_meta["graph_hits"] == 0 and path_type not in ("shortest",):
        cprint(
            f"WARNING: no graph weights found for '{weight_meta['weight_column']}'. "
            "Re-run generate_custom_dataset or subgraph_construction.",
            "yellow",
        )


def subgraph_edge_set(subgraph: Subgraph) -> set:
    edge_ids = {int(edge_id) for edge_id in subgraph}
    for next_edges in subgraph.values():
        edge_ids.update(int(next_edge) for next_edge in next_edges)
    return edge_ids


def retrieved_union_edge_set(
    retrieved_pairs: List[OdPair],
    uncompressed_subgraphs: Dict[OdPair, Subgraph],
) -> set:
    edge_ids: set = set()
    for od_pair in retrieved_pairs:
        subgraph = uncompressed_subgraphs.get(od_pair)
        if subgraph:
            edge_ids.update(subgraph_edge_set(subgraph))
    return edge_ids


def ground_truth_in_retrieved_union(
    ground_truth_edges: List[int],
    retrieved_pairs: List[OdPair],
    uncompressed_subgraphs: Dict[OdPair, Subgraph],
) -> bool:
    if not ground_truth_edges:
        return False
    union_edges = retrieved_union_edge_set(retrieved_pairs, uncompressed_subgraphs)
    return all(int(edge_id) in union_edges for edge_id in ground_truth_edges)


def add_subgraph_to_edge_graph(
    edge_graph: nx.DiGraph,
    subgraph: Subgraph,
    edge_weights: Dict[int, float],
) -> None:
    for edge_id, next_edges in subgraph.items():
        edge_id = int(edge_id)
        edge_graph.add_node(edge_id)
        for next_edge in next_edges:
            next_edge = int(next_edge)
            edge_graph.add_edge(edge_id, next_edge, weight=edge_weights.get(next_edge, 1.0))


def build_corridor_edge_graph(
    od_pairs: Iterable[OdPair],
    uncompressed_subgraphs: Dict[OdPair, Subgraph],
    edge_weights: Dict[int, float],
) -> nx.DiGraph:
    edge_graph = nx.DiGraph()
    for od_pair in od_pairs:
        subgraph = uncompressed_subgraphs.get(od_pair)
        if not subgraph:
            continue
        add_subgraph_to_edge_graph(edge_graph, subgraph, edge_weights)
    return edge_graph


def build_full_edge_transition_graph(
    edges_df: gpd.GeoDataFrame,
    edge_weights: Dict[int, float],
) -> nx.DiGraph:
    outgoing_by_node = defaultdict(list)
    row_by_edge_id = {}

    for edge_id, row in edges_df.iterrows():
        edge_id = int(edge_id)
        row_by_edge_id[edge_id] = row
        outgoing_by_node[row.u].append(edge_id)

    edge_graph = nx.DiGraph()
    edge_graph.add_nodes_from(row_by_edge_id)
    for edge_id, row in row_by_edge_id.items():
        for next_edge in outgoing_by_node.get(row.v, []):
            edge_graph.add_edge(edge_id, next_edge, weight=edge_weights.get(next_edge, 1.0))
    return edge_graph


def add_edge_route_to_graph(
    edge_graph: nx.DiGraph,
    full_edge_graph: nx.DiGraph,
    edge_route: List[int],
) -> None:
    if not edge_route:
        return

    for edge_id in edge_route:
        edge_graph.add_node(int(edge_id))

    for prev_edge, next_edge in zip(edge_route, edge_route[1:]):
        prev_edge = int(prev_edge)
        next_edge = int(next_edge)
        edge_graph.add_node(next_edge)
        if full_edge_graph.has_edge(prev_edge, next_edge):
            edge_graph.add_edge(prev_edge, next_edge, **full_edge_graph.edges[prev_edge, next_edge])


def shortest_path_to_targets(
    full_edge_graph: nx.DiGraph,
    source_edge: Optional[int],
    target_edges: set,
) -> List[int]:
    if source_edge is None or not target_edges:
        return []

    source_edge = int(source_edge)
    if source_edge not in full_edge_graph:
        return []
    if source_edge in target_edges:
        return [source_edge]

    try:
        distances, paths = nx.single_source_dijkstra(full_edge_graph, source_edge, weight="weight")
    except nx.NodeNotFound:
        return []
    best_target = None
    best_distance = float("inf")
    for target_edge in target_edges:
        target_edge = int(target_edge)
        distance = distances.get(target_edge)
        if distance is not None and distance < best_distance:
            best_distance = distance
            best_target = target_edge

    if best_target is None:
        return []
    return [int(edge_id) for edge_id in paths[best_target]]


def shortest_path_from_targets_to_dest(
    full_edge_graph: nx.DiGraph,
    source_edges: set,
    dest_edge: Optional[int],
) -> List[int]:
    if dest_edge is None or not source_edges:
        return []

    dest_edge = int(dest_edge)
    if dest_edge in source_edges:
        return [dest_edge]

    best_route: Optional[List[int]] = None
    best_cost = float("inf")
    for source_edge in source_edges:
        route = search_edge_graph(full_edge_graph, int(source_edge), dest_edge)
        if not route:
            continue
        cost = sum(full_edge_graph[u][v]["weight"] for u, v in zip(route, route[1:]))
        if cost < best_cost:
            best_cost = cost
            best_route = route

    return [int(edge_id) for edge_id in best_route] if best_route else []


def build_bridged_corridor_graph(
    corridor_graph: nx.DiGraph,
    full_edge_graph: nx.DiGraph,
    start_edge: Optional[int],
    dest_edge: Optional[int],
) -> nx.DiGraph:
    # Attach query start/dest to the retrieved corridor via full-network shortest connectors.
    # Only the original corridor can be used as the bridge target/source; connector edges
    # must not become new exit sources, otherwise this degenerates into full-graph search.
    if start_edge is None or dest_edge is None:
        return corridor_graph.copy()

    bridged = corridor_graph.copy()
    original_corridor_nodes = {int(node) for node in bridged.nodes}

    entry_route = shortest_path_to_targets(full_edge_graph, start_edge, original_corridor_nodes)
    if entry_route:
        add_edge_route_to_graph(bridged, full_edge_graph, entry_route)
    else:
        bridged.add_node(int(start_edge))

    if int(start_edge) in bridged:
        reachable_nodes = nx.descendants(bridged, int(start_edge)) | {int(start_edge)}
        exit_source_nodes = original_corridor_nodes & reachable_nodes
    else:
        exit_source_nodes = set()

    exit_route = shortest_path_from_targets_to_dest(
        full_edge_graph,
        exit_source_nodes,
        dest_edge,
    )
    if exit_route:
        add_edge_route_to_graph(bridged, full_edge_graph, exit_route)
    else:
        bridged.add_node(int(dest_edge))

    return bridged


def search_edge_graph(
    edge_graph: nx.DiGraph,
    start_edge: Optional[int],
    dest_edge: Optional[int],
) -> List[int]:
    if start_edge is None or dest_edge is None:
        return []

    start_edge = int(start_edge)
    dest_edge = int(dest_edge)
    if start_edge == dest_edge:
        return [start_edge] if start_edge in edge_graph else []

    if start_edge not in edge_graph or dest_edge not in edge_graph:
        return []

    try:
        return [
            int(edge_id) for edge_id in nx.shortest_path(edge_graph, start_edge, dest_edge, weight="weight")
        ]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def check_directed_edge_connectivity(
    edge_route: List[int],
    edge_id_to_uvk: Dict[int, tuple],
) -> bool:
    if len(edge_route) <= 1:
        return bool(edge_route)

    for prev_edge, next_edge in zip(edge_route, edge_route[1:]):
        prev_uvk = edge_id_to_uvk.get(int(prev_edge))
        next_uvk = edge_id_to_uvk.get(int(next_edge))
        if not prev_uvk or not next_uvk:
            return False
        if prev_uvk[1] != next_uvk[0]:
            return False
    return True


def retrieved_od_pairs(query_metadata: dict) -> List[OdPair]:
    od_pairs = []
    for raw_od_pair in query_metadata.get("retrieved_od_pairs", []):
        od_pair = parse_od_pair(raw_od_pair)
        if od_pair is not None:
            od_pairs.append(od_pair)
    return od_pairs


def ranked_od_pairs(raw_text: str, query_metadata: dict) -> Tuple[List[int], List[OdPair]]:
    retrieved_pairs = retrieved_od_pairs(query_metadata)
    ranked_context_ids = extract_ranked_contexts(raw_text, len(retrieved_pairs))
    od_pairs = [retrieved_pairs[context_id - 1] for context_id in ranked_context_ids]
    return ranked_context_ids, od_pairs


def has_ranked_contexts_json(raw_text: str) -> bool:
    return isinstance(extract_json_object(raw_text).get("ranked_contexts"), list)


def search_first_feasible_ranked_corridor(
    ranked_pairs: List[OdPair],
    uncompressed_subgraphs: Dict[OdPair, Subgraph],
    edge_weights: Dict[int, float],
    start_edge: Optional[int],
    dest_edge: Optional[int],
) -> Tuple[List[int], Optional[OdPair]]:
    for od_pair in ranked_pairs:
        edge_graph = build_corridor_edge_graph([od_pair], uncompressed_subgraphs, edge_weights)
        edge_route = search_edge_graph(edge_graph, start_edge, dest_edge)
        if edge_route:
            return edge_route, od_pair
    return [], None


def corridor_route(
    od_pair: OdPair,
    uncompressed_subgraphs: Dict[OdPair, Subgraph],
    edge_weights: Dict[int, float],
    start_edge: Optional[int],
    dest_edge: Optional[int],
) -> List[int]:
    edge_graph = build_corridor_edge_graph([od_pair], uncompressed_subgraphs, edge_weights)
    return search_edge_graph(edge_graph, start_edge, dest_edge)


def evaluate_corridor(
    od_pair: OdPair,
    uncompressed_subgraphs: Dict[OdPair, Subgraph],
    edge_weights: Dict[int, float],
    start_edge: Optional[int],
    dest_edge: Optional[int],
    ground_truth_edges: List[int],
) -> Tuple[bool, float]:
    edge_route = corridor_route(
        od_pair, uncompressed_subgraphs, edge_weights, start_edge, dest_edge
    )
    if not edge_route:
        return False, 0.0
    _, recall = calculate_metrics(edge_route, ground_truth_edges)
    return True, recall


def compute_jaccard_similarity(
    path1: List[int], path2: List[int], edge_lengths: Dict[int, float]
) -> float:
    # ponytail: copied from filter_custom_dataset to avoid heavy module imports
    s1, s2 = set(path1), set(path2)
    intersection_length = sum(edge_lengths.get(e, 0.0) for e in s1.intersection(s2))
    union_length = sum(edge_lengths.get(e, 0.0) for e in s1.union(s2))
    return intersection_length / union_length if union_length > 0 else 0.0


def normalized_edit_similarity(path1: List[int], path2: List[int]) -> float:
    if not path1 and not path2:
        return 1.0
    if not path1 or not path2:
        return 0.0

    n, m = len(path1), len(path2)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if path1[i - 1] == path2[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return 1.0 - prev[m] / max(n, m)


def route_cost(edge_route: List[int], edge_weights: Dict[int, float]) -> float:
    return sum(edge_weights.get(int(edge_id), 0.0) for edge_id in edge_route)


def route_in_search_graph(edge_route: List[int], edge_graph: Optional[nx.DiGraph]) -> bool:
    if not edge_route or edge_graph is None:
        return False

    normalized_route = [int(edge_id) for edge_id in edge_route]
    if not all(edge_id in edge_graph for edge_id in normalized_route):
        return False
    return all(
        edge_graph.has_edge(prev_edge, next_edge)
        for prev_edge, next_edge in zip(normalized_route, normalized_route[1:])
    )


def dcg(relevances: List[float], k: int) -> float:
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(relevances[:k], start=1))


def ndcg_at_k(ranked_relevances: List[float], all_relevances: List[float], k: int) -> float:
    dcg_val = dcg(ranked_relevances, k)
    idcg_val = dcg(sorted(all_relevances, reverse=True), k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def mrr(feasible_flags: List[bool]) -> float:
    for rank, is_feasible in enumerate(feasible_flags, start=1):
        if is_feasible:
            return 1.0 / rank
    return 0.0


def success_at_k(feasible_flags: List[bool], k: int) -> float:
    return 1.0 if any(feasible_flags[:k]) else 0.0


@dataclass
class StrategyResult:
    edge_route: List[int] = field(default_factory=list)
    source: str = "none"
    selected_od_pair: Optional[OdPair] = None
    search_graph: Optional[nx.DiGraph] = None


@dataclass
class StrategyStats:
    edge_precisions: List[float] = field(default_factory=list)
    edge_recalls: List[float] = field(default_factory=list)
    road_precisions: List[float] = field(default_factory=list)
    road_recalls: List[float] = field(default_factory=list)
    jaccard_scores: List[float] = field(default_factory=list)
    edit_similarities: List[float] = field(default_factory=list)
    cost_ratios: List[float] = field(default_factory=list)
    generated_count: int = 0
    topology_valid_count: int = 0
    valid_in_graph_count: int = 0
    route_lengths: List[int] = field(default_factory=list)

    def add(
        self,
        edge_route: List[int],
        ground_truth_edges: List[int],
        ground_truth_route: List[str],
        edge_id_to_name: Dict[int, str],
        edge_id_to_uvk: Dict[int, tuple],
        edge_weights: Dict[int, float],
        edge_lengths: Dict[int, float],
        search_graph: Optional[nx.DiGraph],
    ) -> None:
        if edge_route:
            self.generated_count += 1
            self.route_lengths.append(len(edge_route))
            if check_directed_edge_connectivity(edge_route, edge_id_to_uvk):
                self.topology_valid_count += 1
            if route_in_search_graph(edge_route, search_graph):
                self.valid_in_graph_count += 1

        predicted_route = edge_ids_to_road_names(edge_route, edge_id_to_name)
        edge_p, edge_r = calculate_metrics(edge_route, ground_truth_edges)
        road_p, road_r = calculate_metrics(predicted_route, ground_truth_route)

        self.edge_precisions.append(edge_p)
        self.edge_recalls.append(edge_r)
        self.road_precisions.append(road_p)
        self.road_recalls.append(road_r)

        if edge_route and ground_truth_edges:
            self.jaccard_scores.append(
                compute_jaccard_similarity(edge_route, ground_truth_edges, edge_lengths)
            )
            self.edit_similarities.append(
                normalized_edit_similarity(edge_route, ground_truth_edges)
            )
            gt_cost = route_cost(ground_truth_edges, edge_weights)
            if gt_cost > 0:
                self.cost_ratios.append(route_cost(edge_route, edge_weights) / gt_cost)

    def summary(self, total_samples: int) -> dict:
        return {
            "generated": self.generated_count,
            "generated_rate": self.generated_count / total_samples if total_samples else 0.0,
            "topology_valid": self.topology_valid_count,
            "topology_valid_rate": (self.topology_valid_count / total_samples if total_samples else 0.0),
            "topology_valid_rate_on_generated": (
                self.topology_valid_count / self.generated_count if self.generated_count else 0.0
            ),
            "valid_in_graph": self.valid_in_graph_count,
            "valid_in_graph_rate": (
                self.valid_in_graph_count / total_samples if total_samples else 0.0
            ),
            "valid_in_graph_rate_on_generated": (
                self.valid_in_graph_count / self.generated_count if self.generated_count else 0.0
            ),
            "edge_precision": float(np.mean(self.edge_precisions)) if self.edge_precisions else 0.0,
            "edge_recall": float(np.mean(self.edge_recalls)) if self.edge_recalls else 0.0,
            "road_precision": float(np.mean(self.road_precisions)) if self.road_precisions else 0.0,
            "road_recall": float(np.mean(self.road_recalls)) if self.road_recalls else 0.0,
            "jaccard": float(np.mean(self.jaccard_scores)) if self.jaccard_scores else 0.0,
            "edit_similarity": (
                float(np.mean(self.edit_similarities)) if self.edit_similarities else 0.0
            ),
            "cost_ratio": float(np.mean(self.cost_ratios)) if self.cost_ratios else 0.0,
            "avg_route_edges": float(np.mean(self.route_lengths)) if self.route_lengths else 0.0,
        }


@dataclass
class RetrievalStats:
    mrr_scores: List[float] = field(default_factory=list)
    success_at: Dict[int, List[float]] = field(default_factory=dict)
    ndcg_scores: List[float] = field(default_factory=list)

    def add(
        self,
        ranked_feasible: List[bool],
        ranked_relevances: List[float],
        all_relevances: List[float],
        k_values: List[int],
        top_k: int,
    ) -> None:
        self.mrr_scores.append(mrr(ranked_feasible))
        self.ndcg_scores.append(ndcg_at_k(ranked_relevances, all_relevances, top_k))
        for k in k_values:
            self.success_at.setdefault(k, []).append(success_at_k(ranked_feasible, k))

    def summary(self, top_k: int) -> dict:
        success_summary = {
            f"success@{k}": float(np.mean(scores)) if scores else 0.0
            for k, scores in sorted(self.success_at.items())
        }
        return {
            "mrr": float(np.mean(self.mrr_scores)) if self.mrr_scores else 0.0,
            **success_summary,
            f"ndcg@{top_k}": float(np.mean(self.ndcg_scores)) if self.ndcg_scores else 0.0,
        }


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_strategy_summary(strategy_name: str, summary: dict, total_samples: int) -> None:
    edge_f1 = f1(summary["edge_precision"], summary["edge_recall"])
    road_f1 = f1(summary["road_precision"], summary["road_recall"])
    cprint(f"\n{strategy_name}", "green", attrs=["bold"])
    print(f"Generated: {summary['generated']}/{total_samples} ({summary['generated_rate'] * 100:.2f}%)")
    print(
        f"Topology Valid: {summary['topology_valid']}/{total_samples} "
        f"({summary['topology_valid_rate'] * 100:.2f}% of all, "
        f"{summary['topology_valid_rate_on_generated'] * 100:.2f}% of generated)"
    )
    cprint(
        f"Edge P/R/F1: {summary['edge_precision'] * 100:.2f} / "
        f"{summary['edge_recall'] * 100:.2f} / {edge_f1 * 100:.2f}",
        "cyan",
    )
    cprint(
        f"Road P/R/F1: {summary['road_precision'] * 100:.2f} / "
        f"{summary['road_recall'] * 100:.2f} / {road_f1 * 100:.2f}",
        "cyan",
    )
    print(f"Average route length: {summary['avg_route_edges']:.2f} edges")
    cprint(
        f"Jaccard / Edit similarity: {summary['jaccard'] * 100:.2f} / "
        f"{summary['edit_similarity'] * 100:.2f}",
        "cyan",
    )
    cprint(f"Cost ratio (pred/gt): {summary['cost_ratio']:.3f}", "cyan")
    print(
        f"Valid in search graph: {summary['valid_in_graph']}/{total_samples} "
        f"({summary['valid_in_graph_rate'] * 100:.2f}% of all, "
        f"{summary['valid_in_graph_rate_on_generated'] * 100:.2f}% of generated)"
    )


def print_retrieval_summary(retrieval_summary: dict) -> None:
    cprint("\nRetrieval ranking (LLM ranked contexts)", "green", attrs=["bold"])
    print(f"MRR: {retrieval_summary['mrr'] * 100:.2f}%")
    for key, value in retrieval_summary.items():
        if key.startswith("success@"):
            print(f"{key}: {value * 100:.2f}%")
    for key, value in retrieval_summary.items():
        if key.startswith("ndcg@"):
            print(f"{key}: {value * 100:.4f}")


def print_cascade_fallback_summary(cascade_fallback: dict) -> None:
    cprint("\nCascade fallback (cascade_ranked_union_bridged_full)", "green", attrs=["bold"])
    for source, rate in cascade_fallback.items():
        print(f"{source}: {rate * 100:.2f}%")


def _self_check() -> None:
    assert normalized_edit_similarity([1, 2, 3], [1, 2, 3]) == 1.0
    assert normalized_edit_similarity([1, 2], [3, 4]) == 0.0
    assert compute_jaccard_similarity([1, 2], [2, 3], {1: 1.0, 2: 1.0, 3: 1.0}) == 1 / 3
    assert mrr([False, True, True]) == 0.5
    assert success_at_k([False, False, True], 2) == 0.0
    assert ndcg_at_k([1.0, 0.0], [1.0, 0.0], 2) == 1.0

    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, 0, travel_time=12.0, length=100.0)
    graph.add_edge(2, 3, 0, length=50.0)
    edges_df = gpd.GeoDataFrame({"length": [100.0, 50.0]}, index=[0, 1])
    edge_id_to_uvk = {0: (1, 2, 0), 1: (2, 3, 0)}
    weights, meta = build_task_edge_weights(edges_df, edge_id_to_uvk, graph, "travel_time")
    assert weights[0] == 12.0
    assert meta["graph_hits"] == 1
    assert meta["fallback_hits"] == 1

    corridor = nx.DiGraph()
    corridor.add_edge(10, 20, weight=1.0)
    full = nx.DiGraph()
    full.add_edge(1, 10, weight=1.0)
    full.add_edge(10, 20, weight=1.0)
    full.add_edge(20, 99, weight=1.0)
    bridged = build_bridged_corridor_graph(corridor, full, 1, 99)
    assert search_edge_graph(bridged, 1, 99) == [1, 10, 20, 99]

    corridor = nx.DiGraph()
    corridor.add_edge(10, 20, weight=1.0)
    full = nx.DiGraph()
    full.add_edge(1, 5, weight=1.0)
    full.add_edge(5, 10, weight=1.0)
    full.add_edge(5, 99, weight=1.0)
    full.add_edge(10, 20, weight=1.0)
    full.add_edge(20, 99, weight=1.0)
    bridged = build_bridged_corridor_graph(corridor, full, 1, 99)
    assert search_edge_graph(bridged, 1, 99) == [1, 5, 10, 20, 99]


def evaluate_ranked_contexts() -> None:
    cprint("\n--- STARTING RANKED-CONTEXT EVALUATION ---", "yellow", attrs=["bold"])
    if variables.llm_task != "rank_contexts":
        cprint("This evaluator expects outputs generated with -llm_task rank_contexts.", "yellow")

    generated_path = f"generated_paths/{variables.path_type}/{ranked_output_name()}"
    metadata_path = f"retrieval_metadata/{variables.path_type}/{retrieval_metadata_name()}"
    test_data_path = f"filtered_test_data/{variables.path_type}/{variables.place_name}_data"
    subgraph_path = f"uncompressed_subgraphs/{variables.path_type}/{variables.place_name}_data.pkl"

    try:
        with open(generated_path, "rb") as f:
            raw_llm_outputs = pickle.load(f)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        with open(test_data_path, "rb") as f:
            ground_truth_data = pickle.load(f)
        with open(subgraph_path, "rb") as f:
            uncompressed_subgraphs = pickle.load(f)
    except FileNotFoundError as exc:
        cprint(f"Error loading files: {exc}", "red")
        raise SystemExit(1)

    if variables.place_name == "chengdu":
        ground_truth_data = ground_truth_data[:1_500]

    metadata_queries = metadata.get("queries", [])
    if len(raw_llm_outputs) != len(ground_truth_data) or len(raw_llm_outputs) != len(metadata_queries):
        cprint(
            "Input length mismatch among generated outputs, ground truth data, and retrieval metadata.",
            "red",
        )
        raise SystemExit(1)

    cprint("Loading edge data and building search graphs...", "yellow")
    edges_df = gpd.read_file(variables.EDGE_DATA)
    edge_id_to_name = build_edge_id_to_name(edges_df)
    edge_id_to_uvk = build_edge_id_to_uvk(edges_df)

    cprint("Loading pickled graph for task-specific edge weights...", "yellow")
    graph = load_graph(fname=variables.PICKLED_GRAPH)
    if variables.path_type in ("poi_aware", "scenic"):
        apply_poi_aware_weights(graph, edges_df)

    weight_column = task_weight_column(variables.path_type)
    edge_weights, weight_meta = build_task_edge_weights(
        edges_df, edge_id_to_uvk, graph, weight_column
    )
    length_column = "length" if "length" in edges_df.columns else weight_column
    edge_lengths = build_edge_weights(edges_df, length_column)
    full_edge_graph = build_full_edge_transition_graph(edges_df, edge_weights)
    top_k = variables.number_of_docs_to_retrieve
    success_k_values = sorted({k for k in (1, 3, 5, top_k) if k > 0})
    print_weight_source(weight_meta, variables.path_type)

    strategy_stats = {
        "ranked_first_feasible_corridor": StrategyStats(),
        "retrieved_top_k_union": StrategyStats(),
        "retrieved_top_k_union_bridged": StrategyStats(),
        "full_graph": StrategyStats(),
        "cascade_ranked_union_bridged_full": StrategyStats(),
    }
    retrieval_stats = RetrievalStats()
    cascade_fallback_counts: Dict[str, int] = defaultdict(int)
    gt_in_retrieved_union_count = 0
    samples = []
    valid_json_count = 0
    non_empty_rank_count = 0

    for i in tqdm(range(len(raw_llm_outputs)), dynamic_ncols=True, desc="Evaluating"):
        raw_text = raw_llm_outputs[i]
        path_collection = ground_truth_data[i]
        query_metadata = metadata_queries[i]

        ground_truth_edges = get_ground_truth_edges(path_collection)
        ground_truth_route = path_collection.get(f"{variables.path_type}_path_edges_names", [])
        start_edge = ground_truth_edges[0] if ground_truth_edges else None
        dest_edge = ground_truth_edges[-1] if ground_truth_edges else None

        ranked_context_ids, ranked_pairs = ranked_od_pairs(raw_text, query_metadata)
        retrieved_pairs = retrieved_od_pairs(query_metadata)
        if ground_truth_in_retrieved_union(
            ground_truth_edges, retrieved_pairs, uncompressed_subgraphs
        ):
            gt_in_retrieved_union_count += 1
        if has_ranked_contexts_json(raw_text):
            valid_json_count += 1
        if ranked_context_ids:
            non_empty_rank_count += 1

        first_route, selected_od_pair = search_first_feasible_ranked_corridor(
            ranked_pairs,
            uncompressed_subgraphs,
            edge_weights,
            start_edge,
            dest_edge,
        )

        union_graph = build_corridor_edge_graph(retrieved_pairs, uncompressed_subgraphs, edge_weights)
        union_route = search_edge_graph(union_graph, start_edge, dest_edge)
        bridged_union_graph = build_bridged_corridor_graph(
            union_graph, full_edge_graph, start_edge, dest_edge
        )
        bridged_union_route = search_edge_graph(bridged_union_graph, start_edge, dest_edge)
        full_route = search_edge_graph(full_edge_graph, start_edge, dest_edge)

        if first_route:
            cascade_route = first_route
            cascade_source = "ranked_first_feasible_corridor"
        elif union_route:
            cascade_route = union_route
            cascade_source = "retrieved_top_k_union"
        elif bridged_union_route:
            cascade_route = bridged_union_route
            cascade_source = "retrieved_top_k_union_bridged"
        elif full_route:
            cascade_route = full_route
            cascade_source = "full_graph"
        else:
            cascade_route = []
            cascade_source = "none"
        cascade_fallback_counts[cascade_source] += 1

        first_graph = (
            build_corridor_edge_graph([selected_od_pair], uncompressed_subgraphs, edge_weights)
            if selected_od_pair
            else None
        )

        corridor_evaluations = [
            evaluate_corridor(
                od_pair,
                uncompressed_subgraphs,
                edge_weights,
                start_edge,
                dest_edge,
                ground_truth_edges,
            )
            for od_pair in retrieved_pairs
        ]
        ranked_evaluations = [
            evaluate_corridor(
                od_pair,
                uncompressed_subgraphs,
                edge_weights,
                start_edge,
                dest_edge,
                ground_truth_edges,
            )
            for od_pair in ranked_pairs
        ]
        retrieval_stats.add(
            ranked_feasible=[feasible for feasible, _ in ranked_evaluations],
            ranked_relevances=[recall for _, recall in ranked_evaluations],
            all_relevances=[recall for _, recall in corridor_evaluations],
            k_values=success_k_values,
            top_k=top_k,
        )

        strategy_results = {
            "ranked_first_feasible_corridor": StrategyResult(
                edge_route=first_route,
                source="ranked_first_feasible_corridor" if first_route else "none",
                selected_od_pair=selected_od_pair,
                search_graph=first_graph,
            ),
            "retrieved_top_k_union": StrategyResult(
                edge_route=union_route,
                source="retrieved_top_k_union" if union_route else "none",
                search_graph=union_graph,
            ),
            "retrieved_top_k_union_bridged": StrategyResult(
                edge_route=bridged_union_route,
                source="retrieved_top_k_union_bridged" if bridged_union_route else "none",
                search_graph=bridged_union_graph,
            ),
            "full_graph": StrategyResult(
                edge_route=full_route,
                source="full_graph" if full_route else "none",
                search_graph=full_edge_graph,
            ),
            "cascade_ranked_union_bridged_full": StrategyResult(
                edge_route=cascade_route,
                source=cascade_source,
                selected_od_pair=selected_od_pair if first_route else None,
                search_graph=(
                    first_graph
                    if first_route
                    else union_graph
                    if union_route
                    else bridged_union_graph
                    if bridged_union_route
                    else full_edge_graph
                    if full_route
                    else None
                ),
            ),
        }

        for strategy_name, result in strategy_results.items():
            strategy_stats[strategy_name].add(
                result.edge_route,
                ground_truth_edges,
                ground_truth_route,
                edge_id_to_name,
                edge_id_to_uvk,
                edge_weights,
                edge_lengths,
                result.search_graph,
            )

        samples.append(
            {
                "query_index": i,
                "query_od_pair": [int(start_edge), int(dest_edge)]
                if start_edge is not None and dest_edge is not None
                else None,
                "ranked_contexts": ranked_context_ids,
                "retrieved_od_pairs": [[int(a), int(b)] for a, b in retrieved_pairs],
                "selected_ranked_od_pair": [int(selected_od_pair[0]), int(selected_od_pair[1])]
                if selected_od_pair
                else None,
                "strategy_sources": {
                    strategy_name: result.source for strategy_name, result in strategy_results.items()
                },
                "strategy_route_lengths": {
                    strategy_name: len(result.edge_route)
                    for strategy_name, result in strategy_results.items()
                },
            }
        )

    total_samples = len(raw_llm_outputs)
    summaries = {
        strategy_name: stats.summary(total_samples) for strategy_name, stats in strategy_stats.items()
    }
    retrieval_summary = retrieval_stats.summary(top_k)
    cascade_fallback = {
        source: count / total_samples if total_samples else 0.0
        for source, count in sorted(cascade_fallback_counts.items())
    }
    gt_in_retrieved_union_rate = (
        gt_in_retrieved_union_count / total_samples if total_samples else 0.0
    )

    cprint(f"\nResults for {variables.place_name} ({variables.path_type})", "green", attrs=["bold"])
    print(f"Total Samples Tested: {total_samples}")
    print(f"Valid Ranked-Context JSON: {valid_json_count}/{total_samples}")
    print(f"Non-empty Ranked Contexts: {non_empty_rank_count}/{total_samples}")
    print(
        f"GT fully in retrieved union: {gt_in_retrieved_union_count}/{total_samples} "
        f"({gt_in_retrieved_union_rate * 100:.2f}%)"
    )
    if gt_in_retrieved_union_rate < 0.5:
        cprint(
            "WARNING: ground-truth routes are often missing from retrieved corridor unions. "
            "Similarity metrics may stay low even with correct task weights.",
            "yellow",
        )
    print("-" * 30)
    print_retrieval_summary(retrieval_summary)
    print_cascade_fallback_summary(cascade_fallback)
    print("-" * 30)
    for strategy_name, summary in summaries.items():
        print_strategy_summary(strategy_name, summary, total_samples)

    output_dir = f"evaluation_results/{variables.path_type}/"
    make_dir(output_dir)
    output_path = output_dir + (
        f"{variables.place_name}_{variables.retrieval_type}_top_"
        f"{variables.number_of_docs_to_retrieve}_rank_contexts.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "place_name": variables.place_name,
                "path_type": variables.path_type,
                "retrieval_type": variables.retrieval_type,
                "top_k": variables.number_of_docs_to_retrieve,
                "weight_column": weight_column,
                "weight_source": weight_meta,
                "gt_in_retrieved_union_rate": gt_in_retrieved_union_rate,
                "total_samples": total_samples,
                "valid_ranked_context_json": valid_json_count,
                "non_empty_ranked_contexts": non_empty_rank_count,
                "retrieval_summary": retrieval_summary,
                "cascade_fallback": cascade_fallback,
                "summaries": summaries,
                "samples": samples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    cprint(f"\nSaved ranked-context evaluation to {output_path}", "green")


if __name__ == "__main__":
    _self_check()
    evaluate_ranked_contexts()
