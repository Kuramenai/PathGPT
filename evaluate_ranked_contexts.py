import json
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


def choose_weight_column(edges_df: gpd.GeoDataFrame) -> str:
    candidates_by_task = {
        "fastest": ["travel_time", "time", "length"],
        "shortest": ["length"],
        "fuel_efficient": ["fuel_cost", "length"],
        "poi_aware": ["touristic_value", "length"],
        "scenic": ["touristic_value", "length"],
        "highway_free": ["fuel_cost", "length"],
    }
    for column in candidates_by_task.get(variables.path_type, ["length"]):
        if column in edges_df.columns:
            return column
    return "length"


def build_edge_weights(edges_df: gpd.GeoDataFrame, weight_column: str) -> Dict[int, float]:
    edge_weights = {}
    for edge_id, row in edges_df.iterrows():
        edge_weights[int(edge_id)] = safe_float(row.get(weight_column, 1.0))
    return edge_weights


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
            int(edge_id)
            for edge_id in nx.shortest_path(edge_graph, start_edge, dest_edge, weight="weight")
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


@dataclass
class StrategyResult:
    edge_route: List[int] = field(default_factory=list)
    source: str = "none"
    selected_od_pair: Optional[OdPair] = None


@dataclass
class StrategyStats:
    edge_precisions: List[float] = field(default_factory=list)
    edge_recalls: List[float] = field(default_factory=list)
    road_precisions: List[float] = field(default_factory=list)
    road_recalls: List[float] = field(default_factory=list)
    generated_count: int = 0
    topology_valid_count: int = 0
    route_lengths: List[int] = field(default_factory=list)

    def add(
        self,
        edge_route: List[int],
        ground_truth_edges: List[int],
        ground_truth_route: List[str],
        edge_id_to_name: Dict[int, str],
        edge_id_to_uvk: Dict[int, tuple],
    ) -> None:
        if edge_route:
            self.generated_count += 1
            self.route_lengths.append(len(edge_route))
            if check_directed_edge_connectivity(edge_route, edge_id_to_uvk):
                self.topology_valid_count += 1

        predicted_route = edge_ids_to_road_names(edge_route, edge_id_to_name)
        edge_p, edge_r = calculate_metrics(edge_route, ground_truth_edges)
        road_p, road_r = calculate_metrics(predicted_route, ground_truth_route)

        self.edge_precisions.append(edge_p)
        self.edge_recalls.append(edge_r)
        self.road_precisions.append(road_p)
        self.road_recalls.append(road_r)

    def summary(self, total_samples: int) -> dict:
        return {
            "generated": self.generated_count,
            "generated_rate": self.generated_count / total_samples if total_samples else 0.0,
            "topology_valid": self.topology_valid_count,
            "topology_valid_rate": (
                self.topology_valid_count / total_samples if total_samples else 0.0
            ),
            "topology_valid_rate_on_generated": (
                self.topology_valid_count / self.generated_count if self.generated_count else 0.0
            ),
            "edge_precision": float(np.mean(self.edge_precisions)) if self.edge_precisions else 0.0,
            "edge_recall": float(np.mean(self.edge_recalls)) if self.edge_recalls else 0.0,
            "road_precision": float(np.mean(self.road_precisions)) if self.road_precisions else 0.0,
            "road_recall": float(np.mean(self.road_recalls)) if self.road_recalls else 0.0,
            "avg_route_edges": float(np.mean(self.route_lengths)) if self.route_lengths else 0.0,
        }


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def print_strategy_summary(strategy_name: str, summary: dict, total_samples: int) -> None:
    edge_f1 = f1(summary["edge_precision"], summary["edge_recall"])
    road_f1 = f1(summary["road_precision"], summary["road_recall"])
    cprint(f"\n{strategy_name}", "green", attrs=["bold"])
    print(
        f"Generated: {summary['generated']}/{total_samples} "
        f"({summary['generated_rate'] * 100:.2f}%)"
    )
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

    metadata_queries = metadata.get("queries", [])
    if len(raw_llm_outputs) != len(ground_truth_data) or len(raw_llm_outputs) != len(metadata_queries):
        cprint(
            "Input length mismatch among generated outputs, ground truth data, and retrieval metadata.",
            "red",
        )
        raise SystemExit(1)

    cprint("Loading edge data and building search graphs...", "yellow")
    edges_df = gpd.read_file(variables.EDGE_DATA)
    weight_column = choose_weight_column(edges_df)
    edge_weights = build_edge_weights(edges_df, weight_column)
    edge_id_to_name = build_edge_id_to_name(edges_df)
    edge_id_to_uvk = build_edge_id_to_uvk(edges_df)
    full_edge_graph = build_full_edge_transition_graph(edges_df, edge_weights)
    cprint(f"Using edge search weight column: {weight_column}", "cyan")

    strategy_stats = {
        "ranked_first_feasible_corridor": StrategyStats(),
        "retrieved_top_k_union": StrategyStats(),
        "full_graph": StrategyStats(),
        "cascade_ranked_union_full": StrategyStats(),
    }
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
        full_route = search_edge_graph(full_edge_graph, start_edge, dest_edge)
        cascade_route = first_route or union_route or full_route
        cascade_source = (
            "ranked_first_feasible_corridor"
            if first_route
            else "retrieved_top_k_union"
            if union_route
            else "full_graph"
            if full_route
            else "none"
        )

        strategy_results = {
            "ranked_first_feasible_corridor": StrategyResult(
                edge_route=first_route,
                source="ranked_first_feasible_corridor" if first_route else "none",
                selected_od_pair=selected_od_pair,
            ),
            "retrieved_top_k_union": StrategyResult(
                edge_route=union_route,
                source="retrieved_top_k_union" if union_route else "none",
            ),
            "full_graph": StrategyResult(
                edge_route=full_route,
                source="full_graph" if full_route else "none",
            ),
            "cascade_ranked_union_full": StrategyResult(
                edge_route=cascade_route,
                source=cascade_source,
                selected_od_pair=selected_od_pair if first_route else None,
            ),
        }

        for strategy_name, result in strategy_results.items():
            strategy_stats[strategy_name].add(
                result.edge_route,
                ground_truth_edges,
                ground_truth_route,
                edge_id_to_name,
                edge_id_to_uvk,
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
        strategy_name: stats.summary(total_samples)
        for strategy_name, stats in strategy_stats.items()
    }

    cprint(f"\nResults for {variables.place_name} ({variables.path_type})", "green", attrs=["bold"])
    print(f"Total Samples Tested: {total_samples}")
    print(f"Valid Ranked-Context JSON: {valid_json_count}/{total_samples}")
    print(f"Non-empty Ranked Contexts: {non_empty_rank_count}/{total_samples}")
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
                "total_samples": total_samples,
                "valid_ranked_context_json": valid_json_count,
                "non_empty_ranked_contexts": non_empty_rank_count,
                "summaries": summaries,
                "samples": samples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    cprint(f"\nSaved ranked-context evaluation to {output_path}", "green")


if __name__ == "__main__":
    evaluate_ranked_contexts()
