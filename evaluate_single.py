import json
import re
import pickle
import variables
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple
from termcolor import cprint
from filter_custom_dataset import clean_street_name


def extract_json_object(raw_text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return {}


def extract_json_route(raw_text: str) -> list:
    """
    Uses regex to safely find and parse the JSON block,
    and unrolls compressed road segments.
    """
    raw_route = extract_json_object(raw_text).get("route", [])
    if not isinstance(raw_route, list):
        return []

    flattened_route = []
    for item in raw_route:
        sub_roads = [road.strip() for road in str(item).split("-")]
        flattened_route.extend(sub_roads)

    return [r for r in flattened_route if r and r != "未知道路"]


def extract_json_segments(raw_text: str) -> List[str]:
    raw_segments = extract_json_object(raw_text).get("route_segments", [])
    if not isinstance(raw_segments, list):
        return []
    return [str(segment_id).strip() for segment_id in raw_segments if str(segment_id).strip()]


def decode_segment_route(segment_route: List[str], id_to_edges: Dict[str, List[int]]) -> List[int]:
    edge_route = []
    for segment_id in segment_route:
        for edge_id in id_to_edges.get(segment_id, []):
            edge_id = int(edge_id)
            if not edge_route or edge_route[-1] != edge_id:
                edge_route.append(edge_id)
    return edge_route


def edge_ids_to_road_names(edge_route: List[int], edge_id_to_name: Dict[int, str]) -> List[str]:
    road_names = []
    for edge_id in edge_route:
        road_name = edge_id_to_name.get(edge_id, "未知道路")
        if road_name != "未知道路" and (not road_names or road_names[-1] != road_name):
            road_names.append(road_name)
    return road_names


def calculate_metrics(generated_route: list, ground_truth_route: list) -> tuple[float, float]:
    """
    Calculates Precision and Recall.
    """
    if not generated_route:
        return 0.0, 0.0

    gen_set = set(generated_route)
    gt_set = set(ground_truth_route)

    intersection = len(gen_set.intersection(gt_set))

    precision = intersection / len(gen_set) if len(gen_set) > 0 else 0.0
    recall = intersection / len(gt_set) if len(gt_set) > 0 else 0.0

    return precision, recall


def build_name_adjacency_graph(edges_df: pd.DataFrame) -> Dict[str, Set[str]]:
    """
    Builds a topological adjacency graph of road names using intersection nodes.

    Args:
        edges_df: The GeoDataFrame of the OpenStreetMap network.

    Returns:
        A dictionary mapping a road name to a set of valid next road names.
    """
    # Step 1: Map every intersection node to the road names that LEAVE it
    outgoing_from_node = defaultdict(set)

    for _, row in edges_df.iterrows():
        raw_name = row.get("name", "Unnamed Road")
        name = clean_street_name(raw_name)
        if name == "Unnamed Road":
            name = "未知道路"

        outgoing_from_node[row.u].add(name)

    # Step 2: Map every road name to the roads that start where it ENDS
    name_adjacency_graph = defaultdict(set)

    for _, row in edges_df.iterrows():
        raw_name = row.get("name", "Unnamed Road")
        name = clean_street_name(raw_name)
        if name == "Unnamed Road":
            name = "未知道路"

        # The destination node of this edge is row.v.
        # Any road leaving row.v is a valid next step.
        valid_next_roads = outgoing_from_node.get(row.v, set())
        name_adjacency_graph[name].update(valid_next_roads)

    # Step 3: Ensure self-continuity
    # A single physical street is often broken into dozens of edges in OSM.
    # We guarantee a road can always transition to itself.
    for name in name_adjacency_graph:
        name_adjacency_graph[name].add(name)

    return dict(name_adjacency_graph)


def check_route_connectivity(generated_route: list, name_adjacency_graph: dict) -> bool:
    """
    Checks if a generated sequence of road names is topologically continuous.

    Args:
        generated_route: List of road names e.g., ["Road A", "Road B", "Road C"]
        name_adjacency_graph: A dictionary mapping a road name to a set of valid next road names.
    """
    # A route with 0 or 1 road is technically continuous (though maybe not a useful trip)
    if len(generated_route) <= 1:
        return True

    for i in range(len(generated_route) - 1):
        current_road = generated_route[i]
        next_road = generated_route[i + 1]

        # Check if they are the exact same road (e.g., self-loops or duplicate generation)
        if current_road == next_road:
            continue

        # Get the valid neighbors for the current road
        valid_neighbors = name_adjacency_graph.get(current_road, set())

        if next_road not in valid_neighbors:
            # The chain is broken! Teleportation detected.
            return False

    # If the loop finishes without returning False, the path is completely valid
    return True


def build_edge_id_to_uvk(edges_df: pd.DataFrame) -> Dict[int, tuple]:
    return {int(i): (row.u, row.v, row.key) for i, row in edges_df.iterrows()}


def check_edge_connectivity(edge_route: List[int], edge_id_to_uvk: Dict[int, tuple]) -> bool:
    # ponytail: endpoint-sharing check; upgrade to directed graph validation if one-way legality matters.
    if len(edge_route) <= 1:
        return True

    for prev_edge, next_edge in zip(edge_route, edge_route[1:]):
        prev_uvk = edge_id_to_uvk.get(int(prev_edge))
        next_uvk = edge_id_to_uvk.get(int(next_edge))
        if not prev_uvk or not next_uvk:
            return False
        if not ({prev_uvk[0], prev_uvk[1]} & {next_uvk[0], next_uvk[1]}):
            return False
    return True


def build_edge_id_to_name(edges_df: pd.DataFrame) -> Dict[int, str]:
    edge_id_to_name = {}
    for edge_id, row in edges_df.iterrows():
        road_name = clean_street_name(row.get("name", "Unnamed Road"))
        edge_id_to_name[int(edge_id)] = "未知道路" if road_name == "Unnamed Road" else road_name
    return edge_id_to_name


def safe_float(value: Any, default: float = 1.0) -> float:
    try:
        if value is None or value != value:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_repair_graph(edges_df: pd.DataFrame) -> nx.Graph:
    repair_graph = nx.Graph()
    for edge_id, row in edges_df.iterrows():
        u, v = row.u, row.v
        length = safe_float(row.get("length", 1.0))
        edge_id = int(edge_id)

        if repair_graph.has_edge(u, v) and repair_graph[u][v]["weight"] <= length:
            continue
        repair_graph.add_edge(u, v, weight=length, edge_id=edge_id)
    return repair_graph


def path_length(node_path: List[Any], repair_graph: nx.Graph) -> float:
    return sum(repair_graph[u][v]["weight"] for u, v in zip(node_path, node_path[1:]))


def node_path_to_edge_ids(node_path: List[Any], repair_graph: nx.Graph) -> List[int]:
    return [int(repair_graph[u][v]["edge_id"]) for u, v in zip(node_path, node_path[1:])]


def shortest_bridge_edges(
    prev_edge: int,
    next_edge: int,
    edge_id_to_uvk: Dict[int, tuple],
    repair_graph: nx.Graph,
    bridge_cache: Dict[Tuple[int, int], Optional[List[int]]],
) -> Optional[List[int]]:
    key = (int(prev_edge), int(next_edge))
    if key in bridge_cache:
        return bridge_cache[key]

    prev_uvk = edge_id_to_uvk.get(int(prev_edge))
    next_uvk = edge_id_to_uvk.get(int(next_edge))
    if not prev_uvk or not next_uvk:
        bridge_cache[key] = None
        return None

    prev_nodes = (prev_uvk[0], prev_uvk[1])
    next_nodes = (next_uvk[0], next_uvk[1])
    if set(prev_nodes) & set(next_nodes):
        bridge_cache[key] = []
        return []

    best_path = None
    best_length = float("inf")
    for source in prev_nodes:
        for target in next_nodes:
            try:
                node_path = nx.shortest_path(repair_graph, source, target, weight="weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            candidate_length = path_length(node_path, repair_graph)
            if candidate_length < best_length:
                best_length = candidate_length
                best_path = node_path

    bridge_edges = node_path_to_edge_ids(best_path, repair_graph) if best_path else None
    bridge_cache[key] = bridge_edges
    return bridge_edges


def append_edge(edge_route: List[int], edge_id: int) -> None:
    edge_id = int(edge_id)
    if not edge_route or edge_route[-1] != edge_id:
        edge_route.append(edge_id)


def repair_edge_route(
    edge_route: List[int],
    edge_id_to_uvk: Dict[int, tuple],
    repair_graph: nx.Graph,
    bridge_cache: Dict[Tuple[int, int], Optional[List[int]]],
    start_edge: Optional[int] = None,
    dest_edge: Optional[int] = None,
) -> Tuple[List[int], int, int, int]:
    # ponytail: shortest-path gap stitching on the full edge graph; upgrade to corridor-bounded repair if needed.
    if not edge_route:
        return [], 0, 0, 0

    route_to_repair = [int(edge_id) for edge_id in edge_route]
    if start_edge is not None and route_to_repair[0] != int(start_edge):
        route_to_repair.insert(0, int(start_edge))
    if dest_edge is not None and route_to_repair[-1] != int(dest_edge):
        route_to_repair.append(int(dest_edge))

    repaired = [route_to_repair[0]]
    attempted = 0
    fixed = 0
    failed = 0

    for next_edge in route_to_repair[1:]:
        bridge_edges = shortest_bridge_edges(
            repaired[-1], next_edge, edge_id_to_uvk, repair_graph, bridge_cache
        )
        if bridge_edges is None:
            attempted += 1
            failed += 1
            append_edge(repaired, next_edge)
            continue

        if bridge_edges:
            attempted += 1
            fixed += 1
            for bridge_edge in bridge_edges:
                append_edge(repaired, bridge_edge)
        append_edge(repaired, next_edge)

    return repaired, attempted, fixed, failed


if __name__ == "__main__":
    cprint("\n--- STARTING EVALUATION ---", "yellow", attrs=["bold"])

    # 1. Load the generated results from the previous vLLM script
    file_path = f"generated_paths/{variables.path_type}/"
    if variables.use_context:
        file_name = f"{variables.retrieval_type}_context_{variables.place_name}_top_{variables.number_of_docs_to_retrieve}"
    else:
        file_name = f"no_context_{variables.place_name}_top_{variables.number_of_docs_to_retrieve}"

    try:
        with open(file_path + file_name, "rb") as f:
            raw_llm_outputs = pickle.load(f)

        test_data_filename = f"filtered_test_data/{variables.path_type}/{variables.place_name}_data"
        with open(test_data_filename, "rb") as f:
            ground_truth_data = pickle.load(f)

    except FileNotFoundError as e:
        cprint(f"Error loading files: {e}", "red")
        exit(1)

    cprint("\nLoading edge data...", "yellow")
    edges_df = gpd.read_file(variables.EDGE_DATA)
    edge_id_to_name = build_edge_id_to_name(edges_df)
    edge_id_to_uvk = build_edge_id_to_uvk(edges_df)
    repair_graph = build_repair_graph(edges_df) if variables.use_context else None
    bridge_cache = {}

    id_to_edges = {}
    if variables.use_context:
        registry_filename = (
            f"symbolic_subgraphs/{variables.path_type}/{variables.place_name}_segment_registry"
        )
        try:
            with open(registry_filename, "rb") as f:
                segment_registry = pickle.load(f)
            id_to_edges = segment_registry.get("id_to_edges", {})
        except FileNotFoundError:
            cprint(
                f"Segment registry not found at {registry_filename}. Please run subgraph_construction.py first.",
                "red",
            )
            exit(1)
    else:
        name_adjacency_graph = build_name_adjacency_graph(edges_df)

    all_precisions = []
    all_recalls = []
    all_road_precisions = []
    all_road_recalls = []
    repaired_edge_precisions = []
    repaired_edge_recalls = []
    repaired_road_precisions = []
    repaired_road_recalls = []
    valid_routes_count = 0
    topologically_valid_count = 0
    repaired_topologically_valid_count = 0
    repair_attempted_count = 0
    repair_fixed_count = 0
    repair_failed_count = 0

    cprint("Evaluating generated routes...", "yellow")
    for i in range(len(raw_llm_outputs)):
        raw_text = raw_llm_outputs[i]
        path_collection = ground_truth_data[i]

        if variables.use_context:
            segment_route = extract_json_segments(raw_text)
            predicted_edges = decode_segment_route(segment_route, id_to_edges)
            predicted_route = edge_ids_to_road_names(predicted_edges, edge_id_to_name)

            ground_truth_edges = path_collection.get(f"{variables.path_type}_path_edges", [])
            ground_truth_route = path_collection.get(f"{variables.path_type}_path_edges_names", [])
            start_edge = ground_truth_edges[0] if ground_truth_edges else None
            dest_edge = ground_truth_edges[-1] if ground_truth_edges else None

            if predicted_edges:
                valid_routes_count += 1
                if check_edge_connectivity(predicted_edges, edge_id_to_uvk):
                    topologically_valid_count += 1

                repaired_edges, attempted, fixed, failed = repair_edge_route(
                    predicted_edges,
                    edge_id_to_uvk,
                    repair_graph,
                    bridge_cache,
                    start_edge=start_edge,
                    dest_edge=dest_edge,
                )
                repaired_route = edge_ids_to_road_names(repaired_edges, edge_id_to_name)

                repair_attempted_count += attempted
                repair_fixed_count += fixed
                repair_failed_count += failed
                if check_edge_connectivity(repaired_edges, edge_id_to_uvk):
                    repaired_topologically_valid_count += 1

                if ground_truth_edges:
                    repaired_p, repaired_r = calculate_metrics(repaired_edges, ground_truth_edges)
                else:
                    repaired_p, repaired_r = calculate_metrics(repaired_route, ground_truth_route)
                repaired_edge_precisions.append(repaired_p)
                repaired_edge_recalls.append(repaired_r)

                repaired_road_p, repaired_road_r = calculate_metrics(repaired_route, ground_truth_route)
                repaired_road_precisions.append(repaired_road_p)
                repaired_road_recalls.append(repaired_road_r)
            else:
                repaired_edge_precisions.append(0.0)
                repaired_edge_recalls.append(0.0)
                repaired_road_precisions.append(0.0)
                repaired_road_recalls.append(0.0)

            if ground_truth_edges:
                p, r = calculate_metrics(predicted_edges, ground_truth_edges)
            else:
                p, r = calculate_metrics(predicted_route, ground_truth_route)

            road_p, road_r = calculate_metrics(predicted_route, ground_truth_route)
            all_road_precisions.append(road_p)
            all_road_recalls.append(road_r)
        else:
            ground_truth_route = path_collection[f"{variables.path_type}_path_edges_names"]
            predicted_route = extract_json_route(raw_text)

            if predicted_route:
                valid_routes_count += 1
                if check_route_connectivity(predicted_route, name_adjacency_graph):
                    topologically_valid_count += 1

            p, r = calculate_metrics(predicted_route, ground_truth_route)

        all_precisions.append(p)
        all_recalls.append(r)

    avg_precision = np.mean(all_precisions) * 100
    avg_recall = np.mean(all_recalls) * 100
    success_rate = (valid_routes_count / len(raw_llm_outputs)) * 100
    topology_rate = (topologically_valid_count / len(raw_llm_outputs)) * 100
    topology_rate_on_generated = (
        (topologically_valid_count / valid_routes_count) * 100 if valid_routes_count else 0.0
    )
    repaired_topology_rate = (repaired_topologically_valid_count / len(raw_llm_outputs)) * 100
    repaired_topology_rate_on_generated = (
        (repaired_topologically_valid_count / valid_routes_count) * 100 if valid_routes_count else 0.0
    )

    cprint(f"\nResults for {variables.place_name} ({variables.path_type}):", "green", attrs=["bold"])
    print(f"Total Samples Tested: {len(raw_llm_outputs)}")
    print(f"Valid JSON Routes Generated: {valid_routes_count}/{len(raw_llm_outputs)} ({success_rate:.2f}%)")
    print("-" * 30)
    metric_prefix = "Average Edge" if variables.use_context else "Average Road"
    cprint(f"{metric_prefix} Precision: {avg_precision:.2f}%", "cyan")
    cprint(f"{metric_prefix} Recall:    {avg_recall:.2f}%", "cyan")
    if variables.use_context and all_road_precisions:
        cprint(f"Average Road Precision: {np.mean(all_road_precisions) * 100:.2f}%", "cyan")
        cprint(f"Average Road Recall:    {np.mean(all_road_recalls) * 100:.2f}%", "cyan")
    cprint(
        f"Topologically Valid Routes: {topologically_valid_count}/{len(raw_llm_outputs)} "
        f"({topology_rate:.2f}% of all samples, {topology_rate_on_generated:.2f}% of generated routes)",
        "cyan",
    )
    if variables.use_context:
        print("-" * 30)
        cprint("After Graph Repair:", "green", attrs=["bold"])
        cprint(f"Repaired Edge Precision: {np.mean(repaired_edge_precisions) * 100:.2f}%", "cyan")
        cprint(f"Repaired Edge Recall:    {np.mean(repaired_edge_recalls) * 100:.2f}%", "cyan")
        cprint(f"Repaired Road Precision: {np.mean(repaired_road_precisions) * 100:.2f}%", "cyan")
        cprint(f"Repaired Road Recall:    {np.mean(repaired_road_recalls) * 100:.2f}%", "cyan")
        cprint(
            f"Repaired Topologically Valid Routes: {repaired_topologically_valid_count}/{len(raw_llm_outputs)} "
            f"({repaired_topology_rate:.2f}% of all samples, "
            f"{repaired_topology_rate_on_generated:.2f}% of generated routes)",
            "cyan",
        )
        cprint(
            f"Repair Gaps: fixed {repair_fixed_count}/{repair_attempted_count}, failed {repair_failed_count}",
            "cyan",
        )
