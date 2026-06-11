import pickle
from pathlib import Path
import variables
import networkx as nx
import geopandas as gpd
from termcolor import cprint
from tqdm import tqdm
from collections import defaultdict
from generate_custom_dataset import edge_id_to_node_id
import generate_custom_dataset as gcd
from typing import List, Dict, Any, Set, Tuple
from generate_custom_dataset import load_graph, remap_to_edges

from itertools import islice


def _get_od_nodes(
    historical_path: List[int], edge_id_to_uvk: Dict[int, Tuple[int, int, int]]
) -> Tuple[int, int]:
    """Derive oriented origin/destination nodes from an edge path."""
    if len(historical_path) >= 2:
        node_path = edge_id_to_node_id(historical_path)
        return node_path[0], node_path[-1]
    u, v, _ = edge_id_to_uvk[historical_path[0]]
    return u, v


def construct_local_subgraphs(
    graph: nx.MultiDiGraph,
    dataset: List[Dict[str, Any]],
    edge_id_to_uvk: Dict[int, Tuple[int, int, int]],
    top_k: int = 3,
) -> Dict[Tuple[int, int], Dict[int, Set[int]]]:
    """
    Constructs a local subgraph (adjacency list) for each edge-level Origin-Destination pair
    by merging historical, fastest, shortest, and k-shortest paths.
    """
    gcd.edge_id_to_uvk = edge_id_to_uvk
    gcd.uvk_to_edge_id = {uvk: edge_id for edge_id, uvk in edge_id_to_uvk.items()}

    # Nested defaultdict: OD_pair -> { current_edge -> {next_edge_1, next_edge_2} }
    subgraphs = defaultdict(lambda: defaultdict(set))

    od_pair_k_shortest_map = defaultdict(list)
    for path_collection in tqdm(
        dataset, total=len(dataset), dynamic_ncols=True, desc="Generating local subgraphs"
    ):
        # Safely get paths (in case some are missing)
        historical_path = path_collection.get("historical_path_edges", [])
        fastest_path = path_collection.get("fastest_path_edges", [])
        shortest_path = path_collection.get("shortest_path_edges", [])
        # custom_path = path_collection.get(f"{variables.path_type}_path_edges", [])

        # Skip if there's no historical path to define the OD pair
        if not historical_path:
            continue

        start_edge, destination_edge = historical_path[0], historical_path[-1]
        od_pair = (start_edge, destination_edge)

        if od_pair not in od_pair_k_shortest_map:
            try:
                start_node, destination_node = _get_od_nodes(historical_path, edge_id_to_uvk)
                top_k_shortest_nodes_paths = nx.shortest_simple_paths(
                    graph, start_node, destination_node, weight="length"
                )
                top_k_shortest_edges_paths = [
                    remap_to_edges(p, graph, "length") for p in islice(top_k_shortest_nodes_paths, top_k)
                ]
            except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
                top_k_shortest_edges_paths = []
            od_pair_k_shortest_map[od_pair] = top_k_shortest_edges_paths

        top_k_shortest = od_pair_k_shortest_map[od_pair]

        # Merge all edges from all paths into the subgraph
        for path in [historical_path, fastest_path, shortest_path, *top_k_shortest]:
            if not path:
                continue  # Skip empty paths

            for i in range(len(path) - 1):
                u = path[i]  # Current edge ID
                v = path[i + 1]  # Next edge ID

                # Because we used defaultdict, we can directly add to the set
                subgraphs[od_pair][u].add(v)

    # Convert back to standard dict for normal usage/printing
    return dict(subgraphs)


def compress_edge_subgraph(
    subgraph: Dict[int, Set[int]], start_edge: int
) -> Dict[Tuple[int, ...], Set[Tuple[int, ...]]]:
    """
    Compresses an edge-based subgraph by merging linear edge sequences.
    """
    # 1. Calculate in-degrees
    in_degree = {edge: 0 for edge in subgraph}
    for u, neighbors in subgraph.items():
        for v in neighbors:
            if v not in in_degree:
                in_degree[v] = 0
            in_degree[v] += 1

    def traverse_and_compress(current_edge: int) -> Tuple[int, ...]:
        """Walks forward until it hits a decision boundary."""
        segment = [current_edge]
        curr = current_edge

        while True:
            neighbors = list(subgraph.get(curr, []))

            # Stop if no neighbors (dead end / destination)
            if not neighbors:
                break

            # Stop if fork in the road
            if len(neighbors) > 1:
                break

            nxt = neighbors[0]

            # Stop if the next edge is a merge point
            if in_degree.get(nxt, 0) > 1:
                break

            # Otherwise, it's a 1-to-1 connection. Add and continue.
            segment.append(nxt)
            curr = nxt

        return tuple(segment)

    # 2. Graph Building Logic (BFS Approach)
    compressed_graph = defaultdict(set)
    visited_segments = set()
    queue = [start_edge]  # Start building from the origin

    while queue:
        curr_start = queue.pop(0)

        # Get the compressed segment starting at this edge
        segment = traverse_and_compress(curr_start)

        if segment in visited_segments:
            continue

        visited_segments.add(segment)

        # Look at the last edge in our newly formed segment
        last_edge = segment[-1]
        neighbors = subgraph.get(last_edge, [])

        # For every neighbor, find its compressed segment and link them
        for nxt in neighbors:
            nxt_segment = traverse_and_compress(nxt)
            compressed_graph[segment].add(nxt_segment)
            # compressed_graph.setdefault(segment, set())

            # Queue the next segment's start edge if we haven't processed it
            if nxt_segment not in visited_segments:
                queue.append(nxt)

    # Convert inner sets back to standard dict for clean output
    return {k: set(v) for k, v in compressed_graph.items()}


if __name__ == "__main__":
    graph = load_graph(fname=variables.PICKLED_GRAPH)

    edges_df = gpd.read_file(variables.EDGE_DATA)

    edge_id_to_uvk = {i: (row.u, row.v, row.key) for i, row in edges_df.iterrows()}
    uvk_to_edge_id = {(row.u, row.v, row.key): i for i, row in edges_df.iterrows()}
    gcd.edge_id_to_uvk = edge_id_to_uvk
    gcd.uvk_to_edge_id = uvk_to_edge_id

    filtered_train_data = f"filtered_train_data/{variables.path_type}/{variables.place_name}_data"
    try:
        with open(filtered_train_data, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Train data not found at {data}! Please run generate_custom_dataset.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(data)} test samples from.", "cyan")

    # Local graph construction and compression below.

    # 1. Build the uncompressed subgraphs
    cprint("Constructing raw local subgraphs...", "yellow")
    uncompressed_subgraphs = construct_local_subgraphs(graph, data, edge_id_to_uvk)

    Path(f"uncompressed_subgraphs/{variables.path_type}").mkdir(parents=True, exist_ok=True)
    with open(f"uncompressed_subgraphs/{variables.path_type}/{variables.place_name}_data.pkl", "wb") as f:
        pickle.dump(uncompressed_subgraphs, f)

    # 2. Loop through and compress each one
    final_compressed_graphs = {}

    cprint("Compressing subgraphs...", "yellow")

    for od_pair, subgraph in tqdm(uncompressed_subgraphs.items(), desc="Compressing", dynamic_ncols=True):
        start_edge, dest_edge = od_pair

        compressed_subgraph = compress_edge_subgraph(subgraph, start_edge)

        final_compressed_graphs[od_pair] = compressed_subgraph

    cprint(f"Successfully generated {len(final_compressed_graphs)} compressed subgraphs!", "green")

    Path(f"compressed_subgraphs/{variables.path_type}").mkdir(parents=True, exist_ok=True)
    with open(f"compressed_subgraphs/{variables.path_type}/{variables.place_name}_data", "wb") as f:
        pickle.dump(final_compressed_graphs, f)
