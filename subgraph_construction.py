import pickle
import concurrent.futures
from pathlib import Path
import variables
import networkx as nx
import geopandas as gpd
from termcolor import cprint
from tqdm import tqdm
from collections import defaultdict, deque
from generate_custom_dataset import edge_id_to_node_id
import generate_custom_dataset as gcd
from typing import List, Dict, Any, Set, Tuple
from generate_custom_dataset import apply_poi_aware_weights, load_graph, poi_catalog_path, remap_to_edges
from filter_custom_dataset import clean_street_name
import osmnx as ox
from itertools import islice

Segment = Tuple[int, ...]
CompressedSubgraph = Dict[Segment, Set[Segment]]
POI_CONTEXT_LIMIT = 3

_GRAPH = None
_EDGE_ID_TO_UVK = None


def _get_od_nodes(
    historical_path: List[int], edge_id_to_uvk: Dict[int, Tuple[int, int, int]]
) -> Tuple[int, int]:
    """Derive oriented origin/destination nodes from an edge path."""
    if len(historical_path) >= 2:
        node_path = edge_id_to_node_id(historical_path)
        return node_path[0], node_path[-1]
    u, v, _ = edge_id_to_uvk[historical_path[0]]
    return u, v


def _init_k_shortest_worker(graph: nx.MultiDiGraph, edge_id_to_uvk: Dict[int, Tuple[int, int, int]]) -> None:
    global _GRAPH, _EDGE_ID_TO_UVK
    _GRAPH = graph
    _EDGE_ID_TO_UVK = edge_id_to_uvk
    gcd.edge_id_to_uvk = edge_id_to_uvk
    gcd.uvk_to_edge_id = {uvk: edge_id for edge_id, uvk in edge_id_to_uvk.items()}


def _compute_k_shortest_for_od(
    task: Tuple[Tuple[int, int], List[int], int],
) -> Tuple[Tuple[int, int], List[List[int]]]:
    od_pair, historical_path, top_k = task
    try:
        start_node, destination_node = _get_od_nodes(historical_path, _EDGE_ID_TO_UVK)
        node_paths = ox.routing.k_shortest_paths(
            _GRAPH, start_node, destination_node, k=top_k, weight="length"
        )
        edge_paths = [remap_to_edges(p, _GRAPH, "length") for p in islice(node_paths, top_k)]
    except (nx.NetworkXNoPath, nx.NodeNotFound, ValueError):
        edge_paths = []
    return od_pair, edge_paths


def construct_local_subgraphs(
    graph: nx.MultiDiGraph,
    dataset: List[Dict[str, Any]],
    edge_id_to_uvk: Dict[int, Tuple[int, int, int]],
    top_k_shortest: bool = False,
    top_k: int = 3,
    n_cores: int = 12,
) -> Dict[Tuple[int, int], Dict[int, Set[int]]]:
    """
    Constructs a local subgraph (adjacency list) for each edge-level Origin-Destination pair
    by merging historical, fastest, shortest, and k-shortest paths.
    """
    _init_k_shortest_worker(graph, edge_id_to_uvk)

    # Collect unique OD pairs
    if top_k_shortest:
        od_to_historical: Dict[Tuple[int, int], List[int]] = {}
        for path_collection in dataset:
            historical_path = path_collection.get("historical_path_edges", [])
            if not historical_path:
                continue
            od_pair = (historical_path[0], historical_path[-1])
            od_to_historical.setdefault(od_pair, historical_path)

        tasks = [(od_pair, hist, top_k) for od_pair, hist in od_to_historical.items()]
        if n_cores > 1 and len(tasks) > 1:
            with concurrent.futures.ProcessPoolExecutor(
                max_workers=n_cores, initializer=_init_k_shortest_worker, initargs=(graph, edge_id_to_uvk)
            ) as executor:
                od_pair_k_shortest_map = dict(
                    tqdm(
                        executor.map(_compute_k_shortest_for_od, tasks, chunksize=50),
                        total=len(tasks),
                        dynamic_ncols=True,
                        desc="k-shortest paths",
                    )
                )
        else:
            od_pair_k_shortest_map = dict(
                tqdm(
                    (_compute_k_shortest_for_od(task) for task in tasks),
                    total=len(tasks),
                    dynamic_ncols=True,
                    desc="k-shortest paths",
                )
            )

    subgraphs = defaultdict(lambda: defaultdict(set))
    for path_collection in tqdm(
        dataset, total=len(dataset), dynamic_ncols=True, desc="Generating local subgraphs"
    ):
        historical_path = path_collection.get("historical_path_edges", [])
        fastest_path = path_collection.get("fastest_path_edges", [])
        shortest_path = path_collection.get("shortest_path_edges", [])

        if not historical_path:
            continue

        od_pair = (historical_path[0], historical_path[-1])
        if top_k_shortest:
            top_k_shortest = od_pair_k_shortest_map.get(od_pair, [])
        else:
            top_k_shortest = []

        for path in [historical_path, fastest_path, shortest_path, *top_k_shortest]:
            if not path:
                continue  # Skip empty paths
            if len(path) == 1:
                subgraphs[od_pair].setdefault(path[0], set())
                continue

            for i in range(len(path) - 1):
                u = path[i]  # Current edge ID
                v = path[i + 1]  # Next edge ID

                # Because we used defaultdict, we can directly add to the set
                subgraphs[od_pair][u].add(v)

    # Convert back to standard dict for normal usage/printing
    return dict(subgraphs)


def compress_edge_subgraph(subgraph: Dict[int, Set[int]], start_edge: int) -> CompressedSubgraph:
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
        compressed_graph.setdefault(segment, set())

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


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(value != value)
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if _is_missing(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_attribute_values(value: Any, default: str = "N/A") -> List[str]:
    if _is_missing(value):
        return [default]
    if isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = [value]
    cleaned_values = [str(v).strip() for v in values if str(v).strip()]
    return cleaned_values or [default]


def build_edge_attribute_dicts(
    graph: nx.MultiDiGraph,
    edges_df: gpd.GeoDataFrame,
    edge_id_to_uvk: Dict[int, Tuple[int, int, int]],
) -> Dict[str, Dict[int, Any]]:
    """
    Collects edge-level attributes used to describe compressed segment tuples.
    """
    edge_id_to_name = {}
    edge_id_to_length = {}
    edge_id_to_time = {}
    edge_id_to_type = {}
    edge_id_to_near_poi = {}
    edge_id_to_poi_ids = {}

    for edge_id, row in edges_df.iterrows():
        edge_id = int(edge_id)
        uvk = edge_id_to_uvk.get(edge_id)
        edge_data = {}
        if uvk:
            u, v, key = uvk
            edge_data = graph.get_edge_data(u, v, key, default={}) or {}

        raw_name = row.get("name", edge_data.get("name", "Unnamed Road"))
        raw_length = row.get("length", edge_data.get("length", 1.0))
        raw_time = row.get("travel_time", edge_data.get("travel_time", 1.0))
        raw_type = row.get("Type", row.get("highway", edge_data.get("highway", "N/A")))

        edge_id_to_name[edge_id] = raw_name
        edge_id_to_length[edge_id] = _safe_float(raw_length)
        edge_id_to_time[edge_id] = _safe_float(raw_time)
        edge_id_to_type[edge_id] = raw_type
        edge_id_to_near_poi[edge_id] = bool(edge_data.get("is_near_poi", False))
        edge_id_to_poi_ids[edge_id] = edge_data.get("poi_ids", frozenset())

    return {
        "edge_id_to_name": edge_id_to_name,
        "edge_id_to_length": edge_id_to_length,
        "edge_id_to_time": edge_id_to_time,
        "edge_id_to_type": edge_id_to_type,
        "edge_id_to_near_poi": edge_id_to_near_poi,
        "edge_id_to_poi_ids": edge_id_to_poi_ids,
    }


def describe_segment(
    segment: Segment,
    edge_dicts: Dict[str, Dict[int, Any]],
    poi_catalog: Dict[int, Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Builds human-readable semantics for one compressed segment tuple.
    """
    road_names = []
    road_types = set()
    total_length = 0.0
    total_time = 0.0
    poi_ids = set()
    near_poi = False

    for edge_id in segment:
        raw_name = edge_dicts["edge_id_to_name"].get(edge_id, "Unnamed Road")
        road_name = clean_street_name(raw_name)
        if road_name == "Unnamed Road":
            road_name = "Unknown Road"
        if not road_names or road_names[-1] != road_name:
            road_names.append(road_name)

        total_length += edge_dicts["edge_id_to_length"].get(edge_id, 0.0)
        total_time += edge_dicts["edge_id_to_time"].get(edge_id, 0.0)

        raw_type = edge_dicts["edge_id_to_type"].get(edge_id, "N/A")
        road_types.update(_normalize_attribute_values(raw_type))
        near_poi = near_poi or edge_dicts["edge_id_to_near_poi"].get(edge_id, False)
        poi_ids.update(edge_dicts["edge_id_to_poi_ids"].get(edge_id, frozenset()))

    poi_catalog = poi_catalog or {}
    poi_names = []
    poi_types = []
    for poi_id in sorted(poi_ids):
        info = poi_catalog.get(int(poi_id), {})
        poi_type = info.get("type", "")
        poi_name = info.get("name", "")
        if poi_type and poi_type not in poi_types:
            poi_types.append(poi_type)
        if poi_name and poi_name not in poi_names:
            poi_names.append(poi_name)

    display_name = " -> ".join(road_names) if road_names else "Unknown Road"
    road_types = sorted(road_types)

    return {
        "edge_count": len(segment),
        "road_names": road_names,
        "display_name": display_name,
        "length_m": round(total_length, 2),
        "travel_time_s": round(total_time, 2),
        "road_types": road_types,
        "near_poi": near_poi,
        "poi_count": len(poi_ids),
        "poi_names": poi_names[:POI_CONTEXT_LIMIT],
        "poi_types": poi_types[:POI_CONTEXT_LIMIT],
    }


def _segment_sort_key(segment: Segment) -> Tuple[int, int, Segment]:
    first_edge = int(segment[0]) if segment else -1
    return first_edge, len(segment), segment


def iter_compressed_segments(compressed_subgraph: CompressedSubgraph) -> Set[Segment]:
    all_segments = set(compressed_subgraph)
    for neighbors in compressed_subgraph.values():
        all_segments.update(neighbors)
    return all_segments


def build_global_segment_registry(
    compressed_subgraphs: Dict[Tuple[int, int], CompressedSubgraph],
    id_prefix: str = "G",
) -> Dict[str, Any]:
    """
    Assigns one deterministic global symbolic ID to each unique compressed edge tuple.
    """
    all_segments = set()
    for compressed_subgraph in compressed_subgraphs.values():
        all_segments.update(iter_compressed_segments(compressed_subgraph))

    ordered_segments = sorted(all_segments, key=_segment_sort_key)
    segment_to_id = {segment: f"{id_prefix}{idx}" for idx, segment in enumerate(ordered_segments)}
    id_to_segment = {segment_id: segment for segment, segment_id in segment_to_id.items()}

    return {
        "id_prefix": id_prefix,
        "segment_count": len(ordered_segments),
        "segment_to_id": segment_to_id,
        "id_to_segment": id_to_segment,
        "id_to_edges": {
            segment_id: [int(edge_id) for edge_id in segment] for segment_id, segment in id_to_segment.items()
        },
    }


def _order_segments(compressed_subgraph: CompressedSubgraph, start_edge: int) -> List[Segment]:
    """Orders segments in a compressed subgraph based on their start edge.

    Args:
        compressed_subgraph (CompressedSubgraph): The compressed subgraph to order.
        start_edge (int): The start edge of the subgraph.

    Returns:
        List[Segment]: The ordered segments.
    """
    all_segments = iter_compressed_segments(compressed_subgraph)

    start_candidates = [segment for segment in all_segments if segment and segment[0] == start_edge]
    if start_candidates:
        start_segment = sorted(start_candidates, key=_segment_sort_key)[0]
    elif all_segments:
        start_segment = sorted(all_segments, key=_segment_sort_key)[0]
    else:
        return []

    ordered_segments = []
    visited = set()
    queue = deque([start_segment])

    while queue:
        segment = queue.popleft()
        if segment in visited:
            continue
        visited.add(segment)
        ordered_segments.append(segment)

        for neighbor in sorted(compressed_subgraph.get(segment, set()), key=_segment_sort_key):
            if neighbor not in visited:
                queue.append(neighbor)

    for segment in sorted(all_segments - visited, key=_segment_sort_key):
        ordered_segments.append(segment)

    return ordered_segments


def build_symbolic_subgraph(
    compressed_subgraph: CompressedSubgraph,
    edge_dicts: Dict[str, Dict[int, Any]],
    start_edge: int,
    dest_edge: int,
    segment_to_id: Dict[Segment, str],
    poi_catalog: Dict[int, Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Adds global symbolic IDs and readable semantics to a compressed edge-tuple graph.
    """
    ordered_segments = _order_segments(compressed_subgraph, start_edge)

    segments = {}
    segment_id_adjacency = {}

    for segment in ordered_segments:
        segment_id = segment_to_id[segment]
        details = describe_segment(segment, edge_dicts, poi_catalog=poi_catalog)

        details.update(
            {
                "segment_id": segment_id,
                "local_order": len(segments),
                "summary": (
                    f"{segment_id}: {details['display_name']} "
                    f"type: {'/'.join(details['road_types'])}, "
                    f"length: {details['length_m']}m, "
                    f"time: {details['travel_time_s']}s)"
                ),
            }
        )
        segments[segment_id] = details

        neighbors = sorted(compressed_subgraph.get(segment, set()), key=_segment_sort_key)
        segment_id_adjacency[segment_id] = [segment_to_id[n] for n in neighbors]

    destination_segment_ids = [segment_to_id[segment] for segment in ordered_segments if dest_edge in segment]

    return {
        "start_segment_id": segment_to_id.get(ordered_segments[0]) if ordered_segments else None,
        "destination_segment_ids": destination_segment_ids,
        "segments": segments,
        "segment_id_adjacency": segment_id_adjacency,
    }


if __name__ == "__main__":
    cprint("Starting subgraph construction...", "yellow")
    cprint(f"-PLACE NAME : {variables.place_name.capitalize()}", "green")
    cprint(f"-PATH TYPE: {variables.path_type}\n", "green")

    cprint("Loading graph...", "yellow")
    try:
        graph = load_graph(fname=variables.PICKLED_GRAPH)
        # Extract edges. The index here will be (u, v, key)
        graph_edges = ox.graph_to_gdfs(graph, nodes=False)

        # Create a safe lookup dictionary from the graph: (u, v, key) -> travel_time
        uvk_to_time = graph_edges["travel_time"].to_dict()
    except Exception as e:
        cprint(f"FATAL: Could not load graph: {e}", "red")
        exit(1)

    cprint("Loading edge data...", "yellow")
    edges_df = gpd.read_file(variables.EDGE_DATA)
    edges_df["travel_time"] = edges_df.apply(
        lambda row: uvk_to_time.get((row.u, row.v, row.key), 1.0), axis=1
    )
    # Clean up any remaining NaNs
    edges_df["travel_time"] = edges_df["travel_time"].fillna(1.0)

    cprint("Building edge ID to UVK dictionary...", "yellow")
    edge_id_to_uvk = {int(i): (row.u, row.v, row.key) for i, row in edges_df.iterrows()}

    if variables.path_type == "poi_aware":
        apply_poi_aware_weights(graph, edges_df)

    poi_catalog = {}
    catalog_file = poi_catalog_path()
    if Path(catalog_file).exists():
        with open(catalog_file, "rb") as f:
            poi_catalog = pickle.load(f)

    cprint("Building edge attribute dictionaries...", "yellow")
    edge_dicts = build_edge_attribute_dicts(graph, edges_df, edge_id_to_uvk)

    cprint("Loading seed paths...", "yellow")
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
    uncompressed_subgraphs = construct_local_subgraphs(
        graph, data, edge_id_to_uvk, top_k_shortest=variables.top_k_shortest, top_k=1, n_cores=16
    )

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

    cprint("Building deterministic global segment registry...", "yellow")
    segment_registry = build_global_segment_registry(final_compressed_graphs)
    cprint(f"Assigned {segment_registry['segment_count']} global segment IDs.", "green")

    final_symbolic_graphs = {}
    segment_to_id = segment_registry["segment_to_id"]

    cprint("Building symbolic subgraphs with global segment IDs...", "yellow")
    for od_pair, compressed_subgraph in tqdm(
        final_compressed_graphs.items(), desc="Symbolizing", dynamic_ncols=True
    ):
        start_edge, dest_edge = od_pair
        symbolic_subgraph = build_symbolic_subgraph(
            compressed_subgraph, edge_dicts, start_edge, dest_edge, segment_to_id, poi_catalog
        )
        final_symbolic_graphs[od_pair] = symbolic_subgraph

    Path(f"symbolic_subgraphs/{variables.path_type}").mkdir(parents=True, exist_ok=True)
    with open(f"symbolic_subgraphs/{variables.path_type}/{variables.place_name}_data", "wb") as f:
        pickle.dump(final_symbolic_graphs, f)

    with open(f"symbolic_subgraphs/{variables.path_type}/{variables.place_name}_segment_registry", "wb") as f:
        pickle.dump(segment_registry, f)

    cprint("Symbolic dual-representation subgraphs and segment registry saved successfully!", "green")
