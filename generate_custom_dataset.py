# %%
import os
import sys
import pickle
import random
import osmnx as ox
import networkx as nx
import concurrent.futures
import variables
import pandas as pd
import geopandas as gpd
import numpy as np
from functools import partial
from tqdm import tqdm
from termcolor import cprint
from utils import make_dir
from typing import List, Dict, Tuple
from copy import deepcopy

GRAPH = None


def welcome_text():
    cprint("\n\nGENERATING DATASET FOR :", "light_yellow", attrs=["bold"])
    cprint(f"-PLACE NAME : {variables.place_name.capitalize()}", "cyan")
    cprint(f"-USING CUSTOM PATH: {variables.path_type}", "cyan")
    # cprint(f"-USE FOR : {variables.dataset_usage}", "green")


def condense_edges(edge_route: List[int]) -> List[int]:
    """
    Sometimes an edge can have different multiple edge ids, this function makes sure that each edge id is unique (one road one edge id)."

    Args:
        edge_route (List[int]): a list of edges ids.

    Returns:
        route (List[int]): a list of edges ids obtained edge_route where duplicate edges ids are removed.
    """

    global edge_id_to_uvk, uvk_to_edge_id
    route = [uvk_to_edge_id[tuple(edge_id_to_uvk[e])] for e in edge_route]
    return route


def remove_loops(path: List[int]) -> List[int]:
    """
    Remove cycle from a path.

    Args:
        path (List[int]): a list of edges ids.

    Returns:
        List[int]: a list of edges ids without cycle i.e no edge id traversed twice.
    """
    reduced = []
    last_occ = {p: -1 for p in path}
    for i in range(len(path) - 1, -1, -1):
        if last_occ[path[i]] == -1:
            last_occ[path[i]] = i
    current = 0
    while current < len(path):
        reduced.append(path[current])
        current = last_occ[path[current]] + 1
    return reduced


def edge_id_to_node_id(path: List[int]) -> List[int]:
    """
    Transforms a list of edges ids to the corresponding list of nodes ids by returning the ids of the endpoints nodes of each edge.

    Args:
        path (List[int]): List of edges ids.

    Returns:
        List[int]: List of nodes ids.
    """
    nodes = []
    # 1. Determine the start node and the first shared node using the first two edges
    u1, v1, k1 = edge_id_to_uvk[path[0]]
    u2, v2, k2 = edge_id_to_uvk[path[1]]

    # If v1 connects to either end of the second edge, the direction is u1 -> v1
    if v1 == u2 or v1 == v2:
        nodes.extend([u1, v1])
    # If u1 connects to either end of the second edge, the direction is v1 -> u1
    elif u1 == u2 or u1 == v2:
        nodes.extend([v1, u1])
    else:
        # Edge case: Map matching failed, and the first two edges don't even touch
        raise ValueError(
            f"Disconnected edges at the start of path: edge {path[0]} and {path[1]}"
        )

    # 2. Iterate through the rest of the path, chaining the connecting nodes
    for edge_id in path[1:]:
        u, v, k = edge_id_to_uvk[edge_id]
        last_node = nodes[-1]

        if last_node == u:
            nodes.append(v)
        elif last_node == v:
            nodes.append(u)
        else:
            # Edge case: Disconnected edge found in the middle of the trajectory
            raise ValueError(
                f"Disconnected edge {edge_id} in path. Does not connect to node {last_node}."
            )

    return nodes
    # return [edge_id_to_uvk[path[0]][0], edge_id_to_uvk[path[0]][1]] + [
    #     edge_id_to_uvk[edge][1] for edge in path[1:]
    # ]


def load_graph(fname: str) -> nx.MultiDiGraph:
    """
    Load the graph associated with a road network.

    Args:
        fname (str): path of the graph file stored  locally.

    Returns:
        graph (nx.MultiDiGraph): graph of the road network
    """

    cprint("\nLoading graph...", "light_yellow")
    try:
        with open(fname, "rb") as f:
            graph = pickle.load(f)
    except FileNotFoundError:
        cprint(f"ERROR: Graph file not found at {fname}", "red")
        raise SystemExit(1)
    except Exception as e:
        cprint(f"ERROR: Failed to load graph: {e}", "red")
        raise SystemExit(1)

    # Add edge speeds and travel times as weights to road network graph in order to compute fastest and shortest paths
    ox.add_edge_speeds(graph)
    ox.add_edge_travel_times(graph)

    # fmt: off
    edges = ox.graph_to_gdfs(graph, nodes=False)[["highway", "travel_time", "speed_kph"]].reset_index(drop=True)

    edges["highway"] = edges["highway"].apply(lambda x: x[0] if isinstance(x, list) else x)

    default_speed = 40.0  # Default speed in kph if highway type is missing
    speed_lookup = (edges.groupby("highway")["speed_kph"].mean().to_dict())

    edges_with_missing_travel_values = 0
    for u, v, k, data in graph.edges(keys=True, data=True):
        
        speed_kph = data.get("speed_kph", default_speed)
        
        # The speeds values might be missing for some edges
        if pd.isna(speed_kph) or speed_kph == 0: 
            highway = data.get("highway", "unclassified")
            if isinstance(highway, list):
                highway = highway[0]
                
            # Grab the fallback speed
            speed_kph = speed_lookup.get(highway, default_speed)
            data["speed_kph"] = float(speed_kph) 
        
        # The travel times are missing for some edges (equals to 0),
        # in that case we replace it with the mean value of the travel times of all edges of the same type
        travel_time = data.get("travel_time", 0)
        if pd.isna(travel_time) or travel_time == 0:
            edges_with_missing_travel_values += 1
            
            speed_ms = float(speed_kph) / 3.6  # Convert km/h to m/s
            
            length_m = data.get("length", 100.0) 
            data["travel_time"] = length_m / speed_ms
        
        data["travel_time"] *= np.random.uniform(0.999, 1.001)
        
        data["fuel_cost"] = calculate_edge_fuel_efficiency_weight(u, v, data)
        data["fuel_cost"] *= np.random.uniform(0.999, 1.001)

    cprint(f"The number of edges with missing travel times found is {edges_with_missing_travel_values}")

    cprint("Graph loaded successfully!", "green")
    return graph

    # fmt: on


def load_data(
    fname: str, less: bool = False, samples: int = 35_000
) -> List[Tuple[int, List[int], int]]:
    """
    Load the data.

    Args:
        fname (str): path of the data to load
        less (bool, optional): _description_. Defaults to False.
        samples (int, optional): _description_. Defaults to 35_000.

    Returns:
        List[Tuple[int, List[int], int]]: A list of historical trajectories where each trajectory is in the format of (idx, path, timestamps).
    """
    cprint("\nLoading data", "blue")
    try:
        with open(fname, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        cprint(f"ERROR: DATA file not found at {fname}", "red")
        raise SystemExit(1)
    except Exception as e:
        cprint(f"ERROR: Failed to load DATA: {e}", "red")
        raise SystemExit(1)

    # Sampling data
    if less:
        random.shuffle(data)
        if samples < len(data):
            data = random.sample(data, samples)
        else:
            data = data

    # Make sure that each edge corresponds to a unique edge it
    data = [
        (idx, condense_edges(t), timestamps)
        for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)
    ]
    # Remove loops
    data = [
        (idx, remove_loops(t), timestamps)
        for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)
    ]
    # ignoring very small trips
    data = [
        (idx, t, timestamps)
        for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)
        if len(t) >= 5
    ]

    return data


def calculate_edge_fuel_efficiency_weight(u: int, v: int, data) -> int:
    """Based on the edges's length, average speed, and type, calculate and returns a custom cost/weight for it.

    Args:
        u (int): node ID of edge endpoint 1
        v (int): node ID of edge endpoint 2
        data (_type_): _description_

    Returns:
        int: new cost for selecting that edge
    """

    # Retrieves length
    length_m = data.get("length", 100.0)
    length_km = length_m / 1000.0

    # Retreives speed (kph)
    # Defaults to 50 km/h if missing
    speed_kph = data.get("speed_kph", 50.0)
    if speed_kph <= 0:
        speed_kph = 50.0

    # Base travel time in SECONDS
    # (Distance in km / Speed in km/h) * 3600 seconds/hour
    time_seconds = (length_km / speed_kph) * 3600

    # --- fuel efficiency curve (Internal Combustion Engine) ---
    if speed_kph < 30:
        speed_factor = 1.6
    elif speed_kph < 50:
        speed_factor = 1.1
    elif speed_kph < 70:
        speed_factor = 1.0  # The peak efficiency for ICE
    elif speed_kph < 90:
        speed_factor = 1.1
    else:
        speed_factor = 1.25

    # --- edge type ---
    highway = data.get("highway", [])
    if isinstance(highway, str):
        highway = [highway]

    highway_penalty = 1.0

    # Define stop penalties in equivalent SECONDS of idling/acceleration waste
    stop_penalty_seconds = 0.0

    if any(h in {"motorway", "motorway_link"} for h in highway):
        stop_penalty_seconds = 0

    elif any(h in {"trunk", "trunk_link"} for h in highway):
        stop_penalty_seconds = 5

    elif any(h in {"primary", "secondary"} for h in highway):
        stop_penalty_seconds = 15

    elif any(h in {"residential", "living_street"} for h in highway):
        stop_penalty_seconds = 30

    # Calculate final eco-cost
    cost = ((time_seconds * speed_factor) + stop_penalty_seconds) * highway_penalty

    return cost


def calculate_edge_touristic_weight():
    "Coming SOON"
    pass


def init_worker(graph: nx.MultiDiGraph) -> None:
    """
    Initializer function for the multiprocessing pool.
    Saves the large road network graph to the worker's local global memory
    so it doesn't need to be serialized and passed with every single task.

    Args:
        graph (nx.MultiDiGraph) : Road network MultiDiGraph.

    Returns:
        None
    """
    global GRAPH
    GRAPH = graph


def compute_paths(
    path: List[int], custom_weight: str
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Given a path, computes the shortest, fastest path and a custom path.

    Args:
        path (List[int]): List of nodes ids.
        custom_weight (str): custom_weight.

    Returns:
        Tuple[List[int], List[int], List[int]]: The historical path, The newly computed fastest path, The newly computed shortest path.
    """
    global GRAPH
    graph = GRAPH

    start_node, destination_node = path[0], path[-1]
    original_path = path
    # fmt: off
    # Compute fastest, shortest path and a custom path.
    try:
        fastest_path = nx.shortest_path(graph, start_node, destination_node, "travel_time")
        shortest_path = nx.shortest_path(graph, start_node, destination_node, "length")
        highway_free_path = nx.shortest_path(graph, start_node, destination_node, weight=custom_weight)
        
        return (original_path, fastest_path, shortest_path, highway_free_path)
    except nx.NetworkXNoPath:
        return []
    # fmt: on


def augment_data(
    data: List[Tuple[int, List[int], Tuple[int, int]]],
    graph: nx.MultiDiGraph,
    custom_path_type: str = "fuel_efficient",
    n_cores: int = 12,
) -> List[Dict[str, List[int]]]:
    """
    Generate new dataset containing the historical paths and the newly computed fastest, shortest and new custom paths.

    Args:
        data (List[Tuple[int, List[int], int]]): A list of historical trajectories where each trajectory is expressed in the format of the following tuple (idx, path, timestamps).
        graph (nx.MultiGraph): MultiDiGraph representing the road network.
        n_cores (int, optional) : Number of cores to be used for computing the paths.

    Returns:
        [List[Dict[str, List[int]]]: List of dictionaries and a list of tuples.
        The list of dictionaries is a list where each dictionary contains the edges names of a historical path plus the fastest, shortest and custom paths.
    """
    cprint("\nAugmenting data...", "light_yellow")

    data_copy = deepcopy(data)

    data = [t for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]

    valid_data = []
    valid_data_copy = []
    # Each instance of the data will now be a sequence of nodes ID, we will use these nodes to compute new paths later
    for i, path in tqdm(enumerate(data), total=len(data), dynamic_ncols=True):
        try:
            node_path = edge_id_to_node_id(path)
            valid_data.append(node_path)
            valid_data_copy.append(data_copy[i])
        except ValueError:
            pass  # Silently drop the broken GPS trajectory

    data = valid_data
    data_copy = valid_data_copy

    cprint("Generating new paths...", "light_green")

    custom_weight_key = (
        "fuel_cost" if custom_path_type == "fuel_efficient" else "touristic_value"
    )
    worker_func = partial(compute_paths, custom_weight=custom_weight_key)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=n_cores, initializer=init_worker, initargs=(graph,)
    ) as executor:
        data = list(
            tqdm(executor.map(worker_func, data, chunksize=100), total=len(data))
        )

    # fmt: off
    cprint("Building dataset...", "light_green")
    dataset = []
    for path_index, paths in tqdm(enumerate(data), total=len(data), dynamic_ncols=True):
        
        if not paths:
            continue
        
        original_path, fastest_path, shortest_path, custom_path = paths
        path_collection = {}
        path_collection["idx"] = data_copy[path_index][0]
        path_collection["timestamps"] = data_copy[path_index][-1]

        path_collection["historical_path_edges"] = data_copy[path_index][1]
        path_collection["fastest_path_edges"] = remap_to_edges(fastest_path, graph, weight_metric="travel_time")
        path_collection["shortest_path_edges"] = remap_to_edges(shortest_path, graph, weight_metric="length")
        path_collection[f"{custom_path_type}_path_edges"] = remap_to_edges(custom_path, graph, weight_metric=custom_weight_key)

        dataset.append(path_collection)
    # fmt: on

    cprint("Data augmentation done!", "green")
    return dataset


def sanity_check(
    neuromlr_paths: List[Tuple[int, List[int], Tuple[int, int]]],
    pathgpt_paths: Dict[str, List[int]],
) -> int:
    """
    Sanity check that the street names of the paths generated from the node ids are the same as the ones extracted from the edges ids.

    Args:
        neuromlr_paths (List[Tuple[int, List[int], Tuple[int, int]]]): historical trajectories extracted from the historical dataset.
        pathgpt_paths (Dict[str, List[int]]): new paths computed from the source and destination information provided from the historical trajectories.

    Returns:
        int: number of dissimilarities if found.
    """

    dissimilarities = 0
    for original_path, path_collection in zip(neuromlr_paths, pathgpt_paths):
        original_edges = original_path[1]
        generated_path = path_collection["historical_path_edges"]
        if generated_path != original_edges:
            dissimilarities += 1

    return dissimilarities


def remap_to_edges(
    nodes: List[int], graph: nx.MultiGraph, weight_metric: str
) -> List[int]:
    """
    Transforms a sequence of node IDs back into edge IDs.
    For multi-edges (parallel roads), it selects the specific edge
    that minimizes the given weight_metric.

    Args:
        nodes (List[int]): a path represented by a sequence of nodes IDs.
        graph (nx.MultiGraph): MultiDiGraph representing the road network.
        weight_metric (str): metric used to compute the path

    Returns:
        List[int]: The same path, but using a sequence of edges IDs.
    """
    edge_ids = []

    for u, v in zip(nodes, nodes[1:]):
        # 1. Handle Bidirectional Travel (Map Matching direction)
        if graph.has_edge(u, v):
            start, end = u, v
        elif graph.has_edge(v, u):
            start, end = v, u
        else:
            raise KeyError(f"No edge found connecting node {u} and node {v}")

        # 2. Get all parallel edges between these two nodes
        parallel_edges = graph[start][end]

        # 3. If there is only one edge, just grab its key
        if len(parallel_edges) == 1:
            best_key = list(parallel_edges.keys())[0]

        # 4. If there are multiple edges, find the one that minimized the weight metric
        else:
            # if callable(weight_metric):
            #     # If the metric is our custom_weight function
            #     best_key = min(
            #         parallel_edges.keys(),
            #         key=lambda k: weight_metric(start, end, parallel_edges[k]),
            #     )
            # else:
            # If the metric is a string like 'length' or 'travel_time'
            best_key = min(
                parallel_edges.keys(),
                key=lambda k: parallel_edges[k].get(weight_metric, float("inf")),
            )

        # 5. Look up the global edge ID using our unique (u, v, key) dictionary
        edge_ids.append(uvk_to_edge_id[(start, end, best_key)])

    return edge_ids


def save_file(file_path: str, file: any) -> None:
    """
    Saves a file

    Args:
        file_path (str): file path.
        file (any): _description_
    """
    make_dir(file_path)
    with open(f"{file_path}/{variables.place_name}_data", "wb") as f:
        pickle.dump(file, f)

    cprint(f"File saved at {file_path}/{variables.place_name}_data.", "green")


# %%

if __name__ == "__main__":
    welcome_text()

    # Load graph
    try:
        graph = load_graph(fname=variables.PICKLED_GRAPH)
    except Exception as e:
        cprint(f"FATAL: Could not load graph: {e}", "red")
        exit(1)

    edges_df = gpd.read_file(variables.EDGE_DATA)

    edge_id_to_uvk = {i: (row.u, row.v, row.key) for i, row in edges_df.iterrows()}
    uvk_to_edge_id = {(row.u, row.v, row.key): i for i, row in edges_df.iterrows()}
    edge_id_to_edge_length = {i: (row.length) for i, row in edges_df.iterrows()}

    # Load data
    try:
        train_data = load_data(fname=variables.TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
        test_data = load_data(fname=variables.TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
    except Exception as e:
        cprint(f"FATAL: Could not load data: {e}", "red")
        exit(1)

    # Set this value according to your system configuration! In our case setting it to to high caused freezing, so beware.
    # n_cores = os.cpu_count()
    n_cores = 8
    custom_path_type = variables.path_type

    # ===== TRAIN DATA =====
    cprint("\n" + "=" * 50, "light_yellow")
    cprint("PROCESSING TRAINING DATA", "light_yellow", attrs=["bold"])
    cprint("=" * 50, "light_yellow")

    augmented_train_data = augment_data(
        data=train_data,
        graph=graph,
        custom_path_type=custom_path_type,
        n_cores=n_cores,
    )

    save_file(f"train_data/{custom_path_type}", file=augmented_train_data)

    # ===== TEST DATA =====
    cprint("\n" + "=" * 50, "light_yellow")
    cprint("PROCESSING TEST DATA", "light_yellow", attrs=["bold"])
    cprint("=" * 50, "light_yellow")

    augmented_test_data = augment_data(
        data=test_data,
        graph=graph,
        custom_path_type=custom_path_type,
        n_cores=n_cores,
    )

    save_file(f"test_data/{custom_path_type}", file=augmented_test_data)

    cprint("\n" + "=" * 50, "light_green", attrs=["bold"])
    cprint("DATASET GENERATION COMPLETE!", "light_green", attrs=["bold"])
    cprint("=" * 50, "light_green")
