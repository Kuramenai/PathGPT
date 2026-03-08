# %%
import os
import re
import ast
import pickle
import random
import osmnx as ox
import networkx as nx
import concurrent.futures
import variables
import geopandas as gpd
from functools import partial
from itertools import groupby
from tqdm import tqdm
from termcolor import cprint
from utils import make_dir
from typing import List, Dict, Tuple
from copy import deepcopy

GRAPH = None


def welcome_text():
    cprint("\n\nGENERATING DATASET FOR :", "light_yellow", attrs=["bold"])
    cprint(f"-PLACE NAME : {variables.place_name}", "green")
    cprint(f"-USING CUSTOM PATH: {variables.path_type}", "green")
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


def load_graph(fname: str) -> any:
    """
    Load the graph associated with a road network.

    Args:
        fname (str): path of the graph file stored  locally.

    Returns:
        any: graph of the road network
    """

    cprint("\nLoading graph...", "light_yellow")
    try:
        with open(fname, "rb") as f:
            graph = pickle.load(f)
    except FileNotFoundError:
        cprint(f"ERROR: Graph file not found at {fname}", "red")
        raise
    except Exception as e:
        cprint(f"ERROR: Failed to load graph: {e}", "red")
        raise

    # Add edge speeds and travel times as weights to road network graph in order to compute fastest and shortest paths
    ox.add_edge_speeds(graph)
    ox.add_edge_travel_times(graph)

    edges = ox.graph_to_gdfs(graph, nodes=False)[
        ["highway", "travel_time"]
    ].reset_index(drop=True)
    edges["highway"] = edges["highway"].apply(
        lambda x: x[0] if isinstance(x, list) else x
    )

    edges_with_missing_travel_values = 0
    for u, v, k, data in graph.edges(keys=True, data=True):
        # Here we divide the length of each edge by 1000 just to maintain consistency with NeuroMLR (although I am not sure why they did that)
        data["length"] = data["length"] / 1000
        # The travel times are missing for some edges (equals to 0),
        # in that case we replace it with the mean value of the travel times of all edges of the same type
        if data["travel_time"] == 0:
            edges_with_missing_travel_values += 1
            data["travel_time"] = (
                edges.loc[edges["highway"] == data["highway"]]["travel_time"]
                .mean()
                .round(1)
            )

        data["fuel_cost"] = calculate_edge_fuel_efficiency_weight(u, v, data)

    f.close()
    cprint(
        f"The number of edges with missing travel times found is {edges_with_missing_travel_values}"
    )

    cprint("Graph loaded successfully!", "green")
    return graph


def load_data(fname: str, less: bool = False, samples: int = 35_000) -> List[List[int]]:
    """
    Load the data.

    Args:
        fname (str): path of the data to load
        less (bool, optional): _description_. Defaults to False.
        samples (int, optional): _description_. Defaults to 35_000.

    Returns:
        List[Tuple[List[int]]]: A list of historical trajectories given by a sequence of nodes ids.
    """
    cprint("\nLoading data", "blue")
    try:
        with open(fname, "rb") as f:
            data = pickle.load(f)
    except FileNotFoundError:
        cprint(f"ERROR: DATA file not found at {fname}", "red")
        raise
    except Exception as e:
        cprint(f"ERROR: Failed to load DATA: {e}", "red")
        raise

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


def clean_street_name(raw_name: any, prefer_chinese: bool = True) -> str:
    """
    Cleans messy OSMnx street names. Extracts the primary name from lists,
    splits bilingual strings, and removes secondary intersection names.

    Args:
        raw_name (any): _description_
        prefer_chinese (bool, optional): _description_. Defaults to True.

    Returns:
        str: clean street name
    """
    # 1. If the name is missing, just return "Unnamed Road"
    if raw_name is None:
        return "Unnamed Road"

    # 2. Safely extract the string(s) into a Python list
    name_list = []
    if isinstance(raw_name, list):
        name_list = raw_name
    elif isinstance(raw_name, str):
        if raw_name.startswith("[") and raw_name.endswith("]"):
            try:
                parsed_list = ast.literal_eval(raw_name)
                if isinstance(parsed_list, list):
                    name_list = parsed_list
            except (ValueError, SyntaxError):
                name_list = [raw_name]
        else:
            name_list = [raw_name]
    else:
        return "Unnamed Road"

    if not name_list:
        return "Unnamed Road"

    # 3. Handle Lists: e.g.,['Confucian Temple Street', '文庙街']
    chosen_name = name_list[0]
    for n in name_list:
        # Check if the string contains Chinese characters
        has_chinese = bool(re.search(r"[\u4e00-\u9fff]", n))
        if prefer_chinese and has_chinese:
            chosen_name = n
            break
        elif not prefer_chinese and not has_chinese:
            chosen_name = n
            break

    # 4. Clean Bilingual & Intersections: e.g., '嵩山路 - Songshan Road' or '友谊东路·东北新街'
    # Split by known OSM delimiters
    for delimiter in [" - ", "-", "·", "/"]:
        if delimiter in chosen_name:
            parts = chosen_name.split(delimiter)

            # Find the part that matches our language preference
            for part in parts:
                has_chinese = bool(re.search(r"[\u4e00-\u9fff]", part))
                if (prefer_chinese and has_chinese) or (
                    not prefer_chinese and not has_chinese
                ):
                    chosen_name = part.strip()
                    break
            else:
                # If neither strictly matches, just take the first part
                chosen_name = parts[0].strip()
            break  # Stop checking delimiters once we successfully split

    return chosen_name.strip()


def get_path_with_edges_names(
    edge_path: List[int], edge_id_to_edge_name: Dict[int, str]
) -> List[str]:
    """
    Given a sequence of edges ids, map each edge id to an edge name.

    Args:
        edge_path (List[int]): a sequence of edges ids.
        map_uv_to_edges_names (Dict[int, str]): dictionary that map a pair of node ids u, v and a key k to a road name.

    Returns:
        List[str]: a sequence of edges names.
    """
    path_with_edges_names = []
    for edge_id in edge_path:
        if edge_id in edge_id_to_edge_name:
            raw_name = edge_id_to_edge_name[edge_id]

            # Pass the messy string to our new cleaner function!
            # Change to prefer_chinese=False if English translations are available
            clean_name = clean_street_name(raw_name, prefer_chinese=True)

            path_with_edges_names.append(clean_name)
        else:
            path_with_edges_names.append("Unnamed Road")

    # Remove consecutive duplicates (so a car driving down the same road doesn't log it 5 times)
    path_with_edges_names = [k for k, g in groupby(path_with_edges_names)]

    return path_with_edges_names


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
    length_km = data.get("length", 0.1)

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
        speed_factor = 1.0  # The peak efficiency sweet spot (~30-45 mph)
    elif speed_kph < 90:
        speed_factor = 1.1
    else:
        speed_factor = 1.3

    # --- edge type ---
    highway = data.get("highway", [])
    if isinstance(highway, str):
        highway = [highway]

    highway_penalty = 1.0

    # Define stop penalties in equivalent SECONDS of idling/acceleration waste
    stop_penalty_seconds = 0.0

    if any(h in {"motorway", "motorway_link"} for h in highway):
        highway_penalty = 1.3
        stop_penalty_seconds = 0

    elif any(h in {"trunk", "trunk_link"} for h in highway):
        highway_penalty = 1.15
        stop_penalty_seconds = (
            5  # ~5 seconds equivalent fuel waste for slight traffic/merging
        )

    elif any(h in {"primary", "secondary"} for h in highway):
        stop_penalty_seconds = (
            15  # ~15 seconds equivalent fuel waste for traffic lights
        )

    elif any(h in {"residential", "living_street"} for h in highway):
        stop_penalty_seconds = (
            30  # ~30 seconds equivalent fuel waste for stop signs & pedestrians
        )

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
        Tuple[List[int], List[int], List[int]]: The original path, The newly computed fastest path, The newly computed shortest path.
    """
    global GRAPH
    graph = GRAPH

    start_node, destination_node = path[0], path[-1]
    original_path = path

    # Compute fastest, shortest path and a custom path.
    fastest_path = nx.shortest_path(graph, start_node, destination_node, "travel_time")
    shortest_path = nx.shortest_path(graph, start_node, destination_node, "length")
    highway_free_path = nx.shortest_path(
        graph, start_node, destination_node, weight=custom_weight
    )
    return (original_path, fastest_path, shortest_path, highway_free_path)


def augment_data(
    data: List[Tuple[int, List[int], Tuple[int, int]]],
    graph: nx.MultiDiGraph,
    custom_path_type: str = "fuel_efficient",
    n_cores: int = 12,
) -> List[Dict[str, List[int]]]:
    """
    Generate new dataset containing the original paths and the newly computed fastest and shortest paths using the names of the roads traversed instead of the nodes ids.

    Args:
        data (List[Tuple[int, List[int], int]]): A list of historical trajectories where each trajectory is expressed in the format of the following tuple (idx, t, timestamps).
        graph (nx.MultiGraph): MultiDiGraph representing the road network.
        n_cores (int, optional) : Number of cores to be used for computing the paths.

    Returns:
        [List[Dict[str, List[int]]]: List of dictionaries and a list of tuples.
        The list of dictionaries is a list where each dictionary contains the edges names of a historical path plus the fastest, shortest and custom paths.
    """
    cprint("\nAugmenting data...", "light_yellow")

    data_copy = deepcopy(data)

    # Each instance of the data will now be a sequence of nodes ID, we will use these nodes to compute new paths later
    data = [t for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]

    valid_data = []
    valid_data_copy = []

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
    cprint("Adding edges names...", "light_green")
    dataset = []
    for path_index, (original_path, fastest_path, shortest_path, custom_path,) in tqdm(enumerate(data), dynamic_ncols=True):
        
        path_collection = {}
        path_collection["idx"] = data_copy[path_index][0]
        path_collection["timestamps"] = data_copy[path_index][-1]

        path_collection["original_path_edges"] = data_copy[path_index][1]
        path_collection["fastest_path_edges"] = remap_to_edges(fastest_path, graph, weight_metric="travel_time")
        path_collection["shortest_path_edges"] = remap_to_edges(shortest_path, graph, weight_metric="length")
        path_collection["custom_path_edges"] = remap_to_edges(custom_path, graph, weight_metric=custom_weight_key)

        dataset.append(path_collection)
    # fmt: on

    cprint("Data augmentation done!", "green")
    return dataset


def compute_jaccard_similarity(
    path1: List[int], path2: List[int], edge_lengths: Dict[int, float]
) -> float:
    """Compute the Jaccard similarity score between two given paths based on the edges lengths.

    Args:
        path1 (List[int]): path1
        path2 (List[int]): path2
        edge_lengths (Dict[int, float]): a dictionary that maps edge ids to their lengths.

    Returns:
        int: _description_
    """

    s1 = set(path1)
    s2 = set(path2)
    intersection = s1.intersection(s2)
    union = s1.union(s2)

    intersection_length = sum(edge_lengths.get(e, 0) for e in intersection)
    union_length = sum(edge_lengths.get(e, 0) for e in union)

    if union_length == 0:
        return 0.0

    similarity = intersection_length / union_length

    return similarity


def filter_dataset(
    dataset: List[Dict[str, List[int]]],
    edge_lengths: Dict[int, float],
    edge_id_to_edge_name: Dict[int, str],
    diversity_threshold: float = 0.6,
    min_path_length: int = 3,
) -> List[Dict[str, List[int]]]:
    """Only keep paths that are relatively diverse and long enough

    Args:
        dataset (List[Dict[str, List[int]]]): dataset to be filtered.
        edge_lengths (Dict[int, float]): a dictionary that maps edge ids to their lengths.
        edge_id_to_edge_name (Dict[int, str]): a dictionary that maps edge ids to their corresponding edge names.
        diversity_threshold (int) : Determine the similarity between two paths.
        path_length (int) : Minimum length for a path to be considered.

    Returns:
        List[Dict[str, List[int]]]: _description_
    """
    filtered_dataset = []
    for path_collection in tqdm(dataset, dynamic_ncols=True):
        custom_path = path_collection["custom_path_edges"]
        original_path = path_collection["original_path_edges"]
        fastest_path = path_collection["fastest_path_edges"]
        shortest_path = path_collection["shortest_path_edges"]

        # fmt: off
        similarity = compute_jaccard_similarity(custom_path, original_path, edge_lengths)
        if similarity > diversity_threshold:
            continue

        similarity = compute_jaccard_similarity(custom_path, fastest_path, edge_lengths)
        if similarity > diversity_threshold:
            continue

        similarity = compute_jaccard_similarity(custom_path, shortest_path, edge_lengths)
        if similarity > diversity_threshold:
            continue


        original_path_with_edges_names  = get_path_with_edges_names(original_path, edge_id_to_edge_name)
        fastest_path_with_edges_names   = get_path_with_edges_names(fastest_path, edge_id_to_edge_name)
        shortest_path_with_edges_names  = get_path_with_edges_names(shortest_path, edge_id_to_edge_name)
        custom_path_with_edges_names    = get_path_with_edges_names(custom_path, edge_id_to_edge_name)

        # fmt: on
        if any(
            "Unnamed Road" in path
            for path in [
                original_path_with_edges_names,
                fastest_path_with_edges_names,
                shortest_path_with_edges_names,
                custom_path_with_edges_names,
            ]
        ):
            continue

        if any(
            len(path) < min_path_length
            for path in [
                original_path_with_edges_names,
                fastest_path_with_edges_names,
                shortest_path_with_edges_names,
                custom_path_with_edges_names,
            ]
        ):
            continue

        path_collection["original_path_edges_names"] = original_path_with_edges_names
        path_collection["fastest_path_edges_names"] = fastest_path_with_edges_names
        path_collection["shortest_path_edges_names"] = shortest_path_with_edges_names
        path_collection["custom_path_edges_names"] = custom_path_with_edges_names

        filtered_dataset.append(path_collection)

    return filtered_dataset


def sanity_check(
    neuromlr_paths: List[Tuple[int, List[int], Tuple[int, int]]],
    pathgpt_paths: Dict[str, List[int]],
) -> int:
    """
    Sanity check that the street names of the paths generated from the node ids are the same as the ones extracted from the edges ids.

    Args:
        neuromlr_paths (List[Tuple[int, List[int], Tuple[int, int]]]): historical trajectories extracted from the original dataset.
        pathgpt_paths (Dict[str, List[int]]): new paths computed from the source and destination information provided from the historical trajectories.

    Returns:
        int: number of dissimilarities if found.
    """

    dissimilarities = 0
    for original_path, path_collection in zip(neuromlr_paths, pathgpt_paths):
        original_edges = original_path[1]
        generated_path = path_collection["original_path_edges"]
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

    edge_id_to_edge_name = {
        i: row.get("name", "Unnamed Road") for i, row in edges_df.iterrows()
    }

    # Load data
    try:
        train_data = load_data(fname=variables.TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
        test_data = load_data(fname=variables.TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
    except Exception as e:
        cprint(f"FATAL: Could not load data: {e}", "red")
        exit(1)

    n_cores = os.cpu_count()
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

    filtered_train_data = filter_dataset(
        augmented_train_data, edge_id_to_edge_length, edge_id_to_edge_name
    )

    save_file("train_data", file=filtered_train_data)

    cprint(
        f"Filtered train data: {len(augmented_train_data)} → {len(filtered_train_data)}",
        "cyan",
    )

    if len(filtered_train_data) == 0:
        cprint("WARNING: Train dataset is empty after filtering!", "red")

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

    filtered_test_data = filter_dataset(
        augmented_test_data, edge_id_to_edge_length, edge_id_to_edge_name
    )

    cprint(
        f"Filtered train data: {len(augmented_test_data)} → {len(filtered_test_data)}",
        "cyan",
    )

    if len(filtered_test_data) == 0:
        cprint("WARNING: Train dataset is empty after filtering!", "red")

    save_file("test_data", file=filtered_test_data)

    cprint("\n" + "=" * 50, "light_green", attrs=["bold"])
    cprint("DATASET GENERATION COMPLETE!", "light_green", attrs=["bold"])
    cprint("=" * 50, "light_green")
