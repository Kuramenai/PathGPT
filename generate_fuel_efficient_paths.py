import os
import pickle
import random
import shutil
import osmnx as ox
import geopandas as gpd
import networkx as nx
import concurrent.futures
import variables
from tqdm import tqdm
from termcolor import cprint
from geopy.geocoders import Nominatim
from utils import make_dir
from typing import List, Dict, Tuple
from copy import deepcopy

HIGHWAY_PENALTY = 10.0
highway_types_to_penalize = {"motorway", "motorway_link"}

edges_df = gpd.read_file(variables.EDGE_DATA)
map_edge_id_to_u_v = edges_df[["u", "v"]].to_numpy()
map_u_v_to_edge_id = {(u, v): i for i, (u, v) in enumerate(map_edge_id_to_u_v)}
edges_uv_road_names = edges_df[["fid", "u", "v", "name"]].to_numpy()
valid_names_first_level = ["路", "道", "街"]

valid_names_second_level = [
    "路",
    "道",
    "街",
    "巷",
    "桥",
    "条",
    "环",
    "速",
    "沿",
    "同",
]

not_valid_names = ["仓", "北", "区", "厂", "口", "园", "堤", "头", "庄", "廊", "期"]


def welcome_text():
    cprint("\n\nGENERATING DATASET FOR :", "light_yellow", attrs=["bold"])
    cprint(f"-PLACE NAME : {variables.place_name}", "green")
    # cprint(f"-USE FOR : {variables.dataset_usage}", "green")


def condense_edges(edge_route: List[int]) -> List[int]:
    """
    Sometimes a road can have different multiple edge ids, this function makes sure that each edge id is unique (one road one edge id)."

    Args:
        edge_route (List[int]): a list of edges ids.

    Returns:
        route (List[int]): a list of edges ids obtained edge_route where duplicate edges ids are removed.
    """

    global map_edge_id_to_u_v, map_u_v_to_edge_id
    route = [map_u_v_to_edge_id[tuple(map_edge_id_to_u_v[e])] for e in edge_route]
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
    return [map_edge_id_to_u_v[path[0]][0], map_edge_id_to_u_v[path[0]][1]] + [
        map_edge_id_to_u_v[edge][1] for edge in path[1:]
    ]


def load_graph(fname: str) -> any:
    """
    Load the graph associated with a road network.

    Args:
        fname (str): path of the graph file stored  locally.

    Returns:
        any: graph of the road network
    """

    cprint("\nLoading graph...", "light_yellow")
    f = open(variables.PICKLED_GRAPH, "rb")
    graph = pickle.load(f)
    # Add edge speeds and travel times as weights to road network graph in order to compute fastest and shortest paths
    ox.add_edge_speeds(graph)
    ox.add_edge_travel_times(graph)
    edges = ox.graph_to_gdfs(graph, nodes=False)[
        ["highway", "travel_time"]
    ].reset_index(drop=True)
    edges_with_missing_travel_values = 0
    for e in graph.edges(data=True):
        # Here we divide the length of each edge by 1000 just to maintain consistency with NeuroMLR (although I am not sure why they did that)
        e[2]["length"] = e[2]["length"] / 1000
        # The travel times are missing for some edges (equals to 0),
        # in that case we replace it with the mean value of the travel times of all edges of the same type
        if e[2]["travel_time"] == 0:
            edges_with_missing_travel_values += 1
            e[2]["travel_time"] = (
                edges.loc[edges["highway"] == e[2]["highway"]]["travel_time"]
                .mean()
                .round(1)
            )
    f.close()
    cprint(
        f"The number of edges with missing travel times found is {edges_with_missing_travel_values}"
    )
    cprint("Graph loaded successfully!", "green")
    return graph


def custom_weight(u, v, data):
    length = data.get("length", 1)
    highway = data.get("highway", "")
    if isinstance(highway, str):
        highway = [highway]
    if any(htype in highway_types_to_penalize for htype in highway):
        return length * HIGHWAY_PENALTY

    return length


def load_data(fname: str, less: bool = False, samples: int = 35_000) -> List[List[int]]:
    """
    Load the training data.

    Args:
        fname (str): path of the data to load
        less (bool, optional): _description_. Defaults to False.
        samples (int, optional): _description_. Defaults to 35_000.

    Returns:
        List[Tuple[List[int]]]: A list of historical trajectories given by a sequence of nodes ids.
    """
    cprint("\nLoading train data", "blue")
    f = open(fname, "rb")
    data = pickle.load(f)
    f.close()

    # Sampling data
    if less:
        random.shuffle(data)
        if samples < len(data):
            data = random.sample(data, samples)
        else:
            data = data

    # Make sure that each road corresponds to a unique edge it
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


def load_test_data(
    fname: str, less=False, samples=1000
) -> List[Tuple[int, List[int], Tuple[int, int]]]:
    """
    Load the training data.

    Args:
        fname (str): path of the data to load
        less (bool, optional): _description_. Defaults to False.
        samples (int, optional): _description_. Defaults to 35_000.

    Returns:
        List[Tuple[int, List[int], int]]: A list of historical trajectories where each element of the list is a tuple of three elements,
        idx : the index of the trajectory,
        t : the actual trajectory which is a sequence of edges ids,
        timestamps : start time.
    """
    cprint("\nLoading test data", "blue")
    f = open(fname, "rb")
    data = pickle.load(f)
    f.close()

    # Sampling data
    if less:
        random.shuffle(data)
        if samples < len(data):
            data = random.sample(data, samples)
        else:
            data = data

    # Make sure that each road corresponds to a unique edge it
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


def get_road_name_from_nominatim(edge_fid: int) -> str:
    """
    Given an edge id, we retrieve its linestring attribute from edges dataframe
    which is later used to reverse geocode the road (street) name associated with that edge.

    Args:
        edge_fid (int): id of an edge (assigned by OpenStreetMap)

    Returns:
        road_name (str) : road name retrieved from Nominatim
    """
    edge = edges_df.loc[edges_df["fid"] == edge_fid]
    edge_dict = edge.to_dict()
    edge_geometry = edge_dict["geometry"]
    edge_linestring = edge_geometry[edge_fid]
    y = edge_linestring.xy[0][-1]
    x = edge_linestring.xy[1][-1]
    geolocator = Nominatim(
        domain="127.0.0.1:8088", scheme="http"
    )  #'localhost:80/nominatim'http://127.0.0.1:8088
    address = geolocator.reverse((x, y), zoom=16)
    address_info = address.address.split(",")
    i = 0
    road_name = "N/A"
    # 东直门枢纽片区, 东直门街道, 首都功能核心区, 东城区, 北京市, 100010, 中国
    while i < len(address_info):
        info = address_info[i]
        if info[-1] in valid_names_first_level:
            road_name = info
            break
        i += 1

    # if road_name == "N/A":
    #     cprint(f"无效路名 : {edge_fid} : {info[-1]} : {address.address}", "red")

    return road_name


def get_path_with_road_names(
    path: List[int], map_uv_to_road_names: Dict[Tuple[int, int], str]
) -> List[str]:
    """
    Given a sequence of node ids, map each consecutive pair (source and target) to a road name.

    Args:
        path (List[int]): a sequence of node ids.
        map_uv_to_road_names (Dict[Tuple[int, int], str]): dictionary that map a pair of node ids to a road name.

    Returns:
        List[str]: a sequence of road names.
    """
    start, destination = 0, 1
    path_with_road_names = []
    while start < destination and destination < len(path):
        road_name = map_uv_to_road_names[(path[start], path[destination])]
        if road_name != "N/A":
            path_with_road_names.append(road_name.strip())
        elif road_name == "N/A":
            return []
        else:
            cprint("How is that possible", "red")
            return

        start = destination
        destination += 1

    # remove duplicates (a road(edge) may pass through different intersections(node))
    path_with_road_names = list(dict.fromkeys(path_with_road_names))
    return path_with_road_names


def clear_road_names(road_names_path: str) -> None:
    """
    Clears the previously extracted road names if found.

    Args:
        road_names_path (str): path of the locally stored roads names.
    """

    if os.path.exists(road_names_path):
        shutil.rmtree(road_names_path)
        cprint("✅ Roads names cleared", "green")
    else:
        cprint(f"No roads names found at {road_names_path}", "yellow")


def extract_road_names(road_names_file: str) -> List[str]:
    """
    Extract the road names for the edges dataframe, if the road name is not present in the dataframe, use Nominatim to reverse geocode it through the linestring attribute.

    Args:
        road_names_file (str): path of the locally stored roads names, if such file doesn't exist will be generated in saved in the given location.
    Returns:
        List[str]: road names extracted from the edges dataframe or obtained through Nominatim.
    """
    cprint("\nExtracting road names...", "light_yellow")

    # Directly return road names if they have been already be extracted
    # Otherwise we find the sreeet/road name associated to each edge ID
    # and map the starting and destination node of an edge to its corresponding road name
    if os.path.exists(road_names_file):
        f = open(road_names_file, "rb")
        road_names = pickle.load(f)
        f.close
    else:
        road_names = {}
        for _, u, v, name in tqdm(edges_uv_road_names, dynamic_ncols=True):
            if name is not None:
                if name[-1] in valid_names_first_level:
                    road_names[(u, v)] = name
                else:
                    road_names[(u, v)] = "N/A"
            else:
                road_names[(u, v)] = "N/A"

        make_dir("road_names")
        with open(road_names_file, "wb") as f:
            pickle.dump(road_names, f)

    cprint("Roads names extracted successfully!", "green")
    return road_names


def compute_paths(path: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """
    Given a path, computes the shortest and fastest path.

    Args:
        path (List[int]): List of nodes ids.

    Returns:
        Tuple[List[int], List[int], List[int]]: The original path, The newly computed fastest path, The newly computed shortest path.
    """
    "Compute fastest and shortest path"
    start_node, destination_node = path[0], path[-1]
    original_path = path
    fastest_path = nx.shortest_path(graph, start_node, destination_node, "travel_time")
    shortest_path = nx.shortest_path(graph, start_node, destination_node, "length")
    highway_free_path = nx.shortest_path(
        graph, start_node, destination_node, weight=custom_weight
    )
    return (original_path, fastest_path, shortest_path, highway_free_path)


def augment_data(
    data: List[Tuple[int, List[int], Tuple[int, int]]],
    road_names: Dict[Tuple[int, int], str],
    n_cores: int = 12,
) -> List[Dict[str, List[int]]]:
    """
    Generate new training dataset containing the original paths and the newly computed fastest and shortest paths using the names of the roads traversed instead of the nodes ids.

    Args:
        data (List[Tuple[int, List[int], int]]): A list of historical trajectories where each trajectory is expressed in the format of the following tuple (idx, t, timestamps).
        road_names (Dict[Tuple[int, int], str]): Dictionary that map a pair of node ids to a road name.
        n_cores (int, optional) : Number of cores to be used for computing the paths.

    Returns:
        Tuple[List[Dict[str, List[int]]], List[Tuple[int, List[int], int]]]: Tuple containing a list of dictionaries and a list of tuples.
        The list of dictionaries is a list where each dictionary contains the roads names of a historical path plus its fastest and shortest path as well.
        These can be retreived by using the keys "most_used", "fastest" and "shortest" respectively.
        In the list of tuples, each tuple is in the original format of the data i.e (idx, t, timestamps).
    """
    cprint("\nAugmenting train data...", "light_yellow")
    data_copy = deepcopy(data)
    # Each sample data is now a sequence of nodes ID, we will use these nodes to compute new paths later
    data = [t for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
    data = [edge_id_to_node_id(path) for path in tqdm(data, dynamic_ncols=True)]

    cprint("Generating new paths...", "light_green")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        data = list(
            tqdm(executor.map(compute_paths, data, chunksize=100), total=len(data))
        )

    cprint("Adding road names...", "light_green")
    neuromlr_train_data, dataset = [], []
    for path_index, (
        original_path,
        fastest_path,
        shortest_path,
        highway_free_path,
    ) in tqdm(enumerate(data), dynamic_ncols=True):
        original_path_with_road_names = get_path_with_road_names(
            original_path, road_names
        )
        fastest_path_with_road_names = get_path_with_road_names(
            fastest_path, road_names
        )
        shortest_path_with_road_names = get_path_with_road_names(
            shortest_path, road_names
        )

        highway_free_path_with_road_names = get_path_with_road_names(
            highway_free_path, road_names
        )

        if (
            len(original_path_with_road_names) >= 3
            and len(fastest_path_with_road_names) >= 3
            and len(shortest_path_with_road_names) >= 3
            and len(highway_free_path_with_road_names) >= 3
        ):
            # fmt: off
            path_collection = {}
            path_collection["original_path_road_names"] = original_path_with_road_names
            path_collection["fastest_path_road_names"] = fastest_path_with_road_names
            path_collection["shortest_path_road_names"] = shortest_path_with_road_names
            path_collection["highway_free_path_road_names"] = highway_free_path_with_road_names 
            dataset.append(path_collection)
            neuromlr_train_data.append(data_copy[path_index])
        else:
            continue
        # fmt: on

    cprint(
        f"The length of train_data is {len(dataset)}, while the length of neuromlr_train_data is {len(neuromlr_train_data)}",
        "yellow",
    )

    cprint("Data augmentation done!", "green")
    return dataset, neuromlr_train_data


def augment_test_data(
    data: List[Tuple[int, List[int], Tuple[int, int]]],
    road_names: Dict[Tuple[int, int], str],
    n_cores: int = 12,
) -> Tuple[List[Dict[str, List[int]]], List[Tuple[int, List[int], Tuple[int, int]]]]:
    """
    Generate new testing dataset containing the original paths and the newly computed fastest and shortest paths using the names of the roads traversed instead of the nodes ids.

    Args:
        data (List[Tuple[int, List[int], int]]): A list of historical trajectories where each trajectory is expressed in the format (idx, t, timestamps).
        road_names (Dict[Tuple[int, int], str]): Dictionary that map a pair of node ids to a road name.
        n_cores (int, optional): Number of cores to be used for computing the paths. Defaults to 12.

    Returns:
        Tuple[List[Dict[str, List[int]]], List[Tuple[int, List[int], int]]]: Tuple containing a list of dictionaries and a list of tuples.
        The list of dictionaries is a list where each dictionary contains the roads names of a historical path plus its fastest and shortest path as well.
        These can be retreived by using the keys "most_used", "fastest" and "shortest" respectively.
        In the list of tuples, each tuple is in the original format of the data i.e (idx, t, timestamps).
    """
    cprint("\nAugmenting test data...", "light_yellow")
    data_copy = deepcopy(data)
    # Each sample data is now a sequence of nodes ID, we will use these nodes to compute new paths later
    data = [t for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
    data = [edge_id_to_node_id(path) for path in tqdm(data, dynamic_ncols=True)]

    cprint("Generating new paths...", "light_green")
    # fmt: off
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_cores) as executor:
        data = list(tqdm(executor.map(compute_paths, data, chunksize=100), total=len(data)))
    # fmt: on

    cprint("Adding road names...", "light_green")
    dataset, neuromlr_test_data = [], []
    for path_index, (
        original_path,
        fastest_path,
        shortest_path,
        highway_free_path,
    ) in tqdm(enumerate(data), dynamic_ncols=True):
        original_path_with_road_names = get_path_with_road_names(
            original_path, road_names
        )
        fastest_path_with_road_names = get_path_with_road_names(
            fastest_path, road_names
        )
        shortest_path_with_road_names = get_path_with_road_names(
            shortest_path, road_names
        )

        highway_free_path_with_road_names = get_path_with_road_names(
            highway_free_path, road_names
        )

        if (
            len(original_path_with_road_names) >= 3
            and len(fastest_path_with_road_names) >= 3
            and len(shortest_path_with_road_names) >= 3
            and len(highway_free_path_with_road_names) >= 3
        ):
            # fmt: off
            path_collection = {}
            path_collection["original_path_road_names"] = original_path_with_road_names
            path_collection["fastest_path_road_names"] = fastest_path_with_road_names
            path_collection["shortest_path_road_names"] = shortest_path_with_road_names
            path_collection["highway_free_path_road_names"] = highway_free_path_with_road_names 
            dataset.append(path_collection)
            neuromlr_test_data.append(data_copy[path_index])
        else:
            continue
        # fmt: on
    cprint(
        f"The length of test_data is {len(dataset)}, while the length of neuromlr_test_data is {len(neuromlr_test_data)}",
        "yellow",
    )
    cprint("Data augmentation done!", "green")
    return dataset, neuromlr_test_data


def sanity_check(
    edges_paths: List[Tuple[int, List[int], Tuple[int, int]]],
    nodes_paths: Dict[str, List[int]],
) -> int:
    """
    Sanity check that the street names of the paths generated from the node ids are the same as the ones extracted from the edges ids.

    Args:
        edges_paths (List[Tuple[int, List[int], Tuple[int, int]]]): historical trajectories extracted from the original dataset.
        nodes_paths (Dict[str, List[int]]): new paths computed from the source and destination information provided from the historical trajectories.

    Returns:
        int: number of dissimilarities if found.
    """

    dissimilarities = 0
    for historical_path_info, path_collection in zip(edges_paths, nodes_paths):
        historical_path = list(
            dict.fromkeys(
                [edges_uv_road_names[edge_id][3] for edge_id in historical_path_info[1]]
            )
        )
        generated_path = path_collection["original_path_road_names"]
        if generated_path != historical_path:
            dissimilarities += 1

    return dissimilarities


def relabel_test_data(dataset, forward):
    for path_collection in dataset:
        path_collection["original_path_edges_id"] = [
            forward[edge_id] for edge_id in path_collection["original_path_edges_id"]
        ]
        path_collection["fastest_path_edges_id"] = [
            forward[edge_id] for edge_id in path_collection["fastest_path_edges_id"]
        ]
        path_collection["shortest_path_edges_id"] = [
            forward[edge_id] for edge_id in path_collection["shortest_path_edges_id"]
        ]
    return dataset


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


if __name__ == "__main__":
    welcome_text()
    train_data = load_data(fname=variables.TRAIN_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)
    test_data = load_test_data(fname=variables.TEST_TRIP_DATA_PICKLED_WITH_TIMESTAMPS)

    graph = load_graph(fname=variables.PICKLED_GRAPH)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    road_names_file = f"road_names/{variables.place_name}_road_names"
    road_names_file = os.path.join(script_dir, road_names_file)
    road_names = extract_road_names(road_names_file)

    augmented_train_data, neuromlr_train_data = augment_data(
        data=train_data, road_names=road_names, n_cores=16
    )
    augmented_test_data, neuromlr_test_data = augment_test_data(
        data=test_data, road_names=road_names, n_cores=16
    )

    train_dataset_dissimilarities = sanity_check(
        neuromlr_train_data, augmented_train_data
    )

    if train_dataset_dissimilarities > 0:
        cprint(
            f"Incoherence between paths with roads names extracted from edges ids ans paths with street names extracted from nodes ids.\
            Found {train_dataset_dissimilarities} dissimilarities in train data."
            "red",
        )
    else:
        cprint("Sanity check passed for test dataset", "green")

    test_dataset_dissimilarities = sanity_check(neuromlr_test_data, augmented_test_data)
    if test_dataset_dissimilarities > 0:
        cprint(
            f"Incoherence between paths with roads names extracted from edges ids ans paths with street names extracted from nodes ids.\
            Found {test_dataset_dissimilarities} dissimilarities in test data."
            "red",
        )
    else:
        cprint("Sanity check passed for train dataset", "green")

    save_file("train_data", file=augmented_train_data)
    save_file("test_data", file=augmented_test_data)
    save_file("neuromlr_test_data", file=neuromlr_test_data)
    save_file("neuromlr_train_data", file=neuromlr_train_data)
