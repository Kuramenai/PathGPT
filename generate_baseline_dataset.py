import pickle
import concurrent.futures
import variables
import osmnx as ox
import geopandas as gpd
import networkx as nx
from typing import List, Tuple
from tqdm import tqdm
from termcolor import cprint
from copy import deepcopy
from enhance_dataset import load_graph, map_u_v_to_edge_id, edge_id_to_node_id, edges_df
# from shapely.geometry import Point, LineString


def touristic_weight(u, v, data):
    """
    Combines a reward for touristic edges with a penalty for major highways.
    """
    length = data[0].get("length", 1)
    highway_type = data[0].get("highway", "")

    # 1. Check if the edge is touristic and apply a big discount
    # if data[0].get("is_touristic", False):
    if data[0]["is_touristic"]:
        # print(data[0]["name"])
        return length * 0.001  # Make this edge very attractive

    # 2. If not touristic, check if it's a major highway and apply a penalty
    if highway_type in ["motorway", "trunk"]:
        return length * 1000  # Make this edge very unattractive

    # 3. Otherwise, return its normal length
    return length


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
    touristic_path = nx.shortest_path(
        graph, start_node, destination_node, weight=touristic_weight
    )
    return (original_path, fastest_path, shortest_path, touristic_path)


if __name__ == "__main__":
    graph = load_graph(fname=variables.PICKLED_GRAPH)

    f = open(f"pois/{variables.place_name}_pois", "rb")
    pois = pickle.load(f)
    f.close()

    touristic_edges_count = 0
    # Ensure both GeoDataFrames use the same Coordinate Reference System (CRS).
    # Project to a local CRS for accurate distance measurement in meters.
    # Here we use the projected CRS of the edges as the standard.
    edges_df = edges_df.to_crs(3857)
    pois_proj = pois.to_crs(edges_df.crs)

    # Create a "zone of interest" around each POI by buffering them.
    # We'll create a 30-meter radius circle around each POI.
    poi_buffer = pois_proj.geometry.buffer(50)

    # Use a spatial join to efficiently find all edges that intersect these zones.
    # This is the key performance step. It's MUCH faster than a nested loop.
    # It returns the edges that are within our 30m buffer of a POI.
    touristic_edges = gpd.sjoin(
        edges_df,
        gpd.GeoDataFrame(geometry=poi_buffer),
        how="inner",
        predicate="intersects",
    )

    # Get the unique IDs (u, v, key) of these touristic edges
    touristic_edge_ids = list(touristic_edges.index)

    # Set a 'touristic' attribute on the edges in your original graph.
    # First, set a default of False for all edges.
    nx.set_edge_attributes(graph, False, "is_touristic")

    # Now, update the touristic edges to True.
    # We create a dictionary of {(u, v, key): True} for the edges we found.
    attrs = {}
    for u, v, key in zip(
        touristic_edges["u"], touristic_edges["v"], touristic_edges["key"]
    ):
        attrs[(u, v, key)] = True
    nx.set_edge_attributes(graph, attrs, "is_touristic")

    print(f"Marked {len(touristic_edge_ids)} edges as touristic.")

    filepath = f"neuromlr_test_data/{variables.place_name}_data"
    f = open(filepath, "rb")
    baseline_dataset = pickle.load(f)
    f.close()

    data = deepcopy(baseline_dataset)
    data = [t for (idx, t, timestamps) in tqdm(data, dynamic_ncols=True)]
    data = [edge_id_to_node_id(path) for path in tqdm(data, dynamic_ncols=True)]

    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        data = list(tqdm(executor.map(compute_paths, data), total=len(data)))

    # fmt:off
    fastest_paths_edges_ids,shortest_paths_edges_ids, touristic_paths_edges_ids = [], [], []
    for _, single_fastest_path_nodes_ids, single_shortest_path_nodes_ids, single_touristic_path_nodes_ids in tqdm(data, dynamic_ncols=True):
        single_fastest_path_edges_ids = [map_u_v_to_edge_id[(u, v)] for u, v in zip(single_fastest_path_nodes_ids, single_fastest_path_nodes_ids[1:])]
        single_shortest_path_edges_ids = [map_u_v_to_edge_id[(u, v)] for u, v in zip(single_shortest_path_nodes_ids, single_shortest_path_nodes_ids[1:])]
        single_touristic_path_edges_ids = [map_u_v_to_edge_id[(u, v)] for u, v in zip(single_touristic_path_nodes_ids, single_touristic_path_nodes_ids[1:])]
        fastest_paths_edges_ids.append(single_fastest_path_edges_ids)
        shortest_paths_edges_ids.append(single_shortest_path_edges_ids)
        touristic_paths_edges_ids.append(single_touristic_path_edges_ids)
    # fmt:on

    new_baseline_dataset = []
    for (
        path_info,
        generated_fastest_path,
        generated_shortest_path,
        generated_touristic_path,
    ) in tqdm(
        zip(
            baseline_dataset,
            fastest_paths_edges_ids,
            shortest_paths_edges_ids,
            touristic_paths_edges_ids,
        )
    ):
        new_baseline_dataset.append(
            [
                path_info,
                (path_info[0], generated_fastest_path, path_info[2]),
                (path_info[0], generated_shortest_path, path_info[2]),
                (path_info[0], generated_touristic_path, path_info[2]),
            ]
        )

    filepath = f"neuromlr_test_data/{variables.place_name}_all_data"
    with open(filepath, "wb") as f:
        pickle.dump(new_baseline_dataset, f)

    cprint(f"File saved at {filepath}", "green")
