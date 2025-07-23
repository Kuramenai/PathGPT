import pickle
import concurrent.futures
import variables
import networkx as nx
from typing import List, Tuple
from tqdm import tqdm
from termcolor import cprint
from copy import deepcopy
from enhance_dataset import load_graph, map_u_v_to_edge_id, edge_id_to_node_id


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
    return (original_path, fastest_path, shortest_path)


if __name__ == "__main__":
    graph = load_graph(variables.PICKLED_GRAPH)
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
    fastest_paths_edges_ids,shortest_paths_edges_ids = [], []
    for _, single_fastest_path_nodes_ids, single_shortest_path_nodes_ids in tqdm(data, dynamic_ncols=True):
        single_fastest_path_edges_ids = [map_u_v_to_edge_id[(u, v)] for u, v in zip(single_fastest_path_nodes_ids, single_fastest_path_nodes_ids[1:])]
        single_shortest_path_edges_ids = [map_u_v_to_edge_id[(u, v)] for u, v in zip(single_shortest_path_nodes_ids, single_shortest_path_nodes_ids[1:])]
        fastest_paths_edges_ids.append(single_fastest_path_edges_ids)
        shortest_paths_edges_ids.append(single_shortest_path_edges_ids)
    # fmt:on

    new_baseline_dataset = []
    for path_info, generated_fastest_path, generated_shortest_path in tqdm(
        zip(baseline_dataset, fastest_paths_edges_ids, shortest_paths_edges_ids)
    ):
        new_baseline_dataset.append(
            [
                path_info,
                (path_info[0], generated_fastest_path, path_info[2]),
                (path_info[0], generated_shortest_path, path_info[2]),
            ]
        )

    filepath = f"neuromlr_test_data/{variables.place_name}_all_data"
    with open(filepath, "wb") as f:
        pickle.dump(new_baseline_dataset, f)

    cprint(f"File saved at {filepath}", "green")
