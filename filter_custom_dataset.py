import re
import ast
import pickle
import variables
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from itertools import groupby
from tqdm import tqdm
from termcolor import cprint
from typing import List, Dict, Tuple
from generate_custom_dataset import save_file


def calculate_dynamic_threshold(
    dataset: List[Dict[str, List[int]]],
    edge_lengths: Dict[int, float],
    discard_percentile: float = 90.0,
) -> Tuple[float, List[float]]:
    """
    Calculates the similarity threshold based on the distribution of the dataset.
    Finds the max similarity of the custom path to any baseline for each sample,
    and returns the value at the specified percentile.
    """
    max_similarities = []

    for path_collection in tqdm(
        dataset,
        total=len(dataset),
        desc="Calculating similarity distribution",
        dynamic_ncols=True,
    ):
        custom_path = path_collection[f"{variables.path_type}_path_edges"]
        historical_path = path_collection["historical_path_edges"]
        fastest_path = path_collection["fastest_path_edges"]
        shortest_path = path_collection["shortest_path_edges"]

        sim_original = compute_jaccard_similarity(
            custom_path, historical_path, edge_lengths
        )
        sim_fastest = compute_jaccard_similarity(
            custom_path, fastest_path, edge_lengths
        )
        sim_shortest = compute_jaccard_similarity(
            custom_path, shortest_path, edge_lengths
        )

        # We care about the maximum similarity to ANY of the baselines
        max_sim = max(sim_original, sim_fastest, sim_shortest)
        max_similarities.append(max_sim)

    # Calculate the threshold
    threshold = float(np.percentile(max_similarities, discard_percentile))
    return threshold, max_similarities


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

    # Remove consecutive duplicates (so a car driving down the same road doesn't log it mutliple times)
    path_with_edges_names = [k for k, g in groupby(path_with_edges_names)]

    return path_with_edges_names


def filter_dataset(
    dataset: List[Dict[str, List[int]]],
    edge_lengths: Dict[int, float],
    edge_id_to_edge_name: Dict[int, str],
    diversity_threshold: float = 0.6,
    min_path_length: int = 4,
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
    cprint(
        f"\nFiltering data with threshold: {diversity_threshold:.4f}...", "light_yellow"
    )
    filtered_dataset = []
    for path_collection in tqdm(
        dataset, total=len(dataset), desc="Filtering dataset", dynamic_ncols=True
    ):
        custom_path = path_collection[f"{variables.path_type}_path_edges"]
        historical_path = path_collection["historical_path_edges"]
        fastest_path = path_collection["fastest_path_edges"]
        shortest_path = path_collection["shortest_path_edges"]

        # fmt: off

        # 1. Skip if the custom route is virtually identical to ALL baselines
        # sim_historical = compute_jaccard_similarity(custom_path, historical_path, edge_lengths)
        # sim_fastest = compute_jaccard_similarity(custom_path, fastest_path, edge_lengths)
        # sim_shortest = compute_jaccard_similarity(custom_path, shortest_path, edge_lengths)

        # if max(sim_historical, sim_fastest, sim_shortest) > diversity_threshold:
        #     continue 

        historical_path_with_edges_names  = get_path_with_edges_names(historical_path, edge_id_to_edge_name)
        fastest_path_with_edges_names   = get_path_with_edges_names(fastest_path, edge_id_to_edge_name)
        shortest_path_with_edges_names  = get_path_with_edges_names(shortest_path, edge_id_to_edge_name)
        custom_path_with_edges_names    = get_path_with_edges_names(custom_path, edge_id_to_edge_name)

        # fmt: on
        if any(
            "Unnamed Road" in path
            for path in [
                historical_path_with_edges_names,
                fastest_path_with_edges_names,
                shortest_path_with_edges_names,
                custom_path_with_edges_names,
            ]
        ):
            continue

        if any(
            len(path) < min_path_length
            for path in [
                historical_path_with_edges_names,
                fastest_path_with_edges_names,
                shortest_path_with_edges_names,
                custom_path_with_edges_names,
            ]
        ):
            continue

        path_collection["historical_path_edges_names"] = (
            historical_path_with_edges_names
        )
        path_collection["fastest_path_edges_names"] = fastest_path_with_edges_names
        path_collection["shortest_path_edges_names"] = shortest_path_with_edges_names
        path_collection[f"{custom_path_type}_path_edges_names"] = (
            custom_path_with_edges_names
        )

        filtered_dataset.append(path_collection)

    return filtered_dataset


if __name__ == "__main__":
    custom_path_type = variables.path_type

    edges_df = gpd.read_file(variables.EDGE_DATA)

    edge_id_to_edge_length = {i: (row.length) for i, row in edges_df.iterrows()}

    # fmt: off
    edge_id_to_edge_name = {i: row.get("name", "Unnamed Road") for i, row in edges_df.iterrows()}

    # FILTERING METRICS
    if variables.place_name == "beijing":
        discard_percentile = 60.0  # For dynamic threshold calculation
    elif variables.place_name == "chengdu":
        discard_percentile = 60.0  # For dynamic threshold calculation
    elif variables.place_name == "harbin":
        discard_percentile = 60.0  # For dynamic threshold calculation
        
    min_path_length = 4

    # ===== TRAIN DATA =====
    cprint("\n" + "=" * 50, "light_yellow")
    cprint("PROCESSING TRAINING DATA", "light_yellow", attrs=["bold"])
    cprint("=" * 50, "light_yellow")

    train_data_filename = (f"train_data/{variables.path_type}/{variables.place_name}_data")

    try:
        with open(train_data_filename, "rb") as f:
            augmented_train_data = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Train data not found at {train_data_filename}! Please run generate_custom_dataset.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(augmented_train_data)} training samples.", "cyan")

    # 1. Calculate the dynamic threshold from the TRAIN data distribution
    dynamic_threshold, max_similarities = calculate_dynamic_threshold(
        augmented_train_data, 
        edge_id_to_edge_length, 
        discard_percentile=discard_percentile
    )


    # 2. Filter train data with dynamic threshold
    filtered_train_data = filter_dataset(
        augmented_train_data,
        edge_id_to_edge_length,
        edge_id_to_edge_name,
        diversity_threshold=dynamic_threshold,  # Use the dynamically calculated threshold
        min_path_length=min_path_length,
    )

    cprint(f"Filtered train data: {len(augmented_train_data)} → {len(filtered_train_data)}","cyan",)

    if len(filtered_train_data) == 0:
        cprint("WARNING: Train dataset is empty after filtering!", "red")

    filtered_train_data_save_path = f"filtered_train_data/{variables.path_type}"
    save_file(filtered_train_data_save_path, file=filtered_train_data)

    # # ===== TEST DATA =====
    cprint("\n" + "=" * 50, "light_yellow")
    cprint("PROCESSING TEST DATA", "light_yellow", attrs=["bold"])
    cprint("=" * 50, "light_yellow")

    test_data_filename = f"test_data/{variables.path_type}/{variables.place_name}_data"

    try:
        with open(test_data_filename, "rb") as f:
            augmented_test_data = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Test data not found at {test_data_filename}! Please run generate_custom_dataset.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(augmented_test_data)} test samples from.", "cyan")

    # 3. Filter test data with dynamic threshold
    filtered_test_data = filter_dataset(
        augmented_test_data,
        edge_id_to_edge_length,
        edge_id_to_edge_name,
        diversity_threshold=dynamic_threshold,  # Use the same threshold calculated from the train data
        min_path_length=min_path_length,
    )

    cprint(f"Filtered test data: {len(augmented_test_data)} → {len(filtered_test_data)}","cyan")

    if len(filtered_test_data) == 0:
        cprint("WARNING: Train dataset is empty after filtering!", "red")

    filtered_test_data_save_path = f"filtered_test_data/{variables.path_type}"
    save_file(filtered_test_data_save_path, file=filtered_test_data)
