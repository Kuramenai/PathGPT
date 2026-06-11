import json
import re
import pickle
import variables
import numpy as np
import pandas as pd
import geopandas as gpd
from collections import defaultdict
from typing import Dict, Set
from termcolor import cprint
from filter_custom_dataset import clean_street_name


def extract_json_route(raw_text: str) -> list:
    """
    Uses regex to safely find and parse the JSON block,
    and unrolls compressed road segments.
    """
    try:
        # Look for everything between the first { and the last }
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            json_str = match.group(0)
            parsed_dict = json.loads(json_str)
            raw_route = parsed_dict.get("route", [])

            # --- UNROLLING STEP ---
            flattened_route = []
            for item in raw_route:
                # Split the bundled string by "-" and clean up whitespace
                sub_roads = [road.strip() for road in item.split("-")]
                flattened_route.extend(sub_roads)

            # Clean up any empty strings or the "未知道路" (Unknown Road) fallbacks
            clean_route = [r for r in flattened_route if r and r != "未知道路"]

            return clean_route

    except json.JSONDecodeError:
        pass  # Fall through to return empty list if parsing fails

    return []


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


if __name__ == "__main__":
    cprint("\n--- STARTING EVALUATION ---", "yellow", attrs=["bold"])

    # 1. Load the generated results from the previous vLLM script
    file_path = f"generated_paths/{variables.path_type}/"
    if variables.use_context:
        file_name = f"with_context_{variables.place_name}_top_{variables.number_of_docs_to_retrieve}"
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

    all_precisions = []
    all_recalls = []
    valid_routes_count = 0

    # 2. Process each output
    for i in range(len(raw_llm_outputs)):
        raw_text = raw_llm_outputs[i]

        # Extract the ground truth for this specific task
        gt_key = f"{variables.path_type}_path_edges_names"
        ground_truth_route = ground_truth_data[i][gt_key]

        # Parse the LLM's JSON prediction
        predicted_route = extract_json_route(raw_text)

        if predicted_route:
            valid_routes_count += 1

        # Calculate scores
        p, r = calculate_metrics(predicted_route, ground_truth_route)
        all_precisions.append(p)
        all_recalls.append(r)

    # 3. Aggregate and display results for the paper
    avg_precision = np.mean(all_precisions) * 100
    avg_recall = np.mean(all_recalls) * 100
    success_rate = (valid_routes_count / len(raw_llm_outputs)) * 100

    cprint("\nBuilding Topological Adjacency Graph...", "yellow")
    # Load OSM edges dataframe (make sure not to reset the index!)
    edges_df = gpd.read_file(variables.EDGE_DATA)
    name_adjacency_graph = build_name_adjacency_graph(edges_df)

    cprint("Evaluating generated routes...", "yellow")

    # Track the new metric
    topologically_valid_count = 0

    for i in range(len(raw_llm_outputs)):
        predicted_route = extract_json_route(raw_llm_outputs[i])

        if predicted_route:
            valid_routes_count += 1

            # --- THE NEW CHECK ---
            is_drivable = check_route_connectivity(predicted_route, name_adjacency_graph)
            if is_drivable:
                topologically_valid_count += 1

    cprint(f"\nResults for {variables.place_name} ({variables.path_type}):", "green", attrs=["bold"])
    print(f"Total Samples Tested: {len(raw_llm_outputs)}")
    print(f"Valid JSON Routes Generated: {success_rate:.2f}%")
    print("-" * 30)
    cprint(f"Average Precision: {avg_precision:.2f}%", "cyan")
    cprint(f"Average Recall:    {avg_recall:.2f}%", "cyan")
    cprint(f"Average Topologically Valid Count:    {topologically_valid_count:.2f}%", "cyan")
