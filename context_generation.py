import json
import pickle
import variables
import geopandas as gpd
import osmnx as ox
from tqdm import tqdm
from termcolor import cprint
from utils import make_dir
from typing import Dict, Any, Tuple, Set
from generate_custom_dataset import load_graph
from filter_custom_dataset import clean_street_name


def get_segment_details(
    edge_tuple: Tuple[int, ...],
    edge_id_to_name: Dict[int, str],
    edge_id_to_length: Dict[int, float],
    edge_id_to_time: Dict[int, float],
    edge_id_to_type: Dict[int, str],
) -> Tuple[str, float, float, str]:
    """
    Extracts and aggregates names, length, time, and type for a compressed edge sequence.
    """
    names = []
    for e in edge_tuple:
        raw_name = edge_id_to_name.get(e, "Unnamed Road")
        name = clean_street_name(raw_name)
        if name == "Unnamed Road":
            name = "未知道路"
        if not names or names[-1] != name:
            names.append(name)

    segment_name = " - ".join(names)
    total_length = sum(edge_id_to_length.get(e, 0) for e in edge_tuple)
    total_time = sum(edge_id_to_time.get(e, 0) for e in edge_tuple)

    types = list(set(edge_id_to_type.get(e, "未知") for e in edge_tuple))
    type_str = "/".join(types)

    return segment_name, round(total_length, 2), round(total_time, 2), type_str


def generate_markdown_prompt(
    od_pair: Tuple[int, int],
    subgraph: Dict[Tuple[int, ...], Set[Tuple[int, ...]]],
    edge_dicts: Dict[str, dict],
) -> str:
    """
    Generates a localized Chinese Markdown prompt using the compressed topological subgraph.
    """
    start_edge, dest_edge = od_pair

    origin_name = edge_dicts["edge_id_to_name"].get(start_edge, "未知起点")
    dest_name = edge_dicts["edge_id_to_name"].get(dest_edge, "未知终点")

    topology_lines = []

    # Build the structural topology string using Markdown lists
    for current_node, neighbors in subgraph.items():
        curr_name, _, _, _ = get_segment_details(current_node, **edge_dicts)

        topology_lines.append(f"- 从 **{curr_name}**:")

        if not neighbors:
            topology_lines.append("  - (终点)")
            topology_lines.append("")
            continue

        for nxt_node in neighbors:
            nxt_name, nxt_len, nxt_time, nxt_type = get_segment_details(nxt_node, **edge_dicts)
            topology_lines.append(
                f"  - 连接到 {nxt_name} (类型: {nxt_type}, 长度: {nxt_len}m, 预计用时: {nxt_time}s)"
            )
        topology_lines.append("")  # blank line for readability

    topology_str = "\n".join(topology_lines)

    task_translation = {
        "fuel_efficient": "最省油",
        "scenic": "风景最好",
        "fastest": "最快",
        "shortest": "最短",
    }
    chinese_task = task_translation.get(variables.path_type, "最优")

    # Assemble the final Markdown string
    markdown_prompt = f"""
                        # 路网拓扑
                        以下是连接起点 (**{origin_name}**) 和终点 (**{dest_name}**) 的局部路网。

                        {topology_str}。"""

    return markdown_prompt


def generate_embedding_text(
    od_pair: Tuple[int, int],
    subgraph: Dict[Tuple[int, ...], Set[Tuple[int, ...]]],
    edge_dicts: Dict[str, dict],
) -> str:
    """
    Generates a dense, natural-language summary of the subgraph optimized for E5 embeddings.
    """
    start_edge, dest_edge = od_pair
    origin_name = edge_dicts["edge_id_to_name"].get(start_edge, "未知起点")
    dest_name = edge_dicts["edge_id_to_name"].get(dest_edge, "未知终点")

    all_road_names = set()
    all_types = set()
    total_len = 0.0
    total_time = 0.0

    # Aggregate all unique data points from the subgraph
    for current_node, neighbors in subgraph.items():
        curr_name, c_len, c_time, c_type = get_segment_details(current_node, **edge_dicts)
        all_road_names.add(curr_name)
        all_types.update(c_type.split("/"))
        total_len += c_len
        total_time += c_time

        for nxt_node in neighbors:
            nxt_name, n_len, n_time, n_type = get_segment_details(nxt_node, **edge_dicts)
            all_road_names.add(nxt_name)
            all_types.update(n_type.split("/"))
            total_len += n_len
            total_time += n_time

    # Clean up sets
    all_road_names.discard("未知道路")
    all_types.discard("未知")
    all_types.discard("N/A")

    roads_str = "，".join(list(all_road_names)[:15])  # Cap at 15 to avoid massive token bloat
    types_str = "、".join(list(all_types))

    # Create a dense paragraph and prepend the mandatory E5 "passage: " prefix
    embedding_text = (
        f"passage: 这是一个连接起点 {origin_name} 和终点 {dest_name} 的局部路网候选区域。 "
        f"该区域主要包含的道路类型有：{types_str}。 "
        f"途径的主要道路包括：{roads_str}。 "
        f"该候选路网涉及的总长度约为 {round(total_len, 5)} 米，预计基础通行时间约为 {round(total_time, 5)} 秒。 "
        f"该子图包含了多条可供选择的连通路径，适合用于寻找最快、最短、最省油或风景最佳的导航路线。"
    )

    return embedding_text


if __name__ == "__main__":
    cprint("\n\nGENERATING MARKDOWN PROMPTS FOR :", "yellow", attrs=["bold"])
    cprint(f"-DATASET : {variables.place_name}", "green")
    cprint(f"-PATH TYPE: {variables.path_type}\n", "green")

    cprint("Loading data...", "light_yellow")
    subgraph_filename = f"compressed_subgraphs/{variables.path_type}/{variables.place_name}_data"
    try:
        with open(subgraph_filename, "rb") as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Subgraph not found at {subgraph_filename} Please run subgraph_construction.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(dataset)} subgraphs successfully!\n", "light_green")

    # Load graph
    try:
        graph = load_graph(fname=variables.PICKLED_GRAPH)
        # Extract edges. The index here will be (u, v, key)
        graph_edges = ox.graph_to_gdfs(graph, nodes=False)

        # Create a safe lookup dictionary from the graph: (u, v, key) -> travel_time
        uvk_to_time = graph_edges["travel_time"].to_dict()
    except Exception as e:
        cprint(f"FATAL: Could not load graph: {e}", "red")
        exit(1)

    edges_df = gpd.read_file(variables.EDGE_DATA)
    #   Safely map the travel times back to your original edges_df
    # We use a lambda function with a fallback of 0.0 to prevent NoneType errors later
    edges_df["travel_time"] = edges_df.apply(
        lambda row: uvk_to_time.get((row.u, row.v, row.key), 1.0), axis=1
    )

    # Clean up any remaining NaNs just to be completely safe
    edges_df["travel_time"] = edges_df["travel_time"].fillna(1.0)

    # edges_df = gpd.read_file(variables.EDGE_DATA)

    edge_dicts = {
        "edge_id_to_name": {i: row.get("name", "未知道路") for i, row in edges_df.iterrows()},
        "edge_id_to_length": {i: row.length for i, row in edges_df.iterrows()},
        "edge_id_to_time": {i: row.get("travel_time") for i, row in edges_df.iterrows()},
        "edge_id_to_type": {i: row.get("Type", "N/A") for i, row in edges_df.iterrows()},
    }

    final_prompts_dataset = []

    for od_pair, subgraph in tqdm(
        dataset.items(), total=len(dataset), dynamic_ncols=True, desc="Generating Markdown prompts"
    ):
        markdown_text = generate_markdown_prompt(od_pair, subgraph, edge_dicts)
        embedding_text = generate_embedding_text(od_pair, subgraph, edge_dicts)

        # Save the OD pair, embedding text and the formatted prompt together
        final_prompts_dataset.append(
            {"od_pair": od_pair, "embedding_text": embedding_text, "llm_prompt": markdown_text}
        )

    path = f"json_files/{variables.path_type}"
    make_dir(path)

    # Save as JSON so it is easily parsed by your retrieval/inference scripts
    with open(f"{path}/{variables.place_name}_markdown_prompts.json", "w", encoding="utf-8") as f:
        json.dump(final_prompts_dataset, f, indent=2, ensure_ascii=False)
