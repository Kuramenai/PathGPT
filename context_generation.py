import json
import pickle
import variables
import geopandas as gpd
from bisect import bisect_left, bisect_right
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint
from utils import make_dir
from typing import Dict, Any, Tuple, List, Set

from data_preprocessing import clean_street_name
from data_augmentation import apply_poi_aware_weights, load_graph, poi_catalog_path
from subgraph_construction import (
    Segment,
    build_edge_attribute_dicts,
    build_global_segment_registry,
    build_symbolic_subgraph,
)


def generate_markdown_prompt(
    od_pair: Tuple[int, int],
    subgraph: Dict[str, Any],
    edge_dicts: Dict[str, dict],
) -> str:
    """
    Generates a localized Markdown prompt using the symbolic corridor graph.
    """
    start_edge, dest_edge = od_pair

    origin_name = clean_road_name(edge_dicts["edge_id_to_name"].get(start_edge, "未知起点"))
    dest_name = clean_road_name(edge_dicts["edge_id_to_name"].get(dest_edge, "未知终点"))
    start_segment_id = subgraph.get("start_segment_id")
    destination_segment_ids = set(subgraph.get("destination_segment_ids", []))
    segments = subgraph.get("segments", {})
    adjacency = subgraph.get("segment_id_adjacency", {})

    topology_lines = []
    for segment_id in ordered_segment_ids(segments):
        segment = segments[segment_id]
        road_types = format_values(segment.get("road_types", []))
        next_ids = adjacency.get(segment_id, [])
        next_text = ", ".join(next_ids) if next_ids else "无"

        role_labels = []
        if segment_id == start_segment_id:
            role_labels.append("起点段")
        if segment_id in destination_segment_ids:
            role_labels.append("终点段")
        role_text = f" ({'、'.join(role_labels)})" if role_labels else ""

        poi_extra = ""
        if segment.get("poi_count", 0):
            poi_names = segment.get("poi_names") or []
            poi_types = segment.get("poi_types") or []
            if poi_names:
                poi_extra += f"景点: {'、'.join(poi_names)} | "
            if poi_types:
                poi_extra += f"景点类型: {'、'.join(poi_types)} | "

        topology_lines.append(
            f"- {segment_id}{role_text}: {segment.get('display_name', '未知道路')} | "
            f"类型: {road_types} | "
            f"长度: {segment.get('length_m', 0)}m | "
            f"预计用时: {segment.get('travel_time_s', 0)}s | "
            f"邻近景点: {'是' if segment.get('near_poi') else '否'} | "
            f"景点数: {segment.get('poi_count', 0)} | "
            f"{poi_extra}"
            f"可连接到: {next_text}"
        )

    topology_str = "\n".join(topology_lines)
    destination_text = (
        ", ".join(
            sorted(
                destination_segment_ids,
                key=lambda segment_id: (
                    segments.get(segment_id, {}).get("local_order", float("inf")),
                    segment_id_sort_key(segment_id),
                ),
            )
        )
        or "未知"
    )

    markdown_prompt = f"""# 符号化路网拓扑
以下是连接起点 (**{origin_name}**) 和终点 (**{dest_name}**) 的局部路网。每个符号编号表示一个不可拆分的连续道路段；路线只能沿“可连接到”中列出的符号编号前进。


起点段: {start_segment_id}
终点段: {destination_text}

## 路段与连接关系
{topology_str}"""

    return markdown_prompt


def uncompressed_to_single_edge_segments(
    subgraph: Dict[int, Set[int]],
) -> Dict[Segment, Set[Segment]]:
    symbolic_adjacency: Dict[Segment, Set[Segment]] = {}
    for edge_id, next_edges in subgraph.items():
        segment = (int(edge_id),)
        symbolic_adjacency.setdefault(segment, set())
        for next_edge in next_edges:
            next_segment = (int(next_edge),)
            symbolic_adjacency[segment].add(next_segment)
            symbolic_adjacency.setdefault(next_segment, set())
    return symbolic_adjacency


def build_uncompressed_symbolic_subgraphs(
    uncompressed_subgraphs: Dict[Tuple[int, int], Dict[int, Set[int]]],
    edge_dicts: Dict[str, Dict[int, Any]],
    poi_catalog: Dict[int, Dict[str, str]],
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[str, Any]]:
    single_edge_graphs = {
        od_pair: uncompressed_to_single_edge_segments(subgraph)
        for od_pair, subgraph in uncompressed_subgraphs.items()
    }
    segment_registry = build_global_segment_registry(single_edge_graphs)
    segment_to_id = segment_registry["segment_to_id"]

    symbolic_subgraphs = {}
    for od_pair, single_edge_graph in tqdm(
        single_edge_graphs.items(),
        total=len(single_edge_graphs),
        dynamic_ncols=True,
        desc="Symbolizing uncompressed corridors",
    ):
        start_edge, dest_edge = od_pair
        symbolic_subgraphs[od_pair] = build_symbolic_subgraph(
            single_edge_graph,
            edge_dicts,
            start_edge,
            dest_edge,
            segment_to_id,
            poi_catalog,
        )

    return symbolic_subgraphs, segment_registry


def generate_embedding_text(
    od_pair: Tuple[int, int],
    subgraph: Dict[str, Any],
    edge_dicts: Dict[str, dict],
    intent_stats: Dict[str, List[float]] = None,
) -> str:
    """
    Generates a dense, natural-language summary of the symbolic subgraph optimized for E5 embeddings.
    """
    start_edge, dest_edge = od_pair
    origin_name = clean_road_name(edge_dicts["edge_id_to_name"].get(start_edge, "未知起点"))
    dest_name = clean_road_name(edge_dicts["edge_id_to_name"].get(dest_edge, "未知终点"))

    segments = subgraph.get("segments", {})
    # adjacency = subgraph.get("segment_id_adjacency", {})
    start_segment_id = subgraph.get("start_segment_id", "未知")
    destination_segment_ids = sorted(
        subgraph.get("destination_segment_ids", []),
        key=lambda segment_id: (
            segments.get(segment_id, {}).get("local_order", float("inf")),
            segment_id_sort_key(segment_id),
        ),
    )

    all_road_names = []
    seen_road_names = set()
    all_types = set()
    for segment_id in ordered_segment_ids(segments):
        segment = segments[segment_id]
        for road_name in segment.get("road_names", []):
            if road_name not in seen_road_names and road_name not in {"未知道路", "Unknown Road"}:
                seen_road_names.add(road_name)
                all_road_names.append(road_name)

        all_types.update(
            road_type for road_type in segment.get("road_types", []) if road_type not in {"未知", "N/A"}
        )

    features = extract_corridor_features(subgraph)
    intent_text = describe_intent_features(features, intent_stats) if intent_stats else ""

    road_signature = "，".join(all_road_names[:20]) or "未知道路"
    type_signature = "、".join(sorted(all_types)) or "未知类型"
    destination_text = "、".join(destination_segment_ids) or "未知"
    topology_signature = build_compact_topology_signature(subgraph, max_lines=12)

    embedding_text = (
        f"passage: 这是一个连接起点 {origin_name} 和终点 {dest_name} 的局部路网候选区域。 "
        f"符号化起点段是 {start_segment_id}，终点段是 {destination_text}。 "
        f"该走廊图包含 {features['segment_count']} 个连续道路段、{features['edge_count']} 条原始边、"
        f"{features['connection_count']} 条段间连接、{features['decision_count']} 个分叉决策点和 "
        f"{features['terminal_count']} 个终止段。 "
        f"该区域主要道路类型包括：{type_signature}。 "
        f"主要道路包括：{road_signature}。 "
        f"候选走廊总长度约为 {round(features['total_length_m'], 2)} 米，"
        f"基础通行时间约为 {round(features['total_time_s'], 2)} 秒。 "
        f"{intent_text}"
        f"紧凑拓扑连接摘要：{topology_signature}。 "
        f"该子图适合检索与起终点、道路名称、道路类型、通行代价和局部拓扑结构相似的路线规划上下文。"
    )

    return embedding_text


def extract_corridor_features(subgraph: Dict[str, Any]) -> Dict[str, float]:
    segments = subgraph.get("segments", {})
    adjacency = subgraph.get("segment_id_adjacency", {})

    total_length = 0.0
    total_time = 0.0
    edge_count = 0
    highway_length = 0.0
    arterial_length = 0.0
    local_length = 0.0

    for segment in segments.values():
        length = float(segment.get("length_m", 0) or 0)
        total_length += length
        total_time += float(segment.get("travel_time_s", 0) or 0)
        edge_count += int(segment.get("edge_count", 0) or 0)

        road_type_text = " ".join(str(t).lower() for t in segment.get("road_types", []))
        if any(token in road_type_text for token in ("motorway", "trunk", "expressway", "高速", "快速")):
            highway_length += length
        elif any(
            token in road_type_text for token in ("primary", "secondary", "tertiary", "主干", "次干", "干道")
        ):
            arterial_length += length
        elif any(
            token in road_type_text
            for token in ("residential", "living_street", "service", "unclassified", "住宅", "生活", "支路")
        ):
            local_length += length

    segment_count = len(segments)
    connection_count = sum(len(next_ids) for next_ids in adjacency.values())
    decision_count = sum(1 for next_ids in adjacency.values() if len(next_ids) > 1)
    terminal_count = sum(1 for next_ids in adjacency.values() if not next_ids)
    safe_length = total_length or 1.0

    return {
        "segment_count": segment_count,
        "edge_count": edge_count,
        "connection_count": connection_count,
        "decision_count": decision_count,
        "terminal_count": terminal_count,
        "total_length_m": total_length,
        "total_time_s": total_time,
        "avg_speed_mps": total_length / total_time if total_time > 0 else 0.0,
        "decision_density": decision_count / segment_count if segment_count else 0.0,
        "highway_ratio": highway_length / safe_length,
        "arterial_ratio": arterial_length / safe_length,
        "local_ratio": local_length / safe_length,
    }


def build_intent_stats(feature_rows: List[Dict[str, float]]) -> Dict[str, List[float]]:
    keys = (
        "segment_count",
        "total_length_m",
        "total_time_s",
        "avg_speed_mps",
        "decision_density",
        "highway_ratio",
        "arterial_ratio",
        "local_ratio",
    )
    return {key: sorted(row.get(key, 0.0) for row in feature_rows) for key in keys}


def percentile(value: float, sorted_values: List[float]) -> float:
    if len(sorted_values) < 2:
        return 0.5
    left = bisect_left(sorted_values, value)
    right = bisect_right(sorted_values, value)
    return (left + right) / (2 * len(sorted_values))


def describe_intent_features(features: Dict[str, float], stats: Dict[str, List[float]]) -> str:
    # ponytail: heuristic, corpus-relative intent labels. Upgrade with POI/fuel/trajectory features.
    low_time = 1 - percentile(features["total_time_s"], stats["total_time_s"])
    low_length = 1 - percentile(features["total_length_m"], stats["total_length_m"])
    low_segments = 1 - percentile(features["segment_count"], stats["segment_count"])
    high_speed = percentile(features["avg_speed_mps"], stats["avg_speed_mps"])
    low_decision_density = 1 - percentile(features["decision_density"], stats["decision_density"])
    low_highway = 1 - percentile(features["highway_ratio"], stats["highway_ratio"])
    high_arterial = percentile(features["arterial_ratio"], stats["arterial_ratio"])
    high_local = percentile(features["local_ratio"], stats["local_ratio"])
    medium_speed = 1 - abs(high_speed - 0.5) * 2

    scores = {
        "最快路线": mean_score(
            low_time, high_speed, max(high_arterial, 1 - low_highway), low_decision_density
        ),
        "最短路线": mean_score(low_length, low_segments, low_decision_density),
        "避开高速路线": mean_score(low_highway, high_local),
        "省油路线": mean_score(low_length, low_decision_density, medium_speed, low_highway),
        "观光路线": mean_score(high_local, low_highway, 1 - high_speed),
    }
    labels = "，".join(f"{name}={score_label(score)}" for name, score in scores.items())

    strongest = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:2]
    reasons = []
    for name, _ in strongest:
        if name == "最快路线":
            reasons.append("用时分位较低、平均速度较高")
        elif name == "最短路线":
            reasons.append("总长度较短、连续路段数量较少")
        elif name == "避开高速路线":
            reasons.append("高速或快速路占比较低")
        elif name == "省油路线":
            reasons.append("路线较短、分叉密度较低且速度更接近中等水平")
        elif name == "观光路线":
            reasons.append("本地道路占比较高且高速占比较低")

    return f"意图标签：{labels}。主要依据：{'；'.join(reasons)}。"


def mean_score(*values: float) -> float:
    return sum(values) / len(values) if values else 0.0


def score_label(score: float) -> str:
    if score >= 0.67:
        return "高"
    if score >= 0.4:
        return "中"
    return "低"


def clean_road_name(raw_name: Any) -> str:
    name = clean_street_name(raw_name)
    if name == "Unnamed Road":
        return "未知道路"
    return name


def segment_id_sort_key(segment_id: str) -> Tuple[int, Any]:
    segment_id_str = str(segment_id)
    prefix = segment_id_str.rstrip("0123456789")
    numeric_suffix = segment_id_str[len(prefix) :]
    if numeric_suffix:
        return (0, prefix, int(numeric_suffix))
    return (1, segment_id_str)


def ordered_segment_ids(segments: Dict[str, Dict[str, Any]]) -> List[str]:
    return sorted(
        segments.keys(),
        key=lambda segment_id: (
            segments[segment_id].get("local_order", float("inf")),
            segment_id_sort_key(segment_id),
        ),
    )


def format_values(values: Any, fallback: str = "未知") -> str:
    if values is None:
        return fallback
    if isinstance(values, str):
        values = [values]
    elif not isinstance(values, (list, tuple, set)):
        values = [values]
    cleaned_values = [str(value).strip() for value in values if str(value).strip()]
    cleaned_values = [value for value in cleaned_values if value not in {"N/A", "Unknown Road"}]
    return "、".join(cleaned_values) if cleaned_values else fallback


def build_compact_topology_signature(subgraph: Dict[str, Any], max_lines: int = 12) -> str:
    segments = subgraph.get("segments", {})
    adjacency = subgraph.get("segment_id_adjacency", {})
    topology_parts = []

    for segment_id in ordered_segment_ids(segments)[:max_lines]:
        next_ids = adjacency.get(segment_id, [])
        next_text = ",".join(next_ids) if next_ids else "END"
        topology_parts.append(f"{segment_id}->{next_text}")

    remaining = max(0, len(segments) - max_lines)
    if remaining:
        topology_parts.append(f"...另有{remaining}段")

    return "；".join(topology_parts) if topology_parts else "无"


if __name__ == "__main__":
    cprint("\n\nGENERATING MARKDOWN PROMPTS FOR :", "yellow", attrs=["bold"])
    cprint(f"-PLACE NAME : {variables.place_name.capitalize()}", "green")
    cprint(f"-PATH TYPE: {variables.path_type}\n", "green")

    if variables.corridor_graph_form == "uncompressed":
        cprint("-CORRIDOR GRAPH FORM: uncompressed\n", "green")
        cprint("Loading uncompressed subgraphs...", "yellow")
        subgraph_filename = f"uncompressed_subgraphs/{variables.path_type}/{variables.place_name}_data.pkl"
        try:
            with open(subgraph_filename, "rb") as f:
                uncompressed_subgraphs = pickle.load(f)
        except FileNotFoundError:
            cprint(
                f"Subgraph not found at {subgraph_filename}. Please run subgraph_construction.py first.",
                "red",
            )
            exit(1)

        cprint("Loading edge and graph attributes...", "yellow")
        edges_df = gpd.read_file(variables.EDGE_DATA)
        graph = load_graph(fname=variables.PICKLED_GRAPH)
        edge_id_to_uvk = {int(i): (row.u, row.v, row.key) for i, row in edges_df.iterrows()}
        if variables.path_type in ("poi_aware", "scenic"):
            apply_poi_aware_weights(graph, edges_df)

        poi_catalog = {}
        catalog_file = poi_catalog_path()
        if Path(catalog_file).exists():
            with open(catalog_file, "rb") as f:
                poi_catalog = pickle.load(f)

        edge_dicts = build_edge_attribute_dicts(graph, edges_df, edge_id_to_uvk)
        symbolic_subgraphs, segment_registry = build_uncompressed_symbolic_subgraphs(
            uncompressed_subgraphs,
            edge_dicts,
            poi_catalog,
        )

        symbolic_path = f"{variables.symbolic_subgraph_root}/{variables.path_type}"
        make_dir(symbolic_path)
        with open(f"{symbolic_path}/{variables.place_name}_data", "wb") as f:
            pickle.dump(symbolic_subgraphs, f)
        with open(f"{symbolic_path}/{variables.place_name}_segment_registry", "wb") as f:
            pickle.dump(segment_registry, f)

        cprint(f"Loaded {len(symbolic_subgraphs)} subgraphs successfully!\n", "light_green")
        cprint("Generating prompts...", "yellow")
        final_prompts_dataset = []
        intent_stats = build_intent_stats(
            [extract_corridor_features(subgraph) for subgraph in symbolic_subgraphs.values()]
        )

        for od_pair, subgraph in tqdm(
            symbolic_subgraphs.items(),
            total=len(symbolic_subgraphs),
            dynamic_ncols=True,
            desc="Generating Markdown prompts",
        ):
            final_prompts_dataset.append(
                {
                    "od_pair": od_pair,
                    "embedding_text": generate_embedding_text(od_pair, subgraph, edge_dicts, intent_stats),
                    "llm_prompt": generate_markdown_prompt(od_pair, subgraph, edge_dicts),
                }
            )

        path = f"json_files/{variables.path_type}"
        make_dir(path)
        output_name = f"{variables.place_name}{variables.context_name_suffix}_markdown_prompts.json"
        with open(f"{path}/{output_name}", "w", encoding="utf-8") as f:
            json.dump(final_prompts_dataset, f, indent=2, ensure_ascii=False)
        cprint(f"Prompts saved at {path}/{output_name}", "green")
        cprint(
            f"Successfully generated {len(final_prompts_dataset)} prompts ready for prompt_generation.py!",
            "light_green",
        )
        exit(0)

    cprint("Loading symbolic subgraphs...", "yellow")
    subgraph_filename = f"symbolic_subgraphs/{variables.path_type}/{variables.place_name}_data"
    try:
        with open(subgraph_filename, "rb") as f:
            symbolic_subgraphs = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Subgraph not found at {subgraph_filename} Please run subgraph_construction.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(symbolic_subgraphs)} subgraphs successfully!\n", "light_green")

    edges_df = gpd.read_file(variables.EDGE_DATA)
    cprint("Loading edge data...", "yellow")
    edge_dicts = {
        "edge_id_to_name": {i: row.get("name", "未知道路") for i, row in edges_df.iterrows()},
    }

    cprint("Generating prompts...", "yellow")
    final_prompts_dataset = []
    intent_stats = build_intent_stats(
        [extract_corridor_features(subgraph) for subgraph in symbolic_subgraphs.values()]
    )

    for od_pair, subgraph in tqdm(
        symbolic_subgraphs.items(),
        total=len(symbolic_subgraphs),
        dynamic_ncols=True,
        desc="Generating Markdown prompts",
    ):
        markdown_text = generate_markdown_prompt(od_pair, subgraph, edge_dicts)
        embedding_text = generate_embedding_text(od_pair, subgraph, edge_dicts, intent_stats)

        # Save the OD pair, embedding text and the formatted prompt together
        final_prompts_dataset.append(
            {
                "od_pair": od_pair,
                "embedding_text": embedding_text,
                "llm_prompt": markdown_text,
            }
        )

    path = f"json_files/{variables.path_type}"
    make_dir(path)

    # Save as JSON so it is easily parsed by prompt_generation.py retrieval/inference scripts
    with open(f"{path}/{variables.place_name}_markdown_prompts.json", "w", encoding="utf-8") as f:
        json.dump(final_prompts_dataset, f, indent=2, ensure_ascii=False)
    cprint(f"Prompts saved at {path}/{variables.place_name}_markdown_prompts.json", "green")
    cprint(
        f"Successfully generated {len(final_prompts_dataset)} prompts ready for prompt_generation.py!",
        "light_green",
    )
