import json
import pickle
import random
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import geopandas as gpd
import networkx as nx
import variables
from termcolor import cprint
from tqdm import tqdm

from evaluate_ranked_contexts import (
    OdPair,
    StrategyResult,
    StrategyStats,
    build_bridged_corridor_graph,
    build_corridor_edge_graph,
    build_edge_weights,
    build_full_edge_transition_graph,
    build_task_edge_weights,
    get_ground_truth_edges,
    ground_truth_in_retrieved_union,
    print_strategy_summary,
    print_weight_source,
    retrieved_od_pairs,
    retrieved_union_edge_set,
    search_edge_graph,
    task_weight_column,
)
from evaluate_single import build_edge_id_to_name, build_edge_id_to_uvk, safe_float
from generate_custom_dataset import apply_poi_aware_weights, load_graph
from utils import make_dir


SEGMENT_ID_PATTERN = re.compile(r"^G[0-9]+$")


def anchor_output_name() -> str:
    return (
        f"{variables.retrieval_type}_context_{variables.place_name}"
        f"{variables.context_name_suffix}_top_"
        f"{variables.number_of_docs_to_retrieve}_anchor_segments"
    )


def anchor_metadata_name() -> str:
    return (
        f"{variables.place_name}{variables.context_name_suffix}_prompts_{variables.retrieval_type}_top_"
        f"{variables.number_of_docs_to_retrieve}_anchor_segments.json"
    )


def extract_json_object(raw_text: str) -> dict:
    try:
        match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
    except json.JSONDecodeError:
        pass
    return {}


def has_anchor_segments_json(raw_text: str) -> bool:
    return isinstance(extract_json_object(raw_text).get("anchor_segments"), list)


def extract_anchor_segments(raw_text: str) -> List[str]:
    raw_segments = extract_json_object(raw_text).get("anchor_segments", [])
    if not isinstance(raw_segments, list):
        return []

    segment_ids = []
    seen = set()
    for raw_segment_id in raw_segments:
        segment_id = str(raw_segment_id).strip().upper()
        if not SEGMENT_ID_PATTERN.match(segment_id) or segment_id in seen:
            continue
        segment_ids.append(segment_id)
        seen.add(segment_id)
    return segment_ids


def filter_known_segments(
    segment_ids: Sequence[str],
    valid_segment_ids: Optional[Set[str]] = None,
) -> List[str]:
    known_segments = []
    seen = set()
    for segment_id in segment_ids:
        if valid_segment_ids is not None and segment_id not in valid_segment_ids:
            continue
        if segment_id in seen:
            continue
        known_segments.append(segment_id)
        seen.add(segment_id)
    return known_segments


def decode_segment_edges(
    segment_ids: Sequence[str],
    id_to_edges: Dict[str, List[int]],
) -> Set[int]:
    edge_ids = set()
    for segment_id in segment_ids:
        edge_ids.update(int(edge_id) for edge_id in id_to_edges.get(segment_id, []))
    return edge_ids


def retrieved_candidate_segment_ids(
    retrieved_pairs: Iterable[OdPair],
    symbolic_subgraphs: Dict[OdPair, dict],
) -> Set[str]:
    candidate_ids = set()
    for od_pair in retrieved_pairs:
        subgraph = symbolic_subgraphs.get(od_pair)
        if not subgraph:
            continue
        candidate_ids.update(str(segment_id) for segment_id in subgraph.get("segments", {}))
    return candidate_ids


def clipped_discount(value: float, default: float) -> float:
    discount = safe_float(value, default)
    if discount <= 0:
        return default
    return min(discount, 1.0)


def soft_prior_weight(
    base_weight: float,
    next_edge: int,
    rewarded_edge_sets: Sequence[Tuple[Set[int], float]],
    min_factor: float = 0.05,
) -> float:
    base_weight = max(safe_float(base_weight, 1.0), 1e-9)
    factor = 1.0
    for edge_ids, discount in rewarded_edge_sets:
        if next_edge in edge_ids:
            factor *= clipped_discount(discount, 1.0)
    factor = max(factor, min_factor)
    return max(base_weight * factor, 1e-9)


def search_full_graph_with_soft_priors(
    full_edge_graph: nx.DiGraph,
    start_edge: Optional[int],
    dest_edge: Optional[int],
    rewarded_edge_sets: Sequence[Tuple[Set[int], float]],
) -> List[int]:
    if start_edge is None or dest_edge is None:
        return []

    start_edge = int(start_edge)
    dest_edge = int(dest_edge)
    if start_edge == dest_edge:
        return [start_edge] if start_edge in full_edge_graph else []
    if start_edge not in full_edge_graph or dest_edge not in full_edge_graph:
        return []

    def weight(_prev_edge: int, next_edge: int, edge_attrs: dict) -> float:
        return soft_prior_weight(
            edge_attrs.get("weight", 1.0),
            int(next_edge),
            rewarded_edge_sets,
        )

    try:
        return [
            int(edge_id)
            for edge_id in nx.shortest_path(
                full_edge_graph,
                start_edge,
                dest_edge,
                weight=weight,
            )
        ]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def random_matched_anchor_segments(
    anchor_segment_ids: Sequence[str],
    candidate_segment_ids: Set[str],
    id_to_edges: Dict[str, List[int]],
    seed: int,
) -> List[str]:
    if not anchor_segment_ids or not candidate_segment_ids:
        return []

    rng = random.Random(seed)
    available = sorted(segment_id for segment_id in candidate_segment_ids if segment_id not in anchor_segment_ids)
    if not available:
        available = sorted(candidate_segment_ids)

    random_segments = []
    used = set()
    for anchor_segment_id in anchor_segment_ids:
        target_len = len(id_to_edges.get(anchor_segment_id, []))
        pool = [segment_id for segment_id in available if segment_id not in used]
        if not pool:
            break

        pool = sorted(
            pool,
            key=lambda segment_id: (
                abs(len(id_to_edges.get(segment_id, [])) - target_len),
                segment_id,
            ),
        )
        close_pool = pool[: min(10, len(pool))]
        selected_segment_id = rng.choice(close_pool)
        random_segments.append(selected_segment_id)
        used.add(selected_segment_id)

    return random_segments


def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _self_check() -> None:
    assert extract_anchor_segments('{"anchor_segments": ["g1", "G2", "bad", "G1"]}') == [
        "G1",
        "G2",
    ]
    assert filter_known_segments(["G1", "G2"], {"G2"}) == ["G2"]
    assert decode_segment_edges(["G1", "G3"], {"G1": [1, 2], "G3": [2, 4]}) == {1, 2, 4}

    graph = nx.DiGraph()
    graph.add_edge(1, 2, weight=10.0)
    graph.add_edge(2, 4, weight=1.0)
    graph.add_edge(1, 3, weight=1.0)
    graph.add_edge(3, 4, weight=1.0)
    assert search_edge_graph(graph, 1, 4) == [1, 3, 4]
    assert search_full_graph_with_soft_priors(graph, 1, 4, [({2}, 0.05)]) == [1, 2, 4]

    random_segments = random_matched_anchor_segments(
        ["G10"],
        {"G1", "G2", "G3"},
        {"G10": [1, 2], "G1": [3], "G2": [4, 5], "G3": [6, 7, 8]},
        seed=1,
    )
    assert len(random_segments) == 1
    assert random_segments[0] in {"G1", "G2", "G3"}


def evaluate_paper_v2() -> None:
    cprint("\n--- STARTING PAPER_V2 SOFT-PRIOR EVALUATION ---", "yellow", attrs=["bold"])
    if variables.llm_task != "anchor_segments":
        cprint(
            "This evaluator loads anchor-segment outputs. Run prompt_generation.py/inference.py "
            "with -llm_task anchor_segments before evaluating.",
            "yellow",
        )

    generated_path = f"generated_paths/{variables.path_type}/{anchor_output_name()}"
    metadata_path = f"retrieval_metadata/{variables.path_type}/{anchor_metadata_name()}"
    test_data_path = f"filtered_test_data/{variables.path_type}/{variables.place_name}_data"
    uncompressed_subgraph_path = (
        f"uncompressed_subgraphs/{variables.path_type}/{variables.place_name}_data.pkl"
    )
    symbolic_subgraph_path = (
        f"{variables.symbolic_subgraph_root}/{variables.path_type}/{variables.place_name}_data"
    )
    segment_registry_path = (
        f"{variables.symbolic_subgraph_root}/{variables.path_type}/"
        f"{variables.place_name}_segment_registry"
    )

    try:
        raw_llm_outputs = _load_pickle(generated_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        ground_truth_data = _load_pickle(test_data_path)
        uncompressed_subgraphs = _load_pickle(uncompressed_subgraph_path)
        symbolic_subgraphs = _load_pickle(symbolic_subgraph_path)
        segment_registry = _load_pickle(segment_registry_path)
    except FileNotFoundError as exc:
        cprint(f"Error loading files: {exc}", "red")
        raise SystemExit(1)

    if variables.place_name == "chengdu":
        ground_truth_data = ground_truth_data[:1_500]

    metadata_queries = metadata.get("queries", [])
    if len(raw_llm_outputs) != len(ground_truth_data) or len(raw_llm_outputs) != len(metadata_queries):
        cprint(
            "Input length mismatch among generated outputs, ground truth data, and retrieval metadata.",
            "red",
        )
        raise SystemExit(1)

    id_to_edges = segment_registry.get("id_to_edges", {})
    all_segment_ids = set(id_to_edges)

    cprint("Loading edge data and building search graphs...", "yellow")
    edges_df = gpd.read_file(variables.EDGE_DATA)
    edge_id_to_name = build_edge_id_to_name(edges_df)
    edge_id_to_uvk = build_edge_id_to_uvk(edges_df)

    cprint("Loading pickled graph for task-specific edge weights...", "yellow")
    graph = load_graph(fname=variables.PICKLED_GRAPH)
    if variables.path_type in ("poi_aware", "scenic"):
        apply_poi_aware_weights(graph, edges_df)

    weight_column = task_weight_column(variables.path_type)
    edge_weights, weight_meta = build_task_edge_weights(
        edges_df,
        edge_id_to_uvk,
        graph,
        weight_column,
    )
    length_column = "length" if "length" in edges_df.columns else weight_column
    edge_lengths = build_edge_weights(edges_df, length_column)
    full_edge_graph = build_full_edge_transition_graph(edges_df, edge_weights)
    print_weight_source(weight_meta, variables.path_type)

    retrieved_discount = clipped_discount(variables.soft_retrieved_discount, 0.85)
    anchor_discount = clipped_discount(variables.soft_anchor_discount, 0.65)
    cprint(
        f"Soft-prior discounts: retrieved_union={retrieved_discount}, anchors={anchor_discount}",
        "cyan",
    )

    strategy_stats = {
        "full_graph_task_objective_oracle": StrategyStats(),
        "retrieved_top_k_union": StrategyStats(),
        "retrieved_top_k_union_bridged": StrategyStats(),
        "full_graph_retrieved_union_soft_prior": StrategyStats(),
        "full_graph_llm_anchor_soft_prior": StrategyStats(),
        "full_graph_retrieved_union_llm_anchor_soft_prior": StrategyStats(),
        "full_graph_random_anchor_soft_prior": StrategyStats(),
    }

    samples = []
    gt_in_retrieved_union_count = 0
    valid_anchor_json_count = 0
    non_empty_anchor_count = 0
    non_empty_random_anchor_count = 0
    invalid_or_out_of_context_anchor_count = 0

    for i in tqdm(range(len(raw_llm_outputs)), dynamic_ncols=True, desc="Evaluating"):
        raw_text = raw_llm_outputs[i]
        path_collection = ground_truth_data[i]
        query_metadata = metadata_queries[i]

        ground_truth_edges = get_ground_truth_edges(path_collection)
        ground_truth_route = path_collection.get(f"{variables.path_type}_path_edges_names", [])
        start_edge = ground_truth_edges[0] if ground_truth_edges else None
        dest_edge = ground_truth_edges[-1] if ground_truth_edges else None

        retrieved_pairs = retrieved_od_pairs(query_metadata)
        retrieved_edges = retrieved_union_edge_set(retrieved_pairs, uncompressed_subgraphs)
        if ground_truth_in_retrieved_union(
            ground_truth_edges,
            retrieved_pairs,
            uncompressed_subgraphs,
        ):
            gt_in_retrieved_union_count += 1

        candidate_segment_ids = retrieved_candidate_segment_ids(retrieved_pairs, symbolic_subgraphs)
        raw_anchor_ids = extract_anchor_segments(raw_text)
        globally_known_anchor_ids = filter_known_segments(raw_anchor_ids, all_segment_ids)
        anchor_segment_ids = filter_known_segments(globally_known_anchor_ids, candidate_segment_ids)
        invalid_or_out_of_context_anchor_count += len(globally_known_anchor_ids) - len(anchor_segment_ids)
        if has_anchor_segments_json(raw_text):
            valid_anchor_json_count += 1
        if anchor_segment_ids:
            non_empty_anchor_count += 1

        anchor_edges = decode_segment_edges(anchor_segment_ids, id_to_edges)
        random_anchor_ids = random_matched_anchor_segments(
            anchor_segment_ids,
            candidate_segment_ids,
            id_to_edges,
            seed=variables.random_anchor_seed + i,
        )
        random_anchor_edges = decode_segment_edges(random_anchor_ids, id_to_edges)
        if random_anchor_ids:
            non_empty_random_anchor_count += 1

        union_graph = build_corridor_edge_graph(
            retrieved_pairs,
            uncompressed_subgraphs,
            edge_weights,
        )
        union_route = search_edge_graph(union_graph, start_edge, dest_edge)
        bridged_union_graph = build_bridged_corridor_graph(
            union_graph,
            full_edge_graph,
            start_edge,
            dest_edge,
        )
        bridged_union_route = search_edge_graph(bridged_union_graph, start_edge, dest_edge)
        full_route = search_edge_graph(full_edge_graph, start_edge, dest_edge)
        retrieved_soft_route = search_full_graph_with_soft_priors(
            full_edge_graph,
            start_edge,
            dest_edge,
            [(retrieved_edges, retrieved_discount)],
        )
        anchor_soft_route = search_full_graph_with_soft_priors(
            full_edge_graph,
            start_edge,
            dest_edge,
            [(anchor_edges, anchor_discount)],
        )
        retrieved_anchor_soft_route = search_full_graph_with_soft_priors(
            full_edge_graph,
            start_edge,
            dest_edge,
            [(retrieved_edges, retrieved_discount), (anchor_edges, anchor_discount)],
        )
        random_anchor_soft_route = search_full_graph_with_soft_priors(
            full_edge_graph,
            start_edge,
            dest_edge,
            [(random_anchor_edges, anchor_discount)],
        )

        strategy_results = {
            "full_graph_task_objective_oracle": StrategyResult(
                edge_route=full_route,
                source="full_graph_task_objective_oracle" if full_route else "none",
                search_graph=full_edge_graph,
            ),
            "retrieved_top_k_union": StrategyResult(
                edge_route=union_route,
                source="retrieved_top_k_union" if union_route else "none",
                search_graph=union_graph,
            ),
            "retrieved_top_k_union_bridged": StrategyResult(
                edge_route=bridged_union_route,
                source="retrieved_top_k_union_bridged" if bridged_union_route else "none",
                search_graph=bridged_union_graph,
            ),
            "full_graph_retrieved_union_soft_prior": StrategyResult(
                edge_route=retrieved_soft_route,
                source="full_graph_retrieved_union_soft_prior" if retrieved_soft_route else "none",
                search_graph=full_edge_graph,
            ),
            "full_graph_llm_anchor_soft_prior": StrategyResult(
                edge_route=anchor_soft_route,
                source="full_graph_llm_anchor_soft_prior" if anchor_soft_route else "none",
                search_graph=full_edge_graph,
            ),
            "full_graph_retrieved_union_llm_anchor_soft_prior": StrategyResult(
                edge_route=retrieved_anchor_soft_route,
                source=(
                    "full_graph_retrieved_union_llm_anchor_soft_prior"
                    if retrieved_anchor_soft_route
                    else "none"
                ),
                search_graph=full_edge_graph,
            ),
            "full_graph_random_anchor_soft_prior": StrategyResult(
                edge_route=random_anchor_soft_route,
                source="full_graph_random_anchor_soft_prior" if random_anchor_soft_route else "none",
                search_graph=full_edge_graph,
            ),
        }

        for strategy_name, result in strategy_results.items():
            strategy_stats[strategy_name].add(
                result.edge_route,
                ground_truth_edges,
                ground_truth_route,
                edge_id_to_name,
                edge_id_to_uvk,
                edge_weights,
                edge_lengths,
                result.search_graph,
            )

        samples.append(
            {
                "query_index": i,
                "query_od_pair": [int(start_edge), int(dest_edge)]
                if start_edge is not None and dest_edge is not None
                else None,
                "retrieved_od_pairs": [[int(a), int(b)] for a, b in retrieved_pairs],
                "raw_anchor_segments": raw_anchor_ids,
                "anchor_segments": anchor_segment_ids,
                "random_anchor_segments": random_anchor_ids,
                "anchor_edge_count": len(anchor_edges),
                "retrieved_union_edge_count": len(retrieved_edges),
                "candidate_segment_count": len(candidate_segment_ids),
                "strategy_route_lengths": {
                    strategy_name: len(result.edge_route)
                    for strategy_name, result in strategy_results.items()
                },
            }
        )

    total_samples = len(raw_llm_outputs)
    summaries = {
        strategy_name: stats.summary(total_samples) for strategy_name, stats in strategy_stats.items()
    }
    gt_in_retrieved_union_rate = (
        gt_in_retrieved_union_count / total_samples if total_samples else 0.0
    )

    cprint(f"\nResults for {variables.place_name} ({variables.path_type})", "green", attrs=["bold"])
    print(f"Total Samples Tested: {total_samples}")
    print(f"Valid Anchor JSON: {valid_anchor_json_count}/{total_samples}")
    print(f"Non-empty In-Context Anchors: {non_empty_anchor_count}/{total_samples}")
    print(f"Non-empty Random Anchors: {non_empty_random_anchor_count}/{total_samples}")
    print(f"Known but out-of-context anchors discarded: {invalid_or_out_of_context_anchor_count}")
    print(
        f"GT fully in retrieved union: {gt_in_retrieved_union_count}/{total_samples} "
        f"({gt_in_retrieved_union_rate * 100:.2f}%)"
    )
    print("-" * 30)
    for strategy_name, summary in summaries.items():
        print_strategy_summary(strategy_name, summary, total_samples)

    output_dir = f"evaluation_results/{variables.path_type}/"
    make_dir(output_dir)
    output_path = output_dir + (
        f"{variables.place_name}{variables.context_name_suffix}_{variables.retrieval_type}_top_"
        f"{variables.number_of_docs_to_retrieve}_paper_v2.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "place_name": variables.place_name,
                "path_type": variables.path_type,
                "retrieval_type": variables.retrieval_type,
                "corridor_graph_form": variables.corridor_graph_form,
                "top_k": variables.number_of_docs_to_retrieve,
                "weight_column": weight_column,
                "weight_source": weight_meta,
                "soft_retrieved_discount": retrieved_discount,
                "soft_anchor_discount": anchor_discount,
                "random_anchor_seed": variables.random_anchor_seed,
                "gt_in_retrieved_union_rate": gt_in_retrieved_union_rate,
                "total_samples": total_samples,
                "valid_anchor_json": valid_anchor_json_count,
                "non_empty_in_context_anchors": non_empty_anchor_count,
                "non_empty_random_anchors": non_empty_random_anchor_count,
                "known_but_out_of_context_anchors_discarded": invalid_or_out_of_context_anchor_count,
                "summaries": summaries,
                "samples": samples,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    cprint(f"\nSaved paper_v2 evaluation to {output_path}", "green")


if __name__ == "__main__":
    _self_check()
    evaluate_paper_v2()
