import os
import time
import json
import pickle
import re
import numpy as np
import jieba
import bm25s
import geopandas as gpd
import variables
from pathlib import Path
from typing import Any
from tqdm import tqdm
from termcolor import cprint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from utils import make_dir


# Set environment variables
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def welcome_text():
    cprint("\n\nGENERATING PROMPTS FOR :", "light_yellow", attrs=["bold"])
    cprint(f"-DATASET : {variables.place_name}", "green")
    cprint(f"-PATH TYPE : {variables.path_type}", "green")
    cprint(f"-USE CONTEXT : {variables.use_context}", "green")
    cprint(f"-LLM TASK : {variables.llm_task}", "green")
    if variables.use_context:
        cprint(f"-CORRIDOR GRAPH FORM : {variables.corridor_graph_form}", "green")
    cprint(f"-NUMBER OF DOCUMENTS TO RETRIEVE : {variables.number_of_docs_to_retrieve}", "green")
    cprint(f"-RETRIEVAL : {variables.retrieval_type}", "green")


def parse_od_pair(raw_od_pair: Any) -> tuple[int, int] | None:
    if not isinstance(raw_od_pair, (list, tuple)) or len(raw_od_pair) < 2:
        return None
    try:
        return int(raw_od_pair[0]), int(raw_od_pair[1])
    except (TypeError, ValueError):
        return None


def jsonable_od_pair(od_pair: tuple[int, int] | None) -> list[int] | None:
    return [int(od_pair[0]), int(od_pair[1])] if od_pair is not None else None


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    if scores.size == 0:
        return scores

    score_min = float(np.min(scores))
    score_max = float(np.max(scores))
    if np.isclose(score_max, score_min):
        return np.ones_like(scores) if score_max > 0 else np.zeros_like(scores)
    return (scores - score_min) / (score_max - score_min)


def midpoint_for_geometry(geometry: Any) -> tuple[float, float] | None:
    if geometry is None or geometry.is_empty:
        return None

    try:
        point = geometry.interpolate(0.5, normalized=True)
    except (TypeError, ValueError):
        point = geometry.centroid

    return float(point.x), float(point.y)


def build_edge_midpoints(edge_data_path: str) -> dict[int, tuple[float, float]]:
    edges_df = gpd.read_file(edge_data_path)

    if edges_df.crs is not None:
        try:
            metric_crs = edges_df.estimate_utm_crs()
        except RuntimeError:
            metric_crs = None
        edges_df = edges_df.to_crs(metric_crs or "EPSG:3857")

    edge_midpoints = {}
    for edge_id, row in edges_df.iterrows():
        midpoint = midpoint_for_geometry(row.geometry)
        if midpoint is not None:
            edge_midpoints[int(edge_id)] = midpoint
    return edge_midpoints


def build_corpus_spatial_arrays(
    context_dataset: list[dict],
    edge_midpoints: dict[int, tuple[float, float]],
) -> tuple[list[tuple[int, int] | None], np.ndarray, np.ndarray]:
    context_od_pairs = [parse_od_pair(item.get("od_pair")) for item in context_dataset]
    origin_points = np.full((len(context_dataset), 2), np.nan, dtype=float)
    destination_points = np.full((len(context_dataset), 2), np.nan, dtype=float)

    for idx, od_pair in enumerate(context_od_pairs):
        if od_pair is None:
            continue

        origin_point = edge_midpoints.get(od_pair[0])
        destination_point = edge_midpoints.get(od_pair[1])
        if origin_point is not None:
            origin_points[idx] = origin_point
        if destination_point is not None:
            destination_points[idx] = destination_point

    return context_od_pairs, origin_points, destination_points


def compute_spatial_scores(
    query_od_pair: tuple[int, int] | None,
    edge_midpoints: dict[int, tuple[float, float]] | None,
    corpus_origin_points: np.ndarray | None,
    corpus_destination_points: np.ndarray | None,
) -> np.ndarray:
    corpus_size = 0 if corpus_origin_points is None else len(corpus_origin_points)
    scores = np.zeros(corpus_size, dtype=float)

    if (
        query_od_pair is None
        or edge_midpoints is None
        or corpus_origin_points is None
        or corpus_destination_points is None
    ):
        return scores

    query_origin = edge_midpoints.get(query_od_pair[0])
    query_destination = edge_midpoints.get(query_od_pair[1])
    if query_origin is None or query_destination is None:
        return scores

    origin_distances = np.linalg.norm(corpus_origin_points - np.asarray(query_origin), axis=1)
    destination_distances = np.linalg.norm(corpus_destination_points - np.asarray(query_destination), axis=1)
    od_distances = origin_distances + destination_distances
    valid = np.isfinite(od_distances)
    if not np.any(valid):
        return scores

    valid_distances = od_distances[valid]
    scale = float(np.percentile(valid_distances, 95))
    if scale <= 0:
        scores[valid] = 1.0
    else:
        scores[valid] = np.clip(1.0 - (valid_distances / scale), 0.0, 1.0)
    return scores


def retrieve_doc_indices(
    query: str,
    query_tokens: list[str],
    retriever: bm25s.BM25 | None,
    representation_model: HuggingFaceEmbeddings | None,
    corpus_embeddings: np.ndarray | None,
    top_k: int,
    retrieval_type: str,
    query_od_pair: tuple[int, int] | None = None,
    edge_midpoints: dict[int, tuple[float, float]] | None = None,
    corpus_origin_points: np.ndarray | None = None,
    corpus_destination_points: np.ndarray | None = None,
    first_stage_k: int = 100,
    spatial_weight: float = 0.5,
    bm25_weight: float = 0.5,
    return_metadata: bool = False,
) -> list[int] | tuple[list[int], dict]:
    metadata = {
        "retrieval_type": retrieval_type,
        "query_od_pair": jsonable_od_pair(query_od_pair),
    }

    if retrieval_type == "bm25":
        bm25_results, bm25_scores = retriever.retrieve([query_tokens], k=top_k)
        final_indices = [int(idx) for idx in bm25_results[0]]
        metadata["final_indices"] = final_indices
        metadata["final_bm25_scores"] = [float(score) for score in bm25_scores[0]]
        return (final_indices, metadata) if return_metadata else final_indices

    if retrieval_type == "semantic":
        query_embedding = np.array(representation_model.embed_query(query))
        scores = corpus_embeddings @ query_embedding
        final_indices = [int(idx) for idx in np.argsort(scores)[-top_k:][::-1]]
        metadata["final_indices"] = final_indices
        metadata["final_semantic_scores"] = [float(scores[idx]) for idx in final_indices]
        return (final_indices, metadata) if return_metadata else final_indices

    if retrieval_type == "spatial_hybrid":
        corpus_size = len(corpus_embeddings)
        first_stage_k = min(max(first_stage_k, top_k), corpus_size)

        spatial_scores = compute_spatial_scores(
            query_od_pair,
            edge_midpoints,
            corpus_origin_points,
            corpus_destination_points,
        )
        spatial_candidate_indices = np.argsort(spatial_scores)[-first_stage_k:][::-1]

        bm25_results, bm25_scores = retriever.retrieve([query_tokens], k=first_stage_k)
        bm25_candidate_indices = np.asarray(bm25_results[0], dtype=int)
        bm25_candidate_scores = np.asarray(bm25_scores[0], dtype=float)
        bm25_score_by_index = {
            int(idx): float(score) for idx, score in zip(bm25_candidate_indices, bm25_candidate_scores)
        }

        candidate_indices = sorted(
            set(int(idx) for idx in spatial_candidate_indices)
            | set(int(idx) for idx in bm25_candidate_indices)
        )
        candidate_spatial_scores = np.asarray(
            [float(spatial_scores[idx]) for idx in candidate_indices], dtype=float
        )
        candidate_bm25_scores = np.asarray(
            [bm25_score_by_index.get(int(idx), 0.0) for idx in candidate_indices], dtype=float
        )

        first_stage_scores = spatial_weight * normalize_scores(
            candidate_spatial_scores
        ) + bm25_weight * normalize_scores(candidate_bm25_scores)
        first_stage_positions = np.argsort(first_stage_scores)[::-1][:first_stage_k]
        first_stage_indices = [int(candidate_indices[pos]) for pos in first_stage_positions]

        query_embedding = np.array(representation_model.embed_query(query))
        semantic_scores = corpus_embeddings[first_stage_indices] @ query_embedding
        best_candidate_positions = np.argsort(semantic_scores)[-top_k:][::-1]
        final_indices = [first_stage_indices[pos] for pos in best_candidate_positions]

        metadata.update(
            {
                "spatial_weight": float(spatial_weight),
                "bm25_weight": float(bm25_weight),
                "first_stage_k": int(first_stage_k),
                "spatial_candidate_indices": [int(idx) for idx in spatial_candidate_indices],
                "bm25_candidate_indices": [int(idx) for idx in bm25_candidate_indices],
                "first_stage_indices": first_stage_indices,
                "final_indices": final_indices,
                "final_semantic_scores": [float(semantic_scores[pos]) for pos in best_candidate_positions],
            }
        )
        return (final_indices, metadata) if return_metadata else final_indices

    bm25_candidate_count = max(9, top_k)
    bm25_results, bm25_scores = retriever.retrieve([query_tokens], k=bm25_candidate_count)
    candidate_indices = bm25_results[0]
    query_embedding = np.array(representation_model.embed_query(query))
    candidate_embeddings = corpus_embeddings[candidate_indices]
    scores = candidate_embeddings @ query_embedding
    best_candidate_positions = np.argsort(scores)[-top_k:][::-1]
    final_indices = [int(candidate_indices[pos]) for pos in best_candidate_positions]
    metadata.update(
        {
            "bm25_candidate_indices": [int(idx) for idx in candidate_indices],
            "bm25_candidate_scores": [float(score) for score in bm25_scores[0]],
            "final_indices": final_indices,
            "final_semantic_scores": [float(scores[pos]) for pos in best_candidate_positions],
        }
    )
    return (final_indices, metadata) if return_metadata else final_indices


def prompts_output_name(top_k: int) -> str:
    task_suffix = (
        f"_{variables.llm_task}" if variables.use_context and variables.llm_task != "route_segments" else ""
    )
    context_suffix = variables.context_name_suffix if variables.use_context else ""
    return f"{variables.place_name}{context_suffix}_prompts_{variables.retrieval_type}_top_{top_k}{task_suffix}"


def tokenize_chinese(text: str) -> list[str]:
    """Reproducible custom tokenization for BM25"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", text)
    return list(jieba.cut_for_search(text))


def get_query_od_pair(path_collection: dict) -> tuple[int, int] | None:
    edge_path = path_collection.get(f"{variables.path_type}_path_edges", [])
    if not edge_path:
        edge_path = path_collection.get("historical_path_edges", [])
    if not edge_path:
        return None
    return int(edge_path[0]), int(edge_path[-1])


def generate_query(path_collection: dict) -> tuple[str, dict, tuple[int, int] | None]:
    """
    Generates both the semantic retriever query and the LLM prompt dictionary.
    """
    ground_truth_path = path_collection[f"{variables.path_type}_path_edges_names"]
    origin_name = ground_truth_path[0]
    dest_name = ground_truth_path[-1]

    task_translation = {
        "fuel_efficient": "最省油",
        "highway_free": "避开高速",
        "poi_aware": "经过景点最多的路线",
        "most_used": "最常用",
        "scenic": "风景最好",
        "fastest": "最快",
        "shortest": "最短",
    }
    chinese_task = task_translation.get(variables.path_type, "最优")

    # E5 models require the "query: " prefix for asymmetric search
    retriever_query = f"query: 寻找一条从 {origin_name} 到 {dest_name} 的{chinese_task}路线。"

    if variables.use_context and variables.llm_task == "rank_contexts":
        question_text = (
            f"请根据上述候选路网，对从 {origin_name} 到 {dest_name} 的{chinese_task}路线所需的候选路网进行排序。\n\n"
            f"请严格遵守以下规则：\n"
            f"1. 只输出候选路网编号，例如 1、2、3；编号来自每个候选路网标题。\n"
            f"2. 最适合作为后续图搜索走廊的候选路网排在最前面。\n"
            f"3. 排序时综合考虑起终点空间相关性、路线需求、道路属性、景点信息（如适用）和拓扑可用性。\n"
            f"4. 不要输出道路名称、G 段编号、完整路径、解释、推理过程或元评论。\n"
            f"5. 如果没有任何候选路网有用，请返回空列表。\n\n"
            f"输出格式（严格 JSON）：\n"
            f'{{\n  "ranked_contexts": [1, 2, 3]\n}}'
        )
    elif variables.use_context and variables.llm_task == "anchor_segments":
        question_text = (
            f"根据上述候选道路网络上下文，选择对于从 {origin_name} 到 {dest_name} 的{chinese_task}路线最有帮助的符号化 G 路段，作为软语义锚点。\n\n"
            f"请严格遵守以下规则：\n"
            f"1. 仅输出候选上下文中出现过的符号化路段 ID，例如 G12。\n"
            f"2. 优先选择道路属性、POI 信息以及局部拓扑结构最符合路线需求的路段。\n"
            f"3. 不要输出道路名称、原始边 ID、完整路径、解释说明或推理过程。\n"
            f"4. 这些锚点仅作为后续全图搜索的软提示，因此只选择最相关的路段。\n"
            f"5. 如果没有任何有用的路段，返回空列表。\n\n"
            f"输出格式（严格 JSON）：\n"
            f'{{\n  "anchor_segments": ["G12", "G18", "G31"]\n}}'
        )
    elif variables.use_context:
        question_text = (
            f"请结合上述符号化路网拓扑，寻找一条从 {origin_name} 到 {dest_name} 的{chinese_task}路线。\n\n"
            f"请严格遵守以下规则：\n"
            f"1. 每个符号编号（例如 G12）表示一个连续道路段，必须直接从上下文中选择。\n"
            f"2. 可以综合使用多个候选路网中的符号，但相邻两个符号必须在至少一个候选路网中满足“可连接到”关系。\n"
            f"3. 不要输出道路名称、解释、推理过程或元评论。\n"
            f"4. 如果找不到任何可行的符号路线，请返回空列表。\n\n"
            f"输出格式（严格 JSON）：\n"
            f'{{\n  "route_segments": ["G12", "G18", "G31"]\n}}'
        )
    else:
        question_text = (
            f"请寻找一条从 {origin_name} 到 {dest_name} 的{chinese_task}路线。\n\n"
            f"请严格遵守以下规则：\n"
            f"1. 不要包含任何推理、解释或元评论（不需要思考过程）。\n"
            f"2. 必须输出一个严格的 JSON 格式，其中包含一个按行驶顺序排列的道路名称列表（从起点到终点）。\n"
            f"3. 如果找不到任何可行的路线，请返回一个空列表。\n\n"
            f"输出格式（严格 JSON）：\n"
            f'{{\n  "route": ["道路1", "道路2", "道路3"]\n}}'
        )

    if variables.use_context and variables.llm_task == "rank_contexts":
        system_instruction = (
            f"您是一位{variables.place_name}候选路网排序助手。"
            f"您的任务是根据用户路线需求，选择最值得交给下游图搜索算法的候选路网。"
        )
    elif variables.use_context:
        system_instruction = (
            f"您是一位{variables.place_name}符号化路网导航助手。"
            f"您的任务是根据给定候选路网生成拓扑连续的全局路段符号序列。"
        )
    else:
        system_instruction = (
            f"您是一位具备丰富本地地理知识的{variables.place_name}道路导航助手。"
            f"您的任务是生成起点和终点之间最符合现实的驾驶路线。"
        )

    if variables.use_context and variables.llm_task == "anchor_segments":
        system_instruction = (
            f"你是一名 {variables.place_name} 的符号化道路路段选择助手。"
            f"你的任务是选择一组紧凑的 G 路段锚点（G-segment anchors），用于为后续的全图路径搜索提供软引导。"
        )

    llm_query_dict = {"system_instruction": system_instruction, "question": question_text}
    return retriever_query, llm_query_dict, get_query_od_pair(path_collection)


def get_prompt(llm_query_dict: dict, retrieved_markdown_docs: list[str], use_context: bool) -> str:
    """
    Formats the final ChatPromptTemplate using the retrieved Markdown subgraphs.
    """
    if use_context:
        query_context = "\n\n---\n\n".join(
            f"## 候选路网 {idx}\n{doc}" for idx, doc in enumerate(retrieved_markdown_docs, start=1)
        )

        PROMPT_TEMPLATE = """
        {system_instruction}
        {context}

        ---
        {question}
        """
    else:
        query_context = (
            "您是一位专业的路线规划和导航助手。您的任务是根据用户的特定需求，在给定的局部路网中寻找最佳路径。"
        )
        PROMPT_TEMPLATE = """
        {context} 

        {question}
        """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=query_context,
        question=llm_query_dict["question"],
        system_instruction=llm_query_dict.get("system_instruction", ""),
    )
    return prompt


if __name__ == "__main__":
    welcome_text()

    if not variables.use_context:
        test_data_filename = f"filtered_test_data/{variables.path_type}/{variables.place_name}_data"
        with open(test_data_filename, "rb") as f:
            test_data = pickle.load(f)

        if variables.place_name == "chengdu":
            test_data = test_data[:1_500]

        prompts = [
            get_prompt(generate_query(path_collection)[1], [], use_context=False)
            for path_collection in tqdm(test_data, dynamic_ncols=True, desc="Generating no-context prompts")
        ]

        prompts_filepath = f"prompts/{variables.path_type}/no_context/"
        make_dir(prompts_filepath)
        output_name = prompts_output_name(variables.number_of_docs_to_retrieve)
        with open(prompts_filepath + output_name, "wb") as f:
            pickle.dump(prompts, f)
        cprint(f"Prompts saved to {prompts_filepath}{output_name}", "green")
        cprint(f"\nSuccessfully generated {len(prompts)} no-context prompts ready for Qwen3-8B!", "light_green")
        exit(0)

    # 1. LOAD DATA (Fixed JSON Loading)
    json_file_path = (
        f"json_files/{variables.path_type}/"
        f"{variables.place_name}{variables.context_name_suffix}_markdown_prompts.json"
    )
    with open(json_file_path, "r", encoding="utf-8") as f:
        context_dataset = json.load(f)

    # Implement Dual Representation!
    embedding_corpus = [item["embedding_text"] for item in context_dataset]
    markdown_corpus = [item["llm_prompt"] for item in context_dataset]
    context_od_pairs = [parse_od_pair(item.get("od_pair")) for item in context_dataset]

    retrieval_type = variables.retrieval_type
    need_bm25 = retrieval_type in ("bm25", "hybrid", "spatial_hybrid")
    need_semantic = retrieval_type in ("semantic", "hybrid", "spatial_hybrid")
    need_spatial = retrieval_type == "spatial_hybrid"

    retriever = None
    if need_bm25:
        cprint("\nTokenizing and indexing the corpus for BM25...", "green")
        start_time = time.time()
        corpus_tokens = [tokenize_chinese(doc) for doc in embedding_corpus]
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        cprint(f"BM25 indexing took {time.time() - start_time:.4f} seconds", "cyan")

    representation_model = None
    corpus_embeddings = None
    if need_semantic:
        cprint("\nPre-computing E5 embeddings for the entire corpus...", "yellow")
        start_time = time.time()
        representation_model = HuggingFaceEmbeddings(
            model_name=variables.embedding_model,
            model_kwargs=variables.model_kwargs,
            encode_kwargs=variables.encode_kwargs,
            show_progress=False,
        )
        corpus_embeddings = np.array(representation_model.embed_documents(embedding_corpus))
        cprint(f"Embedding generation took {time.time() - start_time:.4f} seconds", "cyan")

    edge_midpoints = None
    corpus_origin_points = None
    corpus_destination_points = None
    if need_spatial:
        cprint("\nBuilding spatial retrieval index from edge geometries...", "yellow")
        start_time = time.time()
        edge_midpoints = build_edge_midpoints(variables.EDGE_DATA)
        context_od_pairs, corpus_origin_points, corpus_destination_points = build_corpus_spatial_arrays(
            context_dataset,
            edge_midpoints,
        )
        cprint(f"Spatial index built in {time.time() - start_time:.4f} seconds", "cyan")

    # 4. LOAD TEST DATA
    test_data_filename = f"filtered_test_data/{variables.path_type}/{variables.place_name}_data"
    with open(test_data_filename, "rb") as f:
        test_data = pickle.load(f)

    if variables.place_name == "chengdu":
        test_data = test_data[:1_500]

    # Extract queries safely
    retriever_queries = []
    llm_query_dicts = []
    query_od_pairs = []
    for path_collection in test_data:
        r_query, l_query, query_od_pair = generate_query(path_collection)
        retriever_queries.append(r_query)
        llm_query_dicts.append(l_query)
        query_od_pairs.append(query_od_pair)

    # 5. RETRIEVAL & PROMPT GENERATION (Online Phase)
    cprint(f"\nExecuting {retrieval_type} retrieval...", "green")

    top_k_final = variables.number_of_docs_to_retrieve
    prompts = []
    retrieval_metadata = []

    for i in tqdm(range(len(retriever_queries)), dynamic_ncols=True, desc="Processing Queries"):
        r_query = retriever_queries[i]
        l_query_dict = llm_query_dicts[i]
        query_od_pair = query_od_pairs[i]
        query_tokens = tokenize_chinese(r_query)
        final_indices, query_metadata = retrieve_doc_indices(
            r_query,
            query_tokens,
            retriever,
            representation_model,
            corpus_embeddings,
            top_k_final,
            retrieval_type,
            query_od_pair=query_od_pair,
            edge_midpoints=edge_midpoints,
            corpus_origin_points=corpus_origin_points,
            corpus_destination_points=corpus_destination_points,
            first_stage_k=variables.spatial_candidate_k,
            spatial_weight=variables.spatial_weight,
            bm25_weight=variables.bm25_weight,
            return_metadata=True,
        )
        retrieved_markdowns = [markdown_corpus[idx] for idx in final_indices]
        prompts.append(get_prompt(l_query_dict, retrieved_markdowns, variables.use_context))
        query_metadata.update(
            {
                "query_index": int(i),
                "retrieved_indices": [int(idx) for idx in final_indices],
                "retrieved_od_pairs": [jsonable_od_pair(context_od_pairs[idx]) for idx in final_indices],
            }
        )
        retrieval_metadata.append(query_metadata)

    if variables.use_context:
        prompts_filepath = f"prompts/{variables.path_type}/with_context/"
    else:
        prompts_filepath = f"prompts/{variables.path_type}/no_context/"

    make_dir(prompts_filepath)
    output_name = prompts_output_name(top_k_final)
    with open(prompts_filepath + output_name, "wb") as f:
        pickle.dump(prompts, f)
    cprint(f"Prompts saved to {prompts_filepath}{output_name}", "green")

    metadata_filepath = f"retrieval_metadata/{variables.path_type}/"
    make_dir(metadata_filepath)
    with open(metadata_filepath + output_name + ".json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "place_name": variables.place_name,
                "path_type": variables.path_type,
                "retrieval_type": variables.retrieval_type,
                "corridor_graph_form": variables.corridor_graph_form,
                "top_k": variables.number_of_docs_to_retrieve,
                "spatial_candidate_k": variables.spatial_candidate_k,
                "spatial_weight": variables.spatial_weight,
                "bm25_weight": variables.bm25_weight,
                "queries": retrieval_metadata,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    cprint(f"Retrieval metadata saved to {metadata_filepath}{output_name}.json", "green")
    cprint(f"\nSuccessfully generated {len(prompts)} prompts ready for Qwen3-8B!", "light_green")
