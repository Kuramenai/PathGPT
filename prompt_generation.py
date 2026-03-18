import os
import time
import json
import pickle
import re
import numpy as np
import jieba
import bm25s
import variables
from pathlib import Path
from tqdm import tqdm
from termcolor import cprint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from utils import make_dir


# Set environment variables
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def welcome_text():
    cprint("\n\nGENERATING PATHS USING :", "light_yellow", attrs=["bold"])
    cprint(f"-DATASET : {variables.place_name}", "green")
    cprint(f"-PATH TYPE : {variables.path_type}", "green")
    cprint(f"-LLM : {variables.llm}", "green")
    cprint(f"-EMBEDDING MODEL : {variables.embedding_model_formatted_name}", "green")
    cprint(f"-USE CONTEXT : {variables.use_context}", "green")
    cprint(f"-NO DOCUMENTS TO RETRIEVE : {variables.number_of_docs_to_retrieve}", "green")

def tokenize_chinese(text: str) -> list[str]:
    """Reproducible custom tokenization for BM25"""
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', '', text)
    return list(jieba.cut_for_search(text))

def generate_query(path_collection: dict) -> tuple[str, dict]:
    """
    Generates both the semantic retriever query and the LLM prompt dictionary.
    """
    ground_truth_path = path_collection[f"{variables.path_type}_path_edges_names"]
    origin_name = ground_truth_path[0]
    dest_name = ground_truth_path[-1]
    
    task_translation = {
        "fuel_efficient": "最省油",
        "scenic": "风景最好",
        "fastest": "最快",
        "shortest": "最短"
    }
    chinese_task = task_translation.get(variables.path_type, "最优")
    
    # E5 models require the "query: " prefix for asymmetric search
    retriever_query = f"query: 寻找一条从 {origin_name} 到 {dest_name} 的{chinese_task}路线。"
    
    # Bundle the question and the strict JSON rules together
    question_text = (
        f"请结合上述路网拓扑，寻找一条从 {origin_name} 到 {dest_name} 的{chinese_task}路线。\n\n"
        f"请严格遵守以下规则：\n"
        f"1. 如果提供的上下文中包含相关的道路，请优先使用它们。\n"
        f"2. 如果上下文中没有足够的道路信息，请利用您自身对{variables.place_name}路网的了解进行补全。\n"
        f"3. 不要包含任何推理、解释或元评论（不需要思考过程）。\n"
        f"4. 必须输出一个严格的 JSON 格式，其中包含一个按行驶顺序排列的道路名称列表（从起点到终点）。\n"
        f"5. 【重要】如果路网拓扑中的道路名称是以连字符合并的（例如“A路 - B路”），请在 JSON 数组中将它们拆分为独立的道路字符串（例如 [\"A路\", \"B路\"]）。\n"
        f"6. 如果找不到任何可行的路线，请返回一个空列表。\n\n"
        f"输出格式（严格 JSON）：\n"
        f"{{\n  \"route\": [\"道路1\", \"道路2\", \"道路3\"]\n}}"
    )
    
    llm_query_dict = {
        "system_instruction": f"您是一位具备丰富本地地理知识的{variables.place_name}道路导航助手。您的任务是生成起点和终点之间最符合现实的驾驶路线。",
        "question": question_text
    }
    return retriever_query, llm_query_dict

def get_prompt(
    llm_query_dict: dict, 
    retrieved_markdown_docs: list[str], 
    use_context: bool
) -> str:
    """
    Formats the final ChatPromptTemplate using the retrieved Markdown subgraphs.
    """
    if use_context:
        query_context = "\n\n---\n\n".join(retrieved_markdown_docs)
        
        PROMPT_TEMPLATE = """
        {system_instruction}
        {context}

        ---
        {question}
        """
    else:
        query_context = "您是一位专业的路线规划和导航助手。您的任务是根据用户的特定需求，在给定的局部路网中寻找最佳路径。"
        PROMPT_TEMPLATE = """
        {context} 

        {question}
        """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=query_context, 
        question=llm_query_dict["question"], 
        system_instruction=llm_query_dict.get("system_instruction", "")
    )
    return prompt

if __name__ == "__main__":
    welcome_text()
    
    # 1. LOAD DATA (Fixed JSON Loading)
    json_file_path = f"json_files/{variables.path_type}/{variables.place_name}_markdown_prompts.json"
    with open(json_file_path, "r", encoding="utf-8") as f:
        context_dataset = json.load(f)
        
    # Implement Dual Representation!
    embedding_corpus = [item["embedding_text"] for item in context_dataset]
    markdown_corpus = [item["llm_prompt"] for item in context_dataset]

    # 2. INDEXING BM25 (Offline Phase)
    cprint("\nTokenizing and Indexing the corpus for BM25...", "green")
    start_time = time.time()
    
    # Use our reproducible tokenizer
    corpus_tokens = [tokenize_chinese(doc) for doc in embedding_corpus]
    
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    cprint(f"BM25 Indexing took {time.time() - start_time:.4f} seconds", "cyan")

    # 3. PRE-COMPUTING EMBEDDINGS (Offline Phase)
    # This completely removes the latency bottleneck from the inner loop!
    cprint("\nPre-computing E5 embeddings for the entire corpus...", "yellow")
    start_time = time.time()
    
    representation_model = HuggingFaceEmbeddings(
        model_name=variables.embedding_model,
        model_kwargs=variables.model_kwargs,
        encode_kwargs=variables.encode_kwargs,
        show_progress=True,
    )
    
    # Embed the dense text representation, NOT the markdown
    corpus_embeddings = np.array(representation_model.embed_documents(embedding_corpus))
    
    cprint(f"Embedding generation took {time.time() - start_time:.4f} seconds", "cyan")

    # 4. LOAD TEST DATA
    test_data_filename = f"filtered_test_data/{variables.path_type}/{variables.place_name}_data"
    with open(test_data_filename, 'rb') as f:
        test_data = pickle.load(f)

    # Extract queries safely
    retriever_queries = []
    llm_query_dicts = []
    for path_collection in test_data:
        r_query, l_query = generate_query(path_collection)
        retriever_queries.append(r_query)
        llm_query_dicts.append(l_query)

    # 5. RETRIEVAL & PROMPT GENERATION (Online Phase)
    cprint("\nExecuting Hybrid Retrieval (BM25 + Semantic Re-ranking)...", "green")
    
    number_of_documents_to_retrieve = 9
    top_k_final = variables.number_of_docs_to_retrieve
    
    prompts = []
    semantic_search_results = []
    
    for i in tqdm(range(len(retriever_queries)), dynamic_ncols=True, desc="Processing Queries"):
        r_query = retriever_queries[i]
        l_query_dict = llm_query_dicts[i]
        
        # Step A: BM25 Lexical Filtering
        query_tokens = tokenize_chinese(r_query)
        bm25_results, _ = retriever.retrieve([query_tokens], k=number_of_documents_to_retrieve)
        candidate_indices = bm25_results[0] # Get the indices of the top 9 hits
        
        # Step B: Semantic Re-ranking (Lightning fast because we pre-embedded!)
        query_embedding = np.array(representation_model.embed_query(r_query))
        candidate_embeddings = corpus_embeddings[candidate_indices]
        
        # Dot product for similarity
        scores = candidate_embeddings @ query_embedding
        
        # Sort and get the top_k_final indices relative to the candidate list
        best_candidate_positions = np.argsort(scores)[-top_k_final:][::-1]
        
        # Map back to the original corpus indices
        final_indices = [candidate_indices[pos] for pos in best_candidate_positions]
        
        # Step C: Generate Prompt using the MARKDOWN representations
        retrieved_markdowns = [markdown_corpus[idx] for idx in final_indices]
        semantic_search_results.append(retrieved_markdowns)
        
        prompt = get_prompt(l_query_dict, retrieved_markdowns, variables.use_context)
        prompts.append(prompt)
    
    if variables.use_context:
        prompts_filepath = f"prompts/{variables.path_type}/with_context/"
    else:
        prompts_filepath = f"prompts/{variables.path_type}/no_context/"
        
    make_dir(prompts_filepath)
    with open(prompts_filepath + f"{variables.place_name}_prompts_top_{number_of_documents_to_retrieve}", "wb") as f:
        pickle.dump(prompts, f)
    cprint("Prompts saved.", "green")
    cprint(f"\nSuccessfully generated {len(prompts)} prompts ready for Qwen3-8B!", "light_green")