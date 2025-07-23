import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import signal
import pickle
import json
import re
import bm25s
import jieba
import variables
import numpy as np
from tqdm import tqdm
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from termcolor import cprint
from pathlib import Path
from utils import TimeoutException, timeout_handler, make_dir


# Uncomment to download nltk stopwords
# import nltk
# nltk.download("stopwords")

from nltk.corpus import stopwords
from typing import List

cprint("\n\nGENERATING PATHS USING :", "light_yellow", attrs=["bold"])
cprint(f"-DATASET : {variables.place_name}", "green")
cprint(f"-PATH TYPE : {variables.path_type}", "green")
cprint(f"-LLM : {variables.llm}", "green")
cprint(f"-EMBEDDING MODEL : {variables.embedding_model_formatted_name}", "green")
cprint(f"-USE CONTEXT : {variables.use_context}", "green")
cprint(f"-NO DOCUMENTS TO RETRIEVE : {variables.number_of_docs_to_retrieve}", "green")


def generate_query(path_collection: dict[str, list]) -> tuple[str, str]:
    """
        Generates the retriever and the LLM queries where each query is a question asking to generate a type of path from the provided source and destination.

        Args:
        path_collection : A dictionary which holds three types of paths which are the fastest, shortest and the most_used \
        that can be retrieved by the keys "fastest", "shortest" and "most_used" respectively
    
        Returns
        retriever_query : the retriever query,
        llm_query : the LLM query
        """
    if variables.path_type == "most_used":
        ground_truth_path = path_collection["original_path_road_names"]
    else:
        ground_truth_path = path_collection[f"{variables.path_type}_path_road_names"]
    starting_address = ground_truth_path[0]
    destination_address = ground_truth_path[-1]

    # fmt:off

    if variables.place_name in variables.chinese_cities:
        retriever_query = (f"从{starting_address}到{destination_address}的{variables.map_path_type[variables.path_type]}路线是经过哪些路?")
        llm_query = (retriever_query+ "你的回答中只能包含你推荐的路线所经过的路的名字，不要说别的，并用逗号分开路名。")

    elif variables.place_name in variables.other_cities:
        retriever_query = f"What is the {variables.path_type} path from {starting_address} to {destination_address}?"
        llm_query = (retriever_query + "Your answer should ONLY include the names of the roads traversed by this path and the names must be separated by a comma."
        )
        
    # fmt:on

    return retriever_query, llm_query


def get_prompt(
    llm_query: str, retrieved_documents: list[str], use_context: bool
) -> ChatPromptTemplate:
    """
    Given a retriever query, a query for the LLM, retrieved documents based on the retriever query, generate a prompt for the LLM.

    Args:
    llm_query : query generated for the LLM.
    retrieved_documents : documents retrieved based on the retriever query.
    use_context : boolean value that indicates whether or not to provide context to the LLM.

    Returns
    prompt : a ChatPromptTemplate instance which contains all the information provided above.
    """

    if use_context:
        if variables.place_name in variables.chinese_cities:
            PROMPT_TEMPLATE = """
            假设你是一个导航软件，下面是用户曾经走过的历史路线:
            {context}

            ---
            结合以上信息和你本身对{city_name}路网的了解，请回答下面这个问题:
            {question}
            """
        elif variables.place_name in variables.other_cities:
            PROMPT_TEMPLATE = """
            Suppose you are a navigation app like google maps and you have been given the following historical paths:
            {context}
            
            ---
            use that and your knowledge about the road network of {city_name} to {question}
            """

        query_context = "\n\n---\n\n".join(
            [document for document in retrieved_documents]
        )
    else:
        if variables.place_name in variables.chinese_cities:
            query_context = f"假设你是一个导航软件，根据你对{variables.city_name}路网的了解，请回答请回答下面这个问题"
        elif variables.place_name in variables.other_cities:
            query_context = "Suppose you are a navigation app like google maps,"

        PROMPT_TEMPLATE = """
        {context} 
        {question}
        """

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(
        context=query_context, question=llm_query, city_name=variables.city_name
    )

    return prompt


def generate_single_path(prompt):
    timeout_duration = 60
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_duration)
    try:
        generated_path = []
        response_text = model.invoke(prompt)
        generated_path = response_text.split(",")
    except TimeoutException:
        print("Skipped path due to timeout")
    except Exception as e:
        print(f"Error processing path as {e} type appeared")
    finally:
        signal.alarm(0)

    return generated_path


def generate_paths(prompts):
    cprint(f"\nGenerating paths for {variables.place_name}...", "light_cyan")

    timeout_duration = 45
    signal.signal(signal.SIGALRM, timeout_handler)

    llm_generated_paths = []
    for path_index, prompt in enumerate(tqdm(prompts, dynamic_ncols=True)):
        signal.alarm(timeout_duration)
        try:
            response_text = model.invoke(prompt)
            generated_path = response_text.split(",")
            llm_generated_paths.append(generated_path)
        except TimeoutException:
            print(f"Skipped: path {path_index} due to timeout")
            llm_generated_paths.append([])
        except Exception as e:
            print(f"Error processing path {path_index}: {e}")
            llm_generated_paths.append([])
        finally:
            signal.alarm(0)

        if (path_index + 1) % 1000 == 0:
            cprint(
                f"{(path_index + 1)} paths generated, saving checkpoint", "light_yellow"
            )
            if not os.path.isdir(file_path):
                cprint("Directory not found, creating...", "red")
                Path(file_path).mkdir(parents=True, exist_ok=True)
            with open(file_path + file_name, "wb") as f:
                pickle.dump(llm_generated_paths, f)
            cprint("Checkpoint saved", "light_green")

    cprint("Generating paths complete!", "green")

    return llm_generated_paths


def retrieve_docs_from_refined_corpus(
    retriever_query: str, documents: list[str], k: int = 3
) -> list[str]:
    """
    Given a refined corpus obtained from the BM25 search we perform a semantic search using the query and document embeddings.

    Args:
    retriever_query : query
    documents : list of documents
    k : number of documents to retrieve from the refined corpus

    Returns:
    retrieved_documents
    """
    task_description = (
        "Given a user query, retrieve relevant passages that answer the query"
    )
    # task_description = '给定一个问题，检索所有能帮助回答这个问题的相关段落'
    retriever_query = f"Instruct: {task_description}\nQuery: {retriever_query}"
    query_embeddings = representation_model.embed_query(retriever_query)
    documents_embeddings = representation_model.embed_documents(documents)
    query_embeddings = np.array(query_embeddings)
    documents_embeddings = np.array(documents_embeddings)

    scores = documents_embeddings @ query_embeddings
    top_indices = np.argsort(scores)[-k:]

    return [documents[i] for i in reversed(top_indices)]


def tokenize(text: str) -> List[str]:
    """Tokenize a string of Chinese text"""
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z]", "", text)
    tokens = jieba.cut(text)
    return [token for token in tokens if token and token not in stopwords]


def tokenize_corpus(corpus) -> List[List[str]]:
    """Tokenize the whole corpus"""
    return [tokenize(doc) for doc in tqdm(corpus, dynamic_ncols=True)]


if __name__ == "__main__":
    document = f"pdf_docs/{variables.place_name}_paths_sample.pdf"
    chroma_path = (
        f"chroma_db/{variables.embedding_model_formatted_name}/{variables.place_name}/"  # noqa: F405
    )

    json_file_path = f"json_files/{variables.place_name}_paths.json"
    document = json.loads(Path(json_file_path).read_text())
    corpus = [routing["content"] for routing in document]

    stopwords = stopwords.words("chinese")
    stopwords = set(stopwords)

    cprint("\nTokenizing the corpus...", "green")
    corpus_tokens = bm25s.tokenize(corpus, token_pattern="chinese", stopwords="zh")

    cprint("Indexing the corpus...", "green")
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)

    f = open(f"test_data/{variables.place_name}_data", "rb")
    test_data = pickle.load(f)
    f.close()

    queries = np.array(
        [generate_query(path_collection) for path_collection in test_data]
    )
    retriever_queries, llm_queries = queries[:, 0], queries[:, 1]
    retriever_queries_tokens = bm25s.tokenize(
        retriever_queries, token_pattern="chinese", stopwords="zh"
    )

    cprint("\nRetrieving documents from the corpus using BM25...", "yellow")
    number_of_documents_to_retrieve = 9
    bm25_top_results, _ = retriever.retrieve(
        retriever_queries_tokens,
        k=number_of_documents_to_retrieve,
        leave_progress=True,
        n_threads=-1,
    )

    bm25_results_path = f"bm25_results/{variables.place_name}/{variables.path_type}/"
    make_dir(bm25_results_path)
    with open(bm25_results_path + "results", "wb") as f:
        pickle.dump(bm25_top_results, f)
    cprint("BM25 Search results saved.", "green")

    bm25_results_path = f"bm25_results/{variables.place_name}/{variables.path_type}/"
    f = open(bm25_results_path + "results", "rb")
    bm25_top_results = pickle.load(f)
    f.close()

    cprint("\nLoading embeddings...", "yellow")
    representation_model = HuggingFaceEmbeddings(
        model_name=variables.embedding_model,
        model_kwargs=variables.model_kwargs,
        encode_kwargs=variables.encode_kwargs,
        show_progress=False,
    )  # noqa: F405
    cprint(
        "Retrieving documents from the  refined corpus using semantic search...",
        "green",
    )
    bm25_top_results_docs_ids = [
        [doc_id for doc_id in result] for result in bm25_top_results
    ]  # [[id1,..idn], ..., []]
    bm25_top_results_docs = [
        [corpus[id] for id in ids] for ids in bm25_top_results_docs_ids
    ]  # [[str,..., str], ..., [str,..., str]]

    cprint("Generating prompts...", "green")
    top_k = variables.number_of_docs_to_retrieve
    prompts, semantic_search_results = [], []
    for retriever_query, llm_query, bm25_results in tqdm(
        zip(retriever_queries, llm_queries, bm25_top_results_docs),
        dynamic_ncols=True,
        total=len(retriever_queries),
    ):
        retrieved_docs = retrieve_docs_from_refined_corpus(
            retriever_query, bm25_results, k=top_k
        )
        prompt = get_prompt(
            llm_query, retrieved_docs, use_context=variables.use_context
        )
        prompts.append(prompt)
        semantic_search_results.append(retrieved_docs)

    retriever_results_filepath = (
        f"retriever_results/{variables.place_name}/{variables.path_type}/"
    )
    make_dir(retriever_results_filepath)
    with open(retriever_results_filepath + "semantic_search_results", "wb") as f:
        pickle.dump(semantic_search_results, f)
    cprint("Semantic search results saved.", "green")

    prompts_filepath = f"prompts/{variables.place_name}/{variables.path_type}/"
    make_dir(prompts_filepath)
    with open(prompts_filepath + "prompts", "wb") as f:
        pickle.dump(prompts, f)
    cprint("Prompts saved.", "green")

    cprint("Inference..", "green")
    model = OllamaLLM(model="qwen2.5:14b-instruct")
    file_path = f"generated_paths/{variables.place_name}/{variables.llm}/{variables.embedding_model_formatted_name}/{variables.path_type}/"
    if variables.use_context:
        file_name = f"use_context_{variables.use_context}_k_{variables.number_of_docs_to_retrieve}"
    else:
        file_name = f"use_context_{variables.use_context}"

    results = generate_paths(prompts)

    make_dir(file_path)
    with open(file_path + file_name, "wb") as f:
        pickle.dump(results, f)

    cprint(f"File saved at {file_path}{file_name} ", "green")
