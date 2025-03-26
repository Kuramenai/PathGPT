import os
import time
import signal
import pickle
import gc
from torch.cuda import empty_cache
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.prompts import ChatPromptTemplate
from termcolor import cprint
from pathlib import Path

from utils import TimeoutException, timeout_handler, make_dir
from variables import *

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

cprint('\n\nGENERATING PATHS USING :', 'light_yellow', attrs=['bold'])
cprint(f'-DATASET : {place_name}', 'green')
cprint(f'-PATH TYPE : {path_type}', 'green')
cprint(f'-LLM : {llm}', 'green')
cprint(f'-EMBEDDING MODEL : {embedding_model_formatted_name}', 'green')
cprint(f'-USE CONTEXT : {use_context}', 'green')
cprint(f'-NO DOCUMENTS TO RETRIEVE : {number_of_docs_to_retrieve}', 'green')

chroma_path = f"chroma_db/{embedding_model_formatted_name}/{place_name}/"
file_path = f'generated_paths/{place_name}/{llm}/{embedding_model_formatted_name}/{path_type}/'
file_name = f'use_context_{use_context}_k_{number_of_docs_to_retrieve}'

def generate_query(path_collection):
        ground_truth_path = path_collection['original_path_road_names']
        starting_address = ground_truth_path[0]
        destination_address = ground_truth_path[-1]

        if path_type == 'fastest' :
            retriever_query1 =  f"generate the fastest path from {starting_address} to {destination_address}."
            llm_query1 =  retriever_query1 + "Your answer should ONLY include the names of the roads traversed by this path and the names must be separated by a comma."
            retriever_query =  f"请生成一条从{starting_address}到{destination_address}的路线，要求该路线是最快的。"
            llm_query =  retriever_query + "你的回答中只能包含你推荐的路线所经过的路的名字，不要说别的，并用逗号分开路名。"
        elif path_type == 'shortest' :
            retriever_query1 =  f"generate the shortest path from {starting_address} to {destination_address}."
            llm_query1 =  retriever_query1 + "your answer should ONLY include the names of the roads traversed by this path and the names must be separated by a comma."
            retriever_query =  f"请生成一条从{starting_address}到{destination_address}的路线，要求该路线是最短的。"
            llm_query =  retriever_query + "你的回答中只能包含你推荐的路线所经过的路的名字，不要说别的，并用逗号分开路名。"
        elif path_type == 'most_used':
            retriever_query1 =  f"generate the most commonly used path from {starting_address} to {destination_address}."
            llm_query1 =  retriever_query1 + "Your answer should ONLY include the names of the roads traversed by this path and the names must be separated by a comma."
            retriever_query =  f"请生成一条从{starting_address}到{destination_address}的路线，要求该路线是最常用的。"
            llm_query =  retriever_query + "你的回答中只能包含你推荐的路线所经过的路的名字，不要说别的，并用逗号分开路名."


        task_description = 'Given a user query, retrieve relevant passages that answer or provide information to help answer the query'
        retriever_query = f'Instruct: {task_description}\nQuery: {retriever_query}'

        return retriever_query, llm_query

def retrieve_context(
    query, 
    number_of_docs_to_retrieve = number_of_docs_to_retrieve):

    retrieved_documents = vectordb.similarity_search_with_score(
                                    query= query, 
                                    k=number_of_docs_to_retrieve)
    query_context = "\n\n---\n\n".join([doc.page_content for doc, _score in retrieved_documents])
    return query_context

def get_prompt(path_collection, use_context):

    retriever_query, llm_query =  generate_query(path_collection)

    if use_context:
        PROMPT_TEMPLATE1 = """
        Suppose you are a navigation app like google maps and you have been given the following historical paths:
        {context}

        ---
        use that and your knowledge about the road network of {city_name} to {question}
        """
        PROMPT_TEMPLATE = """
        假设你是一个导航软件，下面是用户曾经走过的历史路线:
        {context}

        ---
        结合以上信息和你本身对{city_name}路网的了解，{question}
        """
        query_context = retrieve_context(retriever_query)
    else:
        query_context1 = f'Suppose you are a navigation app like google maps,'
        query_context = f'假设你是一个导航软件，根据你对{city_name}路网的了解，'
        PROMPT_TEMPLATE = """{context} {question}"""

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=query_context, question=llm_query, city_name=city_name)

    return prompt


def generate_prompts(dataset, use_context):
    global embeddings, vectordb
    cprint("\nGenerating prompts...", 'yellow')
    prompts = []
    for path_collection in tqdm(dataset, dynamic_ncols=True):
        prompt = get_prompt(path_collection, use_context)
        prompts.append(prompt)
    cprint("Generating prompts completed!", 'light_green')
    del embeddings
    del vectordb
    empty_cache()
    return prompts

def generate_paths(prompts):

    cprint(f"\nGenerating paths for {place_name}...", 'light_cyan')

    timeout_duration = 10
    signal.signal(signal.SIGALRM, timeout_handler)

    llm_generated_paths = []

    for path_index, prompt in enumerate(tqdm(prompts, dynamic_ncols=True)):
        signal.alarm(timeout_duration)
        try:

            response_text = model.invoke(prompt)
            generated_path = response_text.split(',')
            llm_generated_paths.append(generated_path)

        except TimeoutException:
            print(f"Skipped: path {path_index} due to timeout")
            llm_generated_paths.append(['N/A'])
        except Exception as e:
            print(f"Error processing path {path_index}: {e}")
            llm_generated_paths.append(['N/A'])
        finally:
            signal.alarm(0)

        if (path_index + 1) % 1000 == 0:
            cprint(f"{(path_index + 1)} paths generated, saving checkpoint", 'light_yellow')
            if not os.path.isdir(file_path):
                cprint("Directory not found, creating...", 'red')
                Path(file_path).mkdir(parents=True, exist_ok=True)
            with open(file_path + file_name, 'wb') as f:
                pickle.dump(llm_generated_paths, f) 
            cprint("Checkpoint saved", 'light_green')

    cprint("Generating paths complete!", 'green')

    return llm_generated_paths


if __name__ == "__main__" :
    

    f = open(f'test_data/{place_name}_data', 'rb')
    test_data = pickle.load(f)
    f.close()
    
    cprint("\nLoading embeddings...", 'yellow')
    embeddings =  HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            show_progress = False)

    vectordb = Chroma(embedding_function=embeddings,persist_directory=chroma_path)
    cprint("Loading embeddings complete!", 'light_green')

    model = OllamaLLM(model='qwen2.5:14b-instruct')
 
    prompts_path = f'prompts/{place_name}/'
    make_dir(prompts_path)
    if not os.path.isfile(prompts_path + f'{path_type}_paths_generation_prompts'):
        prompts = generate_prompts(test_data, use_context)
        with open(prompts_path + f'{path_type}_paths_generation_prompts', 'wb') as f:
            pickle.dump(prompts, f)
        print("Prompts saved at %s " % prompts_path + f'{path_type}_paths_generation_prompts')
    else :
        print("Folder %s already exists, loading prompts..." % prompts_path)
        f = open(prompts_path + f'{path_type}_paths_generation_prompts','rb')
        prompts = pickle.load(f)
        f.close()


    results = generate_paths(prompts)

    make_dir(file_path)
    with open(file_path + file_name, 'wb') as f:
        pickle.dump(results, f)
    
    cprint(f"File saved at {file_path}{file_name} ", 'green')
       



