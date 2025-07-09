import os
import pickle
import os.path
from tqdm import tqdm
from argparse import ArgumentParser
from termcolor import cprint
from datetime import datetime
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers.util import semantic_search, cos_sim

from utils import get_args, make_dir
from variables import *


cprint(
    f"EVALUATING PERFORMANCE ON {place_name} DATASET WITH:",
    "light_yellow",
    attrs=["bold"],
)
cprint(f"-LLM : {llm}", "green")
cprint(f"-PATH TYPE : {path_type}", "green")
cprint(f"-EMBEDDING MODEL : {embedding_model_formatted_name}", "green")
cprint(f"-USE CONTEXT : {use_context}", "green")
cprint(f"-RETRIEVED DOCUMENTS : {number_of_docs_to_retrieve}", "green")

place_name = "beijing"
script_dir = os.path.dirname(os.path.abspath(__file__))
road_names_file = f"road_names/{place_name}_road_names"
road_names_file = os.path.join(script_dir, road_names_file)
if os.path.exists(road_names_file):
    f = open(road_names_file, "rb")
    road_names = pickle.load(f)
    f.close
road_names = list(road_names.values())

generated_data_path = (
    f"generated_paths/{place_name}/{llm}/{embedding_model_formatted_name}/{path_type}"
)
generated_data_filename = f"/use_context_{use_context}_k_3"

curr_dir = os.getcwd()
file_path = os.path.join(
    curr_dir,
    "generated_paths",
    f"{place_name}",
    f"{llm}",
    f"{embedding_model_formatted_name}",
    f"{path_type}",
    f"use_context_{use_context}_k_3",
)

f = open(generated_data_path + generated_data_filename, "rb")
f = open(file_path, "rb")
generated_data = pickle.load(f)
f.close()

test_data_path = f"test_data/"
test_data_filename = f"{place_name}_data"

f = open(test_data_path + test_data_filename, "rb")
test_data = pickle.load(f)
f.close()

in_database_dissimilarities = 0
out_of_database_dissimilarities = 0
in_database_roads = set()
out_of_database_roads = set()

cprint(
    f"Before processing the length of the test data is {len(test_data)} and the length of generated_data is {len(generated_data)}",
    "red",
)
new_test_data, new_generated_data = [], []
for path_collection, generated_path in tqdm(
    zip(test_data, generated_data), dynamic_ncols=True
):
    if generated_path:
        new_test_data.append(path_collection)
        new_generated_data.append(generated_path)

generated_data = new_generated_data
test_data = new_test_data
cprint(
    f"After processing the length of the test data is {len(test_data)} and the length of generated_data is {len(generated_data)}",
    "green",
)
recall, precision = 0, 0
weighted_factor = 1 / len(test_data)
diversity_recall, diversity_precision = 0, 0
for path, path_collection in tqdm(zip(generated_data, test_data), dynamic_ncols=True):
    if path_type == "fastest":
        ground_truth_path = path_collection["fastest_path_road_names"]
    elif path_type == "shortest":
        ground_truth_path = path_collection["shortest_path_road_names"]
    elif path_type == "most_used":
        ground_truth_path = path_collection["original_path_road_names"]

    original_path = path_collection["original_path_road_names"]
    similarities = 0
    path = list(dict.fromkeys(path))
    for road in path:
        if road in ground_truth_path:
            similarities += 1
        elif road in road_names:
            in_database_roads.add(road)
            # ground_truth_path_embeddings = model.embed_documents(ground_truth_path)
            # road_embedding = model.embed_query(road)
            # hits = semantic_search(road_embedding, ground_truth_path_embeddings)
            in_database_dissimilarities += 1
        else:
            out_of_database_dissimilarities += 1
            out_of_database_roads.add(road)

    local_recall = similarities / len(path)
    local_precision = similarities / len(ground_truth_path)
    recall += local_recall * weighted_factor
    precision += local_precision * weighted_factor

    diversity = 0
    for road in path:
        if road in original_path:
            diversity += 1
    local_diversity_recall = diversity / len(path)
    local_diversity_precision = diversity / len(original_path)
    diversity_recall += local_diversity_recall * weighted_factor
    diversity_precision += local_diversity_precision * weighted_factor

file_updated_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

save_results_path = "evaluation_scores/"
save_results_filename = f"eval_scores_of_{path_type}_path_gen_on_{place_name}_dataset_context_set_to_{use_context}.txt"  # noqa: F405

make_dir(save_results_path)

with open(save_results_path + save_results_filename, "a") as f:
    results = f"\
    *****************************************\n\
    Evaluation results on {place_name} dataset for {path_type} generation using:\n\
    Evaluated at : {file_updated_time}\n\
    Evaluation remarks : {evaluation_remarks}\n\
    docs_to_retrieve : {number_of_docs_to_retrieve}\n\
    llm : {llm}\n\
    embedding model : {embedding_model}\n\
    context set to: {use_context} \n\n\
    *****************************************\n\
    Precision               : {precision} \n\
    Similarity (precision)  : {diversity_precision}\n\
    Recall                  : {recall}    \n\
    Similarity (recall)     : {diversity_recall}\n\n\n"  # noqa: F405
    f.write(results)
    f.close()

print(results)
# print(in_database_dissimilarities)
# print(out_of_database_dissimilarities)
# print(out_of_database_roads)
print(in_database_roads)
