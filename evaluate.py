import os
import pickle
import os.path
from tqdm import tqdm
from termcolor import cprint
from datetime import datetime
import geopandas as gpd

import variables
from utils import make_dir


cprint(
    f"EVALUATING PERFORMANCE ON {variables.place_name} DATASET WITH:",
    "light_yellow",
    attrs=["bold"],
)
cprint(f"-LLM : {variables.llm}", "green")
cprint(f"-PATH TYPE : {variables.path_type}", "green")
cprint(f"-EMBEDDING MODEL : {variables.embedding_model_formatted_name}", "green")
cprint(f"-USE CONTEXT : {variables.use_context}", "green")
cprint(f"-RETRIEVED DOCUMENTS : {variables.number_of_docs_to_retrieve}", "green")

edges_df = gpd.read_file(variables.EDGE_DATA)
edges_uv_road_names = edges_df[["name", "length"]].to_numpy()
map_edges_id_to_road_names = {
    edge_info[0]: edge_info[1] for edge_info in edges_uv_road_names
}
place_name = variables.place_name
script_dir = os.path.dirname(os.path.abspath(__file__))
road_names_file = f"road_names/{place_name}_road_names"
road_names_file = os.path.join(script_dir, road_names_file)
if os.path.exists(road_names_file):
    f = open(road_names_file, "rb")
    road_names = pickle.load(f)
    f.close
road_names = list(road_names.values())

generated_data_path = f"generated_paths/{place_name}/{variables.llm}/{variables.embedding_model_formatted_name}/{variables.path_type}/"
if variables.use_context:
    generated_data_filename = (
        f"/use_context_{variables.use_context}_k_{variables.number_of_docs_to_retrieve}"
    )
else:
    generated_data_filename = f"use_context_{variables.use_context}"

f = open(generated_data_path + generated_data_filename, "rb")
generated_data = pickle.load(f)
f.close()

test_data_path = "test_data/"
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
    "yellow",
)
new_test_data, new_generated_data, bad_generated_paths = [], [], 0
for path_collection, generated_path in tqdm(
    zip(test_data, generated_data), dynamic_ncols=True
):
    if generated_path:
        new_test_data.append(path_collection)
        if len(generated_path) == 1:
            bad_generated_paths += 1
            path = generated_path[0]
            generated_path = path.split("，")
        new_generated_data.append(generated_path)

generated_data = new_generated_data
test_data = new_test_data
cprint(
    f"After processing the length of the test data is {len(test_data)} and the length of generated_data is {len(generated_data)}, \
also found {bad_generated_paths} badly encoded paths.",
    "green",
)
recall, precision = 0, 0
weighted_factor = 1 / len(test_data)
for path, path_collection in tqdm(zip(generated_data, test_data), dynamic_ncols=True):
    if variables.path_type == "fastest":
        ground_truth_path = path_collection["fastest_path_road_names"]
    elif variables.path_type == "shortest":
        ground_truth_path = path_collection["shortest_path_road_names"]
    elif variables.path_type == "most_used":
        ground_truth_path = path_collection["original_path_road_names"]

    similarities = 0
    path = list(dict.fromkeys(path))
    for road in path:
        if road in ground_truth_path:
            similarities += 1
        elif road in road_names:
            in_database_roads.add(road)
            in_database_dissimilarities += 1
        else:
            out_of_database_dissimilarities += 1
            out_of_database_roads.add(road)

    recall += similarities / len(path)
    precision += similarities / len(ground_truth_path)

recall = round(recall * weighted_factor, 2)
precision = round(precision * weighted_factor, 2)


file_updated_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

save_results_path = "evaluation_scores/"
save_results_filename = f"eval_scores_of_{variables.path_type}_path_gen_on_{place_name}_dataset_context_set_to_{variables.use_context}.txt"  # noqa: F405

make_dir(save_results_path)

with open(save_results_path + save_results_filename, "a") as f:
    results = f"\
    *****************************************\n\
    Evaluation results on {place_name} dataset for {variables.path_type} generation using:\n\
    Evaluated at : {file_updated_time}\n\
    Evaluation remarks : {variables.evaluation_remarks}\n\
    docs_to_retrieve : {variables.number_of_docs_to_retrieve}\n\
    llm : {variables.llm}\n\
    embedding model : {variables.embedding_model}\n\
    context set to: {variables.use_context} \n\n\
    *****************************************\n\
    Precision               : {precision} \n\
    Recall                  : {recall}    \n\n"  # noqa: F405
    f.write(results)
    f.close()

print(results)
# print(in_database_dissimilarities)
# print(out_of_database_dissimilarities)
# print(out_of_database_roads)
# print(in_database_roads)
