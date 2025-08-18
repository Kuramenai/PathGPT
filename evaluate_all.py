import os
import pickle
import os.path
from termcolor import cprint
import variables


cprint("Overall performance of PathGPT with:", "light_yellow", attrs=["bold"])
cprint(f"-llm : {variables.llm}", "green")
cprint(f"-embedding model : {variables.embedding_model_formatted_name}", "green")


curr_dir = os.getcwd()
cities = ["beijing", "chengdu", "harbin"]
path_types = ["touristic", "highway_free"]


def load_generated_paths(city, path_type, top_k, use_context=True):
    generated_data_path = f"generated_paths/{city}/{variables.llm}/{variables.embedding_model_formatted_name}/{path_type}/"
    if use_context:
        generated_data_filename = f"/use_context_{use_context}_k_{top_k}"
    else:
        generated_data_filename = f"use_context_{use_context}"

    f = open(generated_data_path + generated_data_filename, "rb")
    generated_data = pickle.load(f)
    f.close()

    test_data = load_test_data(city, path_type)

    new_test_data, new_generated_data, bad_generated_paths = [], [], 0
    for path_collection, generated_path in zip(test_data, generated_data):
        if generated_path:
            new_test_data.append(path_collection)
            if len(generated_path) == 1:
                bad_generated_paths += 1
                path = generated_path[0]
                generated_path = path.split("，")
            new_generated_data.append(generated_path)

    return new_generated_data, new_test_data


def load_test_data(city, path_type):
    test_data = f"test_data/{path_type}_paths/{city}_data"
    f = open(test_data, "rb")
    test_data = pickle.load(f)
    f.close()
    return test_data


def get_precision_recall(generated_data, test_data, path_type):
    recall, precision = 0, 0
    weighted_factor = 1 / len(test_data)
    # diversity_recall, diversity_precision = 0, 0
    for path, path_collection in zip(generated_data, test_data):
        if path_type == "fastest":
            ground_truth_path = path_collection["fastest_path_road_names"]
        elif path_type == "shortest":
            ground_truth_path = path_collection["shortest_path_road_names"]
        elif path_type == "most_used":
            ground_truth_path = path_collection["original_path_road_names"]
        elif path_type == "highway_free":
            ground_truth_path = path_collection["highway_free_path_road_names"]
        elif path_type == "touristic":
            ground_truth_path = path_collection["touristic_path_road_names"]

        # original_path = path_collection[f"{variables.path_type}_path_road_names"]
        similarities = 0
        path = list(dict.fromkeys(path))
        for road in path:
            if road in ground_truth_path:
                similarities += 1

        recall += similarities / len(path)
        precision += similarities / len(ground_truth_path)

    recall = recall * weighted_factor
    precision = precision * weighted_factor
    precision = round(precision * 100, 2)
    recall = round(recall * 100, 2)

    return precision, recall


def table_header(top_k):
    metrics = ["", "", "   Precision  ", "   Recall  "]
    models = [
        "path_type",
        "city",
        f"LLM    |  PathGPT@{top_k}",
        f"LLM    |  PathGPT@{top_k}",
    ]
    row_format = "{:25}" * (len(metrics) + 1)
    row_format2 = "{:25}" * (len(models) + 1)
    print("-" * 117)
    print(row_format.format("", *metrics))
    print("-" * 117)
    print(row_format2.format("", *models))
    print("-" * 117)


for top_k in range(3, 12, 3):
    for path_type in path_types:
        table_header(top_k)

        for city in cities:
            paths_generated_with_context, test_data_1 = load_generated_paths(
                city, path_type, top_k
            )
            paths_generated_no_context, test_data_2 = load_generated_paths(
                city, path_type, top_k, use_context=False
            )

            precision_with_context, recall_with_context = get_precision_recall(
                paths_generated_with_context, test_data_1, path_type
            )
            precision_no_context, recall_no_context = get_precision_recall(
                paths_generated_no_context, test_data_2, path_type
            )

            results = [
                precision_no_context,
                precision_with_context,
                recall_no_context,
                recall_with_context,
            ]

            res1 = [
                path_type,
                f"{city}",
                f"{results[0]}  |  {results[1]}",
                f"{results[2]}  |  {results[3]}",
            ]
            row_format3 = "{:25}" * (len(res1) + 1)
            print(row_format3.format("", *res1))

        print("\n\n")
