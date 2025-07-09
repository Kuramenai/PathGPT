import os
import pickle
import os.path
from tqdm import tqdm
from argparse import ArgumentParser
from termcolor import cprint
from datetime import datetime

from utils import get_args, make_dir
from variables import *


cprint(f'\nOverall performance of PathGPT with:', 'light_yellow', attrs=['bold'])
cprint(f'-llm : {llm}', 'green')
cprint(f'-embedding model : {embedding_model_formatted_name}', 'green')
cprint(f'-no. retrieved documents : {number_of_docs_to_retrieve}\n', 'green')

curr_dir = os.getcwd()
cities = ['beijing', 'chengdu', 'harbin']
path_types = ['fastest', 'shortest']

def load_generated_paths(city, path_type, use_context=True):

    file_path = os.path.join(curr_dir, 'generated_paths', f'{city}', f'{llm}',\
                        f'{embedding_model_formatted_name}', f'{path_type}', \
                        f'use_context_{use_context}')
    
    f = open(file_path, 'rb')
    generated_paths = pickle.load(f)
    f.close()
    return generated_paths

def load_test_data(city):

    test_data = f'test_data/{city}_data'

    f = open(test_data, 'rb')
    test_data = pickle.load(f)
    f.close()
    return test_data


def get_precision_recall(generated_paths, test_data, path_type):
    recall, precision = 0, 0
    weighted_factor = 1/len(test_data)

    for path, path_collection in zip(generated_paths, test_data):

        ground_truth_path = path_collection[f'{path_type}_path_road_names']

        similarities = 0
        path = list(dict.fromkeys(path))
        for road in path:
            if road in ground_truth_path:
                similarities += 1
        local_recall = similarities/ len(path)
        local_precision = similarities/len(ground_truth_path)
        recall += local_recall * weighted_factor
        precision += local_precision * weighted_factor

    precision = round(precision*100, 2)
    recall = round(recall*100, 2)

    return precision, recall


def table_header():
    metrics = ['', '', '   Precision  ', '   Recall  ']
    models =  ['path_type', 'city','LLM    |  pathGPT','LLM    |  pathGPT']
    row_format = "{:25}" * (len(metrics) + 1)
    row_format2 = "{:25}" * (len(models) + 1)
    print("-"*117)
    print(row_format.format("", *metrics))
    print("-"*117)
    print(row_format2.format("", *models))
    print("-"*117)


for path_type in path_types:
    table_header()


    for city in cities:

        test_data = load_test_data(city)

        fastest_paths_generated_with_context = load_generated_paths(city, path_type)
        fastest_paths_generated_no_context = load_generated_paths(city, path_type, use_context=False)

        precision_with_context, recall_with_context = get_precision_recall(fastest_paths_generated_with_context, test_data, path_type)
        precision_no_context, recall_no_context = get_precision_recall(fastest_paths_generated_no_context, test_data, path_type)

        results = [precision_no_context, precision_with_context, recall_no_context, recall_with_context]

        res1 = [path_type, f'{city}', f'{results[0]}  |  {results[1]}', f'{results[2]}  |  {results[3]}']
        row_format3 = "{:25}" * (len(res1) + 1)
        print(row_format3.format("", *res1))

    print("\n\n")
