import os
import pickle
import os.path
from tqdm import tqdm
from argparse import ArgumentParser
from termcolor import cprint
from datetime import datetime

from utils import get_args, make_dir
from variables import *


cprint(f'Overalll performance of PathGPT with:', 'light_yellow', attrs=['bold'])
cprint(f'-llm : {llm}', 'green')
cprint(f'-path type : {path_type}', 'green')
cprint(f'-embedding model : {embedding_model_formatted_name}', 'green')
cprint(f'-use conext : {use_context}', 'green')
cprint(f'-no. retrieved documents : {number_of_docs_to_retrieve}', 'green')

curr_dir = os.getcwd()
cities = ['beijing', 'chengdu', 'harbin']

def load_generated_paths(city, path_type, use_context=True):

    file_path = os.path.join(curr_dir, 'generated_paths', f'{city}', f'{llm}',\
                        f'{embedding_model_formatted_name}', f'{path_type}', \
                        f'use_context_{use_context}')
    
    f = open(file_path, 'rb')
    generated_paths = pickle.load(f)
    f.close()
    return generated_paths

def load_test_datasets(city):

    test_data = f'test_data/{city}_data'

    f = open(test_data, 'rb')
    test_data = pickle.load(f)
    f.close()
    return test_data


def get_precision_recall(generated_paths, test_data, path_type):
    recall, precision = 0, 0
    weighted_factor = 1/len(test_data)

    for path, path_collection in tqdm(zip(generated_paths, test_data), dynamic_ncols = True):

        ground_truth_path = path_collection['fastest_path_road_names']

        similarities = 0
        path = list(dict.fromkeys(path))
        for road in path:
            if road in ground_truth_path:
                similarities += 1
        local_recall = similarities/ len(path)
        local_precision = similarities/len(ground_truth_path)
        recall += local_recall * weighted_factor
        precision += local_precision * weighted_factor







all_generated_paths_with_context = []
all_generated_paths_without_context = []
all_test_data = []

for city in cities:

    generated_paths_with_context = load_generated_paths(city)
    generated_paths_without_context = load_generated_paths(city, use_context=False)
    test_data = load_test_datasets(city)

    all_generated_paths_with_context.append(generated_paths_with_context)
    all_test_data.append(all_test_data)
    all_generated_paths_without_context.append(all_generated_paths_without_context)

for generated_paths_with_context, generated_paths_without_context, test_data in \
    zip(all_generated_paths_with_context, all_generated_paths_without_context, all_test_data):
    recall, precision = 0, 0
    weighted_factor = 1/len(test_data)
    diversity_recall, diversity_precision = 0, 0
    for path, path_collection in tqdm(zip(generated_paths_with_context, test_data), dynamic_ncols = True):

        if path_type == 'fastest':
            ground_truth_path = path_collection['fastest_path_road_names']
        elif path_type == 'shortest':
            ground_truth_path = path_collection['shortest_path_road_names']



    results = [precision, recall]
        
        # diversity = 0
        # for road in path:
        #     if road in original_path:
        #         diversity += 1
        # local_diversity_recall =  diversity/ len(path)
        # local_diversity_precision =  diversity/len(original_path)
        # diversity_recall +=  local_diversity_recall * weighted_factor
        # diversity_precision += local_diversity_precision * weighted_factor
        # results = []

    





