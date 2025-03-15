import pickle
from tqdm import tqdm
from argparse import ArgumentParser
from termcolor import cprint
from datetime import datetime

from utils import get_args
from variables import *


cprint(f'EVALUATING PERFORMANCE ON {place_name} DATASET WITH:', 'light_yellow', attrs=['bold'])
cprint(f'-LLM : {llm}', 'green')
cprint(f'-PATH TYPE : {path_type}', 'green')
cprint(f'-EMBEDDING MODEL : {embedding_model_formatted_name}', 'green')
cprint(f'-USE CONTEXT : {use_context}', 'green')
cprint(f'-RETRIEVED DOCUMENTS : {number_of_docs_to_retrieve}', 'green')

generated_data_path = f'generated_paths/{place_name}/{llm}/{embedding_model_formatted_name}/{path_type}/'
generated_data_filename = f'use_context_{use_context}_k_{number_of_docs_to_retrieve}'

f = open(generated_data_path + generated_data_filename, 'rb')
generated_data = pickle.load(f)
f.close()

test_data_path = f'test/'
test_data_filename = f'{place_name}_data'

f = open(test_data_path + test_data_filename, 'rb')
test_data = pickle.load(f)
f.close()

recall, precision = 0, 0
weighted_factor = 1/len(test_data)
diversity_recall, diversity_precision = 0, 0
for path, path_collection in tqdm(zip(generated_data, test_data), dynamic_ncols = True):
    if path_type == 'fastest':
        ground_truth_path = path_collection['fastest_path_with_road_names']
    elif path_type == 'shortest':
        ground_truth_path = path_collection['shortest_path_with_road_names']
    elif path_type == 'most_used':
        ground_truth_path = path_collection['original_path_with_road_names']
    
    original_path = path_collection['original_path_with_road_names']
    similarities = 0
    path = list(dict.fromkeys(path))
    for road in path:
        if road in ground_truth_path:
            similarities += 1
    local_recall = similarities/ len(path)
    local_precision = similarities/len(ground_truth_path)
    recall += local_recall * weighted_factor
    precision += local_precision * weighted_factor
    
    diversity = 0
    for road in path:
        if road in original_path:
            diversity += 1
    local_diversity_recall =  diversity/ len(path)
    local_diversity_precision =  diversity/len(original_path)
    diversity_recall +=  local_diversity_recall * weighted_factor
    diversity_precision += local_diversity_precision * weighted_factor

file_updated_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

save_results_path = 'evaluation_scores/'
save_results_filename = f'{place_name}_dataset_eval_scores.txt'

with open(save_results_path + save_results_filename, 'a') as f:
    results = f"\
    *****************************************\n\
    Evaluated at : {file_updated_time}\n\
    *****************************************\n\
    【updated dataset】Evaluation results on {place_name} dataset for {path_type} generation using:\n\
    docs_to_retrieve : {number_of_docs_to_retrieve}\n\
    llm : {llm}\n\
    embedding model : {embedding_model}\n\
    context set to: {use_context} \n\n\
    Precision : {precision}\n\
    Recall : {recall}\n\
    Diversity(precision) : {diversity_precision}\n\
    Diversity(recall) : {diversity_recall}\n\n\n"
    f.write(results)
    f.close()

print(precision, recall, diversity_precision, diversity_recall)