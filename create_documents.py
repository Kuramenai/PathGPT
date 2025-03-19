import csv
import pickle
from tqdm import tqdm
from termcolor import cprint 

from utils import get_args, make_dir
from variables import * 


cprint('\n\nGENERATING PDF FILES FOR :', 'yellow', attrs=['bold'])
cprint(f'-DATASET : {place_name}', 'green')
cprint(f'-SAVING AS : {save_as}\n', 'green')


data = f"train/{place_name}_data"

cprint('Loading data...', 'light_yellow')

f = open(data, 'rb')
dataset = pickle.load(f)
f.close()

cprint('Dataset loaded successfully!\n', 'light_green')

cprint('Generating documents...', 'light_yellow')

if args.save_as == 'one_path_one_doc':

    cprint("Unsupported right now, please use the 'save_as' argument with 'all_paths_one_doc'.", 'red')
       

# Save all the textual representations of each path in one big document
elif args.save_as == 'all_paths_one_doc':

    document_path = f'markdown_docs/'
    mdFile = document_path + f'{place_name}_paths.md'
    csv_dataset_path = f'csv_dataset/{place_name}/'
    csv_dataset_for_fastest_paths = 'fastest_paths.csv'
    csv_dataset_for_shortest_paths = 'shortest_paths.csv'
    csv_dataset_for_most_used_paths = 'most_used_paths.csv'

    make_dir(document_path)
    make_dir(csv_dataset_path)

    """MARKDOWN AND CSV"""

    csv_columns = ["question", "context", "answer"]
    with open(mdFile, 'w') as f, \
         open(csv_dataset_path + csv_dataset_for_fastest_paths, 'w') as csv_file1, \
         open(csv_dataset_path + csv_dataset_for_shortest_paths, 'w') as csv_file2,\
         open(csv_dataset_path + csv_dataset_for_most_used_paths, 'w') as csv_file3:
        
        writer1 = csv.writer(csv_file1)
        writer2 = csv.writer(csv_file2)
        writer3 = csv.writer(csv_file3)
        
        writer1.writerow(csv_columns)
        writer2.writerow(csv_columns)
        writer3.writerow(csv_columns)

        
        for path_info in tqdm(dataset, dynamic_ncols=True):

            original_path_with_road_names = path_info['original_path_with_road_names']
            shortest_path_with_road_names = path_info['shortest_path_with_road_names']
            fastest_path_with_road_names = path_info['fastest_path_with_road_names']
            
            starting_address = original_path_with_road_names[0]
            destination_address = original_path_with_road_names[-1]


            task_description = 'Given the following user query, retrieve relevant passages that answer or provide information to help answer it'

            question1 = f'Generate the fastest path from {starting_address} to {destination_address}.'
            question2 = f'Generate the shortest path from {starting_address} to {destination_address}.'
            question3 = f'Generate the most commonly used path from {starting_address} to {destination_address}.'

            retriever_query1 = f'Instruct: {task_description}\nQuery: {question1}'
            retriever_query2 = f'Instruct: {task_description}\nQuery: {question2}'
            retriever_query3 = f'Instruct: {task_description}\nQuery: {question3}'

            context1 = f"### Below are three paths that start from {starting_address} to reach {destination_address}:\n\
            * The fastest path which crosses: {','.join(fastest_path_with_road_names)}\n\
            * Then the shortest path that crosses: {','.join(shortest_path_with_road_names)}\n\
            * And the last but not least is the most used path that crosses: {','.join(original_path_with_road_names)}\n\n"

            context = f"### 从{starting_address}到{destination_address}可以走一下三个路线:\n\
            * 最**快**的路线，它经过的路为: {', '.join(fastest_path_with_road_names)}\n\
            * 最**短**的路线，它经过的路为: {', '.join(shortest_path_with_road_names)}\n\
            * 最**常用**的路线，它经过的路为: {', '.join(original_path_with_road_names)}\n\n"

            answer1 = ','.join(fastest_path_with_road_names)
            answer2 = ','.join(shortest_path_with_road_names)
            answer3 = ','.join(original_path_with_road_names)

            row1 = [retriever_query1, context, answer1]
            row2 = [retriever_query2, context, answer2]
            row3 = [retriever_query3, context, answer3]


            writer1.writerow(row1)
            writer2.writerow(row2)
            writer3.writerow(row3)

            f.write(context)

    f.close()

cprint(f'Document {mdFile} generated successfully!\n', 'light_green')
cprint(f'CSV dataset {csv_dataset_path}{csv_dataset_for_fastest_paths} generated successfully!\n', 'light_green')
cprint(f'CSV dataset {csv_dataset_path}{csv_dataset_for_shortest_paths} generated successfully!\n', 'light_green')
cprint(f'CSV dataset {csv_dataset_path}{csv_dataset_for_most_used_paths} generated successfully!\n', 'light_green')


    