import os
import torch
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator, SequentialEvaluator
from sentence_transformers.util import cos_sim
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from termcolor import cprint

from variables import *

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

cprint("\nLoading dataset...", 'light_yellow')
dataset = load_dataset('csv', data_files=f'csv_dataset/{place_name}/{path_type}_paths.csv', split='train')
cprint("Dataset loaded successfully!", 'light_green')

dataset = dataset.rename_column("question", "anchor")
dataset = dataset.rename_column("context", "positive")

dataset_without_answers = dataset.remove_columns('answer')

dataset_without_answers = dataset_without_answers.add_column("id", range(len(dataset_without_answers)))

dataset_without_answers = dataset_without_answers.train_test_split(test_size = 0.1)

# dataset_without_answers["train"].to_json("/train_dataset.json", orient="records")

cprint("\nLoading embeddings...", 'light_yellow')
# model = HuggingFaceEmbeddings(
#         model_name=embedding_model,
#         model_kwargs=model_kwargs,
#         encode_kwargs=encode_kwargs,
#         show_progress = False)
model_id = "intfloat/multilingual-e5-large-instruct"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer(
        model_id, 
        device)
cprint("Embeddings loaded successfully!", 'light_green')

test_dataset = dataset_without_answers['test']
train_dataset = dataset_without_answers['train']
corpus_dataset = concatenate_datasets([train_dataset, test_dataset])

corpus = dict(zip(corpus_dataset['id'], corpus_dataset["positive"]))
queries = dict(zip(test_dataset['id'], test_dataset['anchor']))

relevant_documents = {}
for q_id in tqdm(queries, dynamic_ncols=True):
    relevant_documents[q_id] = [q_id]

ir_evaluator = InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_documents,
                name='hello',
                score_functions={"cosine": cos_sim},
                show_progress_bar=True,
                batch_size=4)

cprint("\nEvaluation started...", 'light_yellow')
results = ir_evaluator(model)

cprint("\nEvaluation results", 'light_green')
cprint("-" * 85, 'light_blue')
cprint(f"{'Metric':15} {'Values':.12}", 'light_red')
cprint("-" * 85, 'light_blue')

metrics = [
    'ndcg@10',
    'mrr@10',
    'map@100',
    'accuracy@1',
    'accuracy@3',
    'accuracy@5',
    'accuracy@10',
    'precision@1',
    'precision@3',
    'precision@5',
    'precision@10',
    'recall@1',
    'recall@3',
    'recall@5',
    'recall@10']

for metric in metrics:
    values = []
    key = f"hello_cosine_{metric}"
    values.append(results[key])

    metric_name = f"=={metric}==" if metric == "ndcg@10" else metric
    print(f"{metric_name:15}", end="  ")
    for val in values:
        print(f"{val:12.4f}", end=" ")
    print()

print("-" * 85)
print(f"{'seq_score:'} {results['sequential_score']:1f}")