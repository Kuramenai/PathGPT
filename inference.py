import os
import pickle
import json
import variables
from termcolor import cprint
from tqdm import tqdm
from pathlib import Path
from utils import TimeoutException, timeout_handler, make_dir
from prompt_generation import welcome_text
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

if __name__ == "__main__":
    
    welcome_text()
    
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=1024)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    # Initialize the vLLM engine. It will automatically manage the batch queue.
    llm = LLM(model="/root/autodl-tmp/hf-models/Qwen3-8B", gpu_memory_utilization=0.95)

    # Prepare the input to the model
    if variables.use_context:
        prompts_filepath = f"prompts/{variables.path_type}/with_context/"
    else:
        prompts_filepath = f"prompts/{variables.path_type}/no_context/"
        
    number_of_documents_to_retrieve = variables.number_of_docs_to_retrieve
    
    try:
        with open(prompts_filepath + f"{variables.place_name}_prompts_top_{number_of_documents_to_retrieve}", "rb") as f:
            prompts = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Prompts not found at {prompts_filepath}. Please run prompt_generation.py first.", "red")
        exit(1)
    
    cprint(f"Loaded {len(prompts)} prompts.", "green")
    
    formatted_messages = []
    for prompt in prompts:
        message = [
            {"role": "user", "content": prompt}
        ]
    
        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        formatted_messages.append(text)

    cprint(f"Starting bulk generation via vLLM...", "yellow")

    responses = llm.generate(formatted_messages, sampling_params)

    results = []
    i = 0
    for output in responses:
        i += 1
        # output is a RequestOutput object
        generated_text = output.outputs[0].text.strip()
        
        results.append(generated_text)
        print(f"Generated text: {generated_text}")
        if i == 5:
            break
        
    file_path = f"generated_paths/{variables.path_type}/"
    if variables.use_context:
        file_name = f"with_context_{variables.place_name}_top_{variables.number_of_docs_to_retrieve}"
    else:
        file_name = f"no_context_{variables.place_name}_top_{variables.number_of_docs_to_retrieve}"

    make_dir(file_path)
    with open(file_path + file_name, "wb") as f:
        pickle.dump(results, f)

    cprint(f"File saved at {file_path}/{file_name} ", "green")