import os
import pickle
import json
import variables
from termcolor import cprint
from tqdm import tqdm
from pathlib import Path
from utils import TimeoutException, timeout_handler, make_dir
from prompt_generation import prompts_output_name, welcome_text
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

try:
    from vllm.sampling_params import StructuredOutputsParams
except ImportError:
    StructuredOutputsParams = None


def get_output_json_schema() -> dict:
    if variables.use_context and variables.llm_task == "rank_contexts":
        return {
            "type": "object",
            "properties": {
                "ranked_contexts": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                }
            },
            "required": ["ranked_contexts"],
            "additionalProperties": False,
        }

    if variables.use_context:
        return {
            "type": "object",
            "properties": {
                "route_segments": {
                    "type": "array",
                    "items": {"type": "string", "pattern": "^G[0-9]+$"},
                }
            },
            "required": ["route_segments"],
            "additionalProperties": False,
        }

    return {
        "type": "object",
        "properties": {
            "route": {
                "type": "array",
                "items": {"type": "string"},
            }
        },
        "required": ["route"],
        "additionalProperties": False,
    }


def generated_output_name() -> str:
    if variables.use_context:
        task_suffix = f"_{variables.llm_task}" if variables.llm_task != "route_segments" else ""
        return (
            f"{variables.retrieval_type}_context_{variables.place_name}_top_"
            f"{variables.number_of_docs_to_retrieve}{task_suffix}"
        )

    return f"no_context_{variables.place_name}_top_{variables.number_of_docs_to_retrieve}"


def build_sampling_params() -> SamplingParams:
    base_kwargs = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1024}
    schema = get_output_json_schema()

    if StructuredOutputsParams is not None:
        try:
            return SamplingParams(
                **base_kwargs,
                structured_outputs=StructuredOutputsParams(json=schema),
            )
        except TypeError:
            pass

    try:
        # ponytail: old vLLM compatibility path. Remove when all runs use vLLM >= 0.12.
        return SamplingParams(**base_kwargs, guided_json=schema)
    except TypeError:
        cprint(
            "Structured JSON output is not supported by this vLLM version; using plain sampling.", "yellow"
        )
        return SamplingParams(**base_kwargs)


if __name__ == "__main__":
    welcome_text()

    sampling_params = build_sampling_params()

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
        with open(prompts_filepath + prompts_output_name(number_of_documents_to_retrieve), "rb") as f:
            prompts = pickle.load(f)
    except FileNotFoundError:
        cprint(f"Prompts not found at {prompts_filepath}. Please run prompt_generation.py first.", "red")
        exit(1)

    cprint(f"Loaded {len(prompts)} prompts.", "green")

    formatted_messages = []
    for prompt in prompts:
        message = [{"role": "user", "content": prompt}]

        text = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        formatted_messages.append(text)

    cprint(f"Starting bulk generation via vLLM...", "yellow")

    responses = llm.generate(formatted_messages, sampling_params)

    results = []
    for output in responses:
        # output is a RequestOutput object
        generated_text = output.outputs[0].text.strip()

        results.append(generated_text)

    file_path = f"generated_paths/{variables.path_type}/"
    file_name = generated_output_name()

    make_dir(file_path)
    with open(file_path + file_name, "wb") as f:
        pickle.dump(results, f)

    cprint(f"File saved at {file_path}/{file_name} ", "green")

    del llm
