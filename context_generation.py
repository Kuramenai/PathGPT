import json
import pickle
import variables
from tqdm import tqdm
from termcolor import cprint
from utils import make_dir
from typing import List, Dict, Any


def generate_llm_context(path_collection: List[Dict[str, List[int]]]) -> Dict[str, Any]:
    """
    Generates a prompt context tailored for Chinese-language LLM routing.

    Args:
        path_collection List[Dict[str, List[int]]: dictionary containing path information, including: fastest, shortest, fuel-efficient (more coming), and historical driver paths.

    Returns:
        Dict[str, Any]: context in JSON format, structured to highlight key route information for LLM analysis.
    """

    orig_names = path_collection["original_path_edges_names"]

    origin = orig_names[0] if orig_names else "未知起点"
    destination = orig_names[-1] if orig_names else "未知终点"

    # fmt: off
    llm_context = {
        "task_info": "请根据以下路线信息进行路径规划分析。",
        "origin_起点": origin,
        "destination_终点": destination,
        "available_routes_可选路线": {
            "historical_driver_path_历史真实轨迹": " -> ".join(orig_names),
            "fastest_path_最快路线": " -> ".join(path_collection["fastest_path_edges_names"]),
            "shortest_path_最短路线": " -> ".join(path_collection["shortest_path_edges_names"]),
            "fuel_efficient_path_最省油路线": " -> ".join(path_collection["custom_path_edges_names"]),
        },
    }

    # fmt: on

    return llm_context


if __name__ == "__main__":
    cprint("\n\nGENERATING CONTEXT FOR :", "yellow", attrs=["bold"])
    cprint(f"-DATASET : {variables.place_name}", "green")
    cprint(f"-SAVING AS : {variables.save_as}", "green")
    cprint(f"-PATH TYPE: {variables.path_type}\n", "green")

    cprint("Loading data...", "light_yellow")
    # fmt: off
    try:
        with open(f"train_data/{variables.path_type}/{variables.place_name}_data", "rb") as f:
            dataset = pickle.load(f)
    except FileNotFoundError:
        cprint("Dataset not found. Please run the path extraction script first.", "red")
        exit(1)

    cprint("Dataset loaded successfully!\n", "light_green")

    llm_contexts = []
    for path_collection in tqdm(
        dataset, total=len(dataset), dynamic_ncols=True, desc="Generating LLM contexts"
    ):
        llm_context = generate_llm_context(path_collection)
        llm_contexts.append(llm_context)

    # fmt: off
    path = f"json_files/{variables.path_type}"
    make_dir(path)

    with open(f"{path}/{variables.place_name}_paths.json", "w", encoding="utf-8") as f:
        json.dump(llm_contexts, f, indent=2, ensure_ascii=False)
