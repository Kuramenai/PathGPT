import json
import pickle
from tqdm import tqdm
from termcolor import cprint
from utils import make_dir
import variables


cprint("\n\nGENERATING PDF FILES FOR :", "yellow", attrs=["bold"])
cprint(f"-DATASET : {variables.place_name}", "green")
cprint(f"-SAVING AS : {variables.save_as}\n", "green")


data = f"train_data/{variables.place_name}_data"

cprint("Loading data...", "light_yellow")

f = open(data, "rb")
dataset = pickle.load(f)
f.close()

cprint("Dataset loaded successfully!\n", "light_green")

cprint("Generating documents...", "light_yellow")

if variables.args.save_as == "one_path_one_doc":
    cprint(
        "Unsupported right now, please use the 'save_as' argument with 'all_paths_one_doc'.",
        "red",
    )


# Save all the textual representations of each path in one big document
elif variables.args.save_as == "all_paths_one_doc":
    make_dir("json_files")
    routing_database = []

    for path_info in tqdm(dataset, dynamic_ncols=True):
        original_path_road_names = path_info["original_path_road_names"]
        shortest_path_road_names = path_info["shortest_path_road_names"]
        fastest_path_road_names = path_info["fastest_path_road_names"]

        starting_address = original_path_road_names[0]
        destination_address = original_path_road_names[-1]

        routing_info = {
            "content": f"从{starting_address}到{destination_address}的最常走的路线是经过{'，'.join(original_path_road_names)}这几个路段。而从{starting_address}到{destination_address}的最短的路线是经过{'，'.join(shortest_path_road_names)}这些路。另外同样从{starting_address}到{destination_address}的最快路线是经过{'，'.join(fastest_path_road_names)}这些路。"
        }

        routing_database.append(routing_info)

    with open(
        f"json_files/{variables.place_name}_paths.json", "w", encoding="utf-8"
    ) as f:
        json.dump(routing_database, f, indent=2, ensure_ascii=False)
