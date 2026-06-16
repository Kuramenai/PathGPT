# python generate_touristics_paths.py -place_name $1
# python create_json_file.py -place_name $1 -path_type touristic
# python inference.py -use-context -place_name $1 -path_type touristic | tee -a inference_log.txt
# python evaluate.py -place_name $1 -path_type $2  -$3

# python generate_fuel_efficient_paths.py -place_name $1
# python create_json_file.py -place_name $1 -path_type highway_free
# python inference.py -use-context -place_name $1 -path_type highway_free | tee -a inference_log.txt
# python evaluate.py -place_name $1 -path_type $2  -$3

python subgraph_construction.py    -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task rank_contexts -top_k 9
python context_generation.py       -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task rank_contexts -top_k 9
python prompt_generation.py        -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task rank_contexts -top_k 9
python inference.py                -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task rank_contexts -top_k 9
python evaluate_ranked_contexts.py -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task rank_contexts -top_k 9
