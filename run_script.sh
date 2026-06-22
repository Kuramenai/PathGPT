# python generate_touristics_paths.py -place_name $1
# python create_json_file.py -place_name $1 -path_type touristic
# python inference.py -use-context -place_name $1 -path_type touristic | tee -a inference_log.txt
# python evaluate.py -place_name $1 -path_type $2  -$3

# python generate_fuel_efficient_paths.py -place_name $1
# python create_json_file.py -place_name $1 -path_type highway_free
# python inference.py -use-context -place_name $1 -path_type highway_free | tee -a inference_log.txt
# python evaluate.py -place_name $1 -path_type $2  -$3

# python subgraph_construction.py    -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9 -top_k_shortest
# python context_generation.py       -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9
# python prompt_generation.py        -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9
# python inference.py                -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9
# python evaluate_paper_v2.py        -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9


# python context_generation.py -use-context -place_name $1 -path_type poi_aware -corridor_graph_form uncompressed
# python prompt_generation.py  -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task route_segments -top_k 9 -corridor_graph_form uncompressed
# python inference.py          -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task route_segments -top_k 9 -corridor_graph_form uncompressed
# python evaluate_paper_v2.py    -use-context -place_name $1 -path_type poi_aware -retrieval spatial_hybrid -llm_task route_segments -top_k 9 -corridor_graph_form uncompressed

python context_generation.py -use-context -place_name beijing -path_type poi_aware -corridor_graph_form uncompressed

python prompt_generation.py -use-context -place_name beijing -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9 -corridor_graph_form uncompressed

python inference.py -use-context -place_name beijing -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9 -corridor_graph_form uncompressed

python evaluate_paper_v2.py -use-context -place_name beijing -path_type poi_aware -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9 -corridor_graph_form uncompressed
