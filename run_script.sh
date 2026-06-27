# Data Augmentation
python data_augmentation.py -place_name $1 -path_type $2

# Data Preprocessing
python data_preprocessing.py -place_name $1 -path_type $2

# Subgraph Construction
python subgraph_construction.py -use-context -place_name $1 -path_type $2 -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9 -top_k_shortest

# Context Generation
python context_generation.py -use-context -place_name $1 -path_type $2 -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9

# Prompt Generation
python prompt_generation.py -use-context -place_name $1 -path_type $2 -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9

# Inference
python inference.py -use-context -place_name $1 -path_type $2 -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9

# Path Generation and Evaluation
python evaluate_paper_v2.py -use-context -place_name $1 -path_type $2 -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9