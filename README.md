# LLMDrive: LLM-Guided Path Recommendation with Retrieved Topological Corridors

This repository hosts the implementation of **LLMDrive**: a retrieval-augmented path recommendation framework. It builds local topological corridors from trajectories and graph paths, retrieves relevant corridors for each query, and uses an LLM to recommend paths.

Supported cities: `beijing`, `chengdu`, `harbin`.

# Environment setup

For reference, experiments were run on a server with Ubuntu 22.04 LTS and an NVIDIA RTX 5090 (CUDA 13.0). Python **3.12.13** is recommended.

## Install dependencies

Create and activate a virtual environment, then install requirements:

```bash
conda create -n llmdrive python=3.12.13
conda activate llmdrive
pip install -r requirements.txt
```

**N.B.** We use `bm25s` for lexical search. For Chinese text, replace the default tokenizer with `jieba.cut` in `tokenize.py` after installing `jieba`.

## LLM inference (Qwen3-8B)

The reproduction script uses **vLLM** (`inference.py`). Update the model path in `inference.py` if needed:

```python
llm = LLM(model="/path/to/Qwen3-8B", gpu_memory_utilization=0.95)
```

Qwen3-8B (FP16) needs roughly **16 GB VRAM** on GPU for the model only. You can also serve the model with **Ollama** for interactive use, but the bundled pipeline expects vLLM outputs with structured JSON (`anchor_segments`).

## Data — credits to [NeuroMLR](https://github.com/idea-iitd/NeuroMLR)

Follow [NeuroMLR](https://github.com/idea-iitd/NeuroMLR) and download the [preprocessed data](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view?usp=sharing). Unzip it so each city lives under `preprocessed_data/{place_name}_data/`.

Map paths are set via `-place_name` (see `variables.py`); default layout:

- `preprocessed_data/{place_name}_data/map/nodes.shp`
- `preprocessed_data/{place_name}_data/map/edges.shp`
- `preprocessed_data/{place_name}_data/map/graph_with_haversine.pkl`
- Pickled trajectories: `preprocessed_train_trips_all.pkl`, `preprocessed_test_trips_all.pkl`, etc.

For **POI-aware** tasks (`poi_aware`, `scenic`), extract POIs into `pois/{place_name}_pois` (from `pois.rar` in the release).

# Reproducibility

## Quick start (paper v2 — anchor segments)

```bash
bash run_script.sh beijing
```

This runs, in order:

1. `data_augmentation.py`
2. `data_preprocessing.py`
3. `subgraph_construction.py` (`-top_k_shortest` enabled in `run_script.sh`)
4. `context_generation.py`
5. `prompt_generation.py`
6. `inference.py`
7. `evaluate_paper_v2.py`

**Defaults in `run_script.sh`:** `retrieval=spatial_hybrid`, `llm_task=anchor_segments`, `top_k=9`, `corridor_graph_form=compressed`.

To change city, pass one argument: `bash run_script.sh chengdu poi_aware`. To change path type or flags, edit `run_script.sh` or run steps manually (below).

## Manual run (single step)

Example matching the v2 pipeline:

```bash
python prompt_generation.py -use-context -place_name beijing -path_type poi_aware \
  -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9

python inference.py -use-context -place_name beijing -path_type poi_aware \
  -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9
```

`top_k` is the number of retrieved corridor contexts shown to the LLM.

## Evaluation

Primary evaluator:

```bash
python evaluate_paper_v2.py -use-context -place_name beijing -path_type poi_aware \
  -retrieval spatial_hybrid -llm_task anchor_segments -top_k 9
```

Report the main system row. Other rows are ablations (retrieved union, anchor-only soft prior, random anchors, task-optimal oracle on the full graph).

Common `-path_type` values: `poi_aware`, `scenic`, `fuel_efficient`, `fastest`, `shortest`, `highway_free`.

Optional flags: `-top_k_shortest`, `-k_shortest 3`, `-corridor_graph_form uncompressed`.