# LLMDrive: LLM-Guided Path Recommendation with Retrieved Topological Corridors

This repository hosts the implementation code of the paper LLMDrive: LLM-Guided Path Recommendation with Retrieved Topological Corridors

<!-- # Introduction

Path recommendation aims to generate routes tailored to user preferences, yet traditional routing methods optimize fixed cost metrics and learning-based methods require retraining for each new preference—neither scales to diverse, open-ended requirements. Although large language models (LLMs) enable natural-language-driven planning, they often produce topologically invalid routes due to weak spatial reasoning. We present LLMDrive, a retrieval-augmented generation framework with three key designs. (1) We construct local topological corridors by merging historical trajectories with graph-derived paths, and compress them into decision graphs that preserve road connectivity within LLM context budgets. (2) We convert road edges into natural-language descriptions with semantic attributes and explicit adjacency, bridging the modality gap with LLMs. (3) We design a two-stage pipeline (BM25 lexical filtering followed by dense semantic retrieval) to efficiently identify relevant corridors. Experiments on taxi trajectory datasets from three cities over scenic and fuel-efficient route recommendation tasks show that LLMDrive, without task-specific fine-tuning, is competitive with supervised baselines and outperforms the same LLM without retrieved context by over 30 precision points.

![context_generation](https://github.com/user-attachments/assets/e3c94bdb-c2ad-4eb8-b80a-ac830c29dd34)
![context_generation](https://github.com/user-attachments/assets/e3c94bdb-c2ad-4eb8-b80a-ac830c29dd34)

Inspired by recent advances in the field of Large Language
Models (LLMs), we leveraged their natural language understanding capabilities to develop a unified model while being seamlessly adaptable to new scenarios without additional training.
To accomplish this, we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context in
formation, similar to RAG (Retrieved Augmented Generation) systems, to enhance their ability to generate paths according to user-defined requirements.  -->

<!-- ![pathgpt_framework](https://github.com/user-attachments/assets/9160e97f-12ea-4905-b752-2ca7e0ed6519) -->

# Environment setup
For reference, all of our experiments were conducted on a server machine running Ubuntu 22.04 LTS with a NVIDIA RTX 4050.

## Local deployment of Qwen3-8b
We use Qwen3-8b as the base LLM for our experiments which can be locally deployed through frameworks such as Ollama. Below is an example for local deployment using Ollama.
- First, install Ollama on your local machine
    * If you are on Linux, run:
      ```bash
      curl -fsSL https://ollama.com/install.sh | sh
      ```
    * If you are on Windows or macOS, you can download it [here](https://ollama.com/download) and follow the official installation instructions
- Then download and deploy Qwen3-8b by running:
    ```bash
   ollama run qwen3:8b
    ```
**N.B.**
- For optimal performance, we recommend running the model on a GPU
- Qwen3-8b requires at least 10 GB of VRAM, so make sure you have enough VRAM  if you are planning to run it on a GPU.
- The ollama service needs to be launched manually on Ubuntu 22.04 LTS. This can be accomplished by running:
  ```bash
  sudo systemctl enable ollama
  sudo systemctl start ollama
  ```
## Data - Credits to [NeuroMLR](https://github.com/idea-iitd/NeuroMLR)
Following instructions listed from [NeuroMLR](https://github.com/idea-iitd/NeuroMLR).
Download the [preprocessed data](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view?usp=sharing) and unzip the downloaded .zip file in the same directory as the other files.

Set the PREFIX_PATH variable in `varibales.py` through the `place_name` variable.

For each city (Beijing, Chengdu, Harbin), there are two types of data:

#### 1. Mapmatched pickled trajectories

Stored as a Python pickled list of tuples, where each tuple is of the form (trip_id, trip, time_info). Here, each trip is a list of edge identifiers.


#### 2. OSM map data
	
In the map folder, there are the following files-

1. `nodes.shp`: Contains OSM node information (global node id mapped to (latitude, longitude)) 
2. `edges.shp`: Contains network connectivity information (global edge id mapped to corresponding node ids)
3. `graph_with_haversine.pkl`: Pickled NetworkX graph corresponding to the OSM data

### JSON files 
In place extract the json documents located in the json_files/highway_free folder.

### POIs
Extract the POIs (Points of Interests) from the folder pois.rar.
   
## Install dependencies
Before installing the dependencies, we recommend first creating a virtual environment. 
For example, you can use conda (assuming it's already installed on your system) to create and activate a virtual environment called pathgpt by entering the following commands:
```bash
conda create -n pathgpt python=3.10
conda activate pathgpt
```
The dependencies can then be installed by running:
```bash
python install -r requirements.txt
```
The recommended Python version is 3.10. If you are using CUDA, we recommend CUDA 12.4.

N.B: We use the bm25s library to perform lexical search, however this library doesn't have a native tokenizer for Chinese text yet, therefore the default tokenizer method has to be replaced with jieba.cut for Chinese text after installing and importing jieba in the tokenize.py file.

# Reproducibility
## End to End reproduction
To reproduce the results listed in the paper, you can run:
```bash
bash run_script.sh dataset path-type use-context
```
The first argument "dataset" is the dataset from which the  OD (origin-destination) pairs will be extracted to perform the recommendation task. This argument can take on the value of beijing, chengdu or harbin
The second argument is the type of paths that the user would like to be recommended to. At this point in time, we support around 5 path types: most_used, fastest, shortest, touristic (scenic) and highway_free.
Finally, the argument "use-context" enables the use of PathGPT.

```bash
python inference.py -use-context -place_name beijing -path_type touristic -top_k 9
```
The above example recommends the most scenic paths for different OD pairs retrieved from the beijing dataset using PathGPT. Here, top_k represents the number of examples that will be shown to the LLM.

## Evaluation
For a simple evaluation on a specific dataset, run
```bash
python evaluate.py -use-context -place_name beijing -path_type highway_free -top_k 6
```
To get the evaluation results on the three used datasets, run 
```bash
python evaluate_all.py
```
which outputs the following information, including the precision and recall scores for the base LLM and PathGPT when the number of retrieved documents is set to 3, 6, and 9, respectively.
<!-- ```console
Overall performance of PathGPT with:
-llm : qwen2.5-14b
-embedding model : multilingual-e5-large-instruct
---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@3      LLM    |  PathGPT@3      
---------------------------------------------------------------------------------------------------------------------
                         touristic                beijing                  49.94  |  82.63          32.84  |  84.56          
                         touristic                chengdu                  30.14  |  88.34          29.14  |  92.46          
                         touristic                harbin                   46.63  |  84.36          25.65  |  86.85          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@3      LLM    |  PathGPT@3      
---------------------------------------------------------------------------------------------------------------------
                         highway_free             beijing                  54.72  |  86.71          31.63  |  86.95          
                         highway_free             chengdu                  35.29  |  89.74          26.06  |  92.78          
                         highway_free             harbin                   50.59  |  84.91          26.89  |  85.87          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@6      LLM    |  PathGPT@6      
---------------------------------------------------------------------------------------------------------------------
                         touristic                beijing                  49.94  |  84.19          32.84  |  84.13          
                         touristic                chengdu                  30.14  |  88.52          29.14  |  91.61          
                         touristic                harbin                   46.63  |  86.82          25.65  |  87.94          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@6      LLM    |  PathGPT@6      
---------------------------------------------------------------------------------------------------------------------
                         highway_free             beijing                  54.72  |  87.81          31.63  |  87.48          
                         highway_free             chengdu                  35.29  |  90.39          26.06  |  92.61          
                         highway_free             harbin                   50.59  |  86.44          26.89  |  86.56          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@9      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         touristic                beijing                  49.94  |  84.21          32.84  |  84.6           
                         touristic                chengdu                  30.14  |  89.85          29.14  |  92.91          
                         touristic                harbin                   46.63  |  88.32          25.65  |  88.29          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@9      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         highway_free             beijing                  54.72  |  88.29          31.63  |  88.38          
                         highway_free             chengdu                  35.29  |  90.79          26.06  |  92.81          
                         highway_free             harbin                   50.59  |  86.56          26.89  |  86.17          
     
``` -->



  



