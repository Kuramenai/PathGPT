# PathGPT : Leveraging Large Language Models for Personalized Route Generation

This repository contains the implementation code of the paper PathGPT : Leveraging Large Language Models for Personalized Route Generation.

# Introduction

The proliferation of GPS-enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By leveraging these data for training machine learning models, researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. In contrast to conventional algorithms such as Dijkstra’s shortest path algorithm, these novel algorithms possess the capacity to discern and learn patterns within the data, thereby facilitating the generation of more personalized paths. However, once these models have been trained, their application is constrained to 
the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of multiple machine learning models might be necessary to address new possible scenarios, which can be costly as each model must be trained separately. 

![context_generation](https://github.com/user-attachments/assets/e3c94bdb-c2ad-4eb8-b80a-ac830c29dd34)

Inspired by recent advances in the field of Large Language
Models (LLMs), we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training.
To accomplish this, we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context in
formation, similar to RAG (Retrieved Augmented Generation) systems, to enhance their ability to generate paths according to user-defined re quirements. 

![pathgpt_framework](https://github.com/user-attachments/assets/9160e97f-12ea-4905-b752-2ca7e0ed6519)

# Environment setup
For reference, all of our experiments were done on a server machine running Unbuntu 22.04 LTS with a NVIDIA RTX 4090.

## Local deployment of qwen2.5:14b-instruct
We use the 14b instruct version of qwen2.5 as LLM during our experiments, you can deploy it locally through Ollama.
- First install Ollama on your local machine
    * If you are on linux run:
      ```bash
      curl -fsSL https://ollama.com/install.sh | sh
      ```
    * If you are on windows or mac you can download it through [Ollama](https://ollama.com/download) and follow the official installation instructions
- Then download and deploy qwen2.5:14b-instruct by running:
    ```bash
   ollama run qwen2.5:14b-instruct
    ```
**N.B.**
- For optimal performance, we recommend to run the model on GPU
- The 14b version of qwen2.5 requires at least 10 GB of VRAM, so make sure you have enough VRAM capacity if you are running it on GPU.
- Sometimes, the Ollama service has to be launched manually, on Unbuntu 22.04 LTS, this can be accomplished by running:
  ```bash
  sudo systemctl enable ollama
  sudo systemctl start ollama
  ```
## Data - Credits to [NeuroMLR](https://github.com/idea-iitd/NeuroMLR)
Following instructions listed from [NeuroMLR](https://github.com/idea-iitd/NeuroMLR), 
download the [preprocessed data](https://drive.google.com/file/d/1bICE26ndR2C29jkfG2qQqVkmpirK25Eu/view?usp=sharing) and unzip the downloaded .zip file in the same directory as the other files.

Set the PREFIX_PATH variable in `varibales.py` through the `place_name` variable.

For each city (Beijing, Chengdu, Harbin), there are two types of data:

#### 1. Mapmatched pickled trajectories

Stored as a python pickled list of tuples, where each tuple is of the form (trip_id, trip, time_info). Here each trip is a list of edge identifiers.


#### 2. OSM map data
	
In the map folder, there are the following files-

1. `nodes.shp` : Contains OSM node information (global node id mapped to (latitude, longitude)) 
2. `edges.shp` : Contains network connectivity information (global edge id mapped to corresponding node ids)
3. `graph_with_haversine.pkl` : Pickled NetworkX graph corresponding to the OSM data
   
## Install dependencies
The dependencies can be installed by running:
```bash
python install -r requirements.txt
```
The recommended python version is 3.10. If you are using CUDA, we recommend CUDA 12.4.

# Reproducibility
## End to End reproduction
To reproduce the results listed in the paper you can run:
```bash
bash run_script.sh city path-type use-context
```
Here the first argument city is the dataset to generate paths for, in our case city can be beijing, chengdu or harbin
the second dataset is the type of paths to generate fastest or shortest
and the presence of use-context specifies to use the generated contexts from PathGPT.

```bash
bash run_script.sh beijing fastest use-context
```
The above example generates the fastest path from the beijing dataset using retreived contexts.

## Evaluation
For a simple evaluation on a specific dataset run
```bash
python evaluate.py -place_name beijing -path_type fastest -use-context
```
To get the evaluation results on the three used datasets, run 
```bash
python evaluate_all.py
```
which outputs the following information
```console
Overall performance of PathGPT with:
-llm : qwen2.5-14b-instruct
-embedding model : gte-Qwen2-1.5B-instruct
-no. retrieved documents : 3

---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  pathGPT        LLM    |  pathGPT
---------------------------------------------------------------------------------------------------------------------
                         fastest                  beijing                  37.11  |  52.29          26.44  |  51.3
                         fastest                  chengdu                  30.61  |  57.07          22.77  |  53.9
                         fastest                  harbin                   33.27  |  48.41          20.48  |  37.42



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  pathGPT        LLM    |  pathGPT
---------------------------------------------------------------------------------------------------------------------
                         shortest                 beijing                  34.47  |  46.4           26.15  |  48.86
                         shortest                 chengdu                  27.81  |  48.74          22.56  |  50.33
                         shortest                 harbin                   29.38  |  41.45          20.74  |  37.89
```



  
