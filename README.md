# PathGPT: Leveraging Large Language Models for Personalized Route Generation

This repository contains the implementation code of the paper PathGPT: Leveraging Large Language Models for Personalized Route Generation.

# Introduction

The proliferation of GPS-enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By leveraging these data for training machine learning models, researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. In contrast to conventional algorithms such as Dijkstra’s shortest path algorithm, these novel algorithms possess the capacity to discern and learn patterns within the data, thereby facilitating the generation of more personalized paths. However, once these models have been trained, their application is constrained to 
the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of multiple machine learning models might be necessary to address new possible scenarios, which can be costly as each model must be trained separately. 

![context_generation](https://github.com/user-attachments/assets/e3c94bdb-c2ad-4eb8-b80a-ac830c29dd34)

Inspired by recent advances in the field of Large Language
Models (LLMs), we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training.
To accomplish this, we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context in
formation, similar to RAG (Retrieved Augmented Generation) systems, to enhance their ability to generate paths according to user-defined requirements. 

![pathgpt_framework](https://github.com/user-attachments/assets/9160e97f-12ea-4905-b752-2ca7e0ed6519)

# Environment setup
For reference, all of our experiments were done on a server machine running Ubuntu 22.04 LTS with a NVIDIA RTX 4090.

## Local deployment of qwen2.5:14b-instruct
We use the 14b instruct version of qwen2.5 as LLM during our experiments. You can deploy it locally through Ollama.
- First, install Ollama on your local machine
    * If you are on Linux, run:
      ```bash
      curl -fsSL https://ollama.com/install.sh | sh
      ```
    * If you are on Windows or macOS, you can download it through [Ollama](https://ollama.com/download) and follow the official installation instructions
- Then download and deploy qwen2.5:14b-instruct by running:
    ```bash
   ollama run qwen2.5:14b-instruct
    ```
**N.B.**
- For optimal performance, we recommend running the model on a GPU
- The 14b version of qwen2.5 requires at least 10 GB of VRAM, so make sure you have enough VRAM capacity if you are running it on a GPU.
- Sometimes, the Ollama service has to be launched manually, on Ubuntu 22.04 LTS. This can be accomplished by running:
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

# Reproducibility
## End to End reproduction
To reproduce the results listed in the paper, you can run:
```bash
bash run_script.sh city path-type use-context
```
Here, the first argument, city, is the dataset to generate paths for. In our case city can be beijing, chengdu or harbin
The second dataset is the type of paths to generate the fastest or shortest
and the presence of use-context specifies to use the generated contexts from PathGPT.

```bash
bash run_script.sh beijing fastest use-context
```
The above example generates the fastest path from the Beijing dataset using retrieved contexts.

## Evaluation
For a simple evaluation on a specific dataset run
```bash
python evaluate.py -place_name beijing -path_type fastest -use-context
```
To get the evaluation results on the three used datasets, run 
```bash
python evaluate_all.py
```
which outputs the following information, including the precision and recall scores for the base LLM and PathGPT when the number of retrieved documents is set to 3, 6, and 9, respectively.
```console
Overall performance of PathGPT with:
-llm : qwen2.5-14b
-embedding model: multilingual-e5-large-instruct
-no. retrieved documents : 3



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@3      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         fastest                  beijing                  56.06  |  87.01          39.5  |  88.71           
                         fastest                  chengdu                  38.29  |  92.58          33.07  |  95.41          
                         fastest                  harbin                   45.99  |  86.45          29.72  |  89.08          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@3      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         most_used                beijing                  12.67  |  85.62          8.39  |  87.56           
                         most_used                chengdu                  37.49  |  83.66          32.76  |  86.63          
                         most_used                harbin                   48.16  |  77.22          30.92  |  84.52          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@6      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         fastest                  beijing                  56.06  |  87.01          39.5  |  88.71           
                         fastest                  chengdu                  38.29  |  92.58          33.07  |  95.41          
                         fastest                  harbin                   45.99  |  86.45          29.72  |  89.08          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@6      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         most_used                beijing                  12.67  |  85.62          8.39  |  87.56           
                         most_used                chengdu                  37.49  |  83.66          32.76  |  86.63          
                         most_used                harbin                   48.16  |  77.22          30.92  |  84.52          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@9      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         fastest                  beijing                  56.06  |  87.01          39.5  |  88.71           
                         fastest                  chengdu                  38.29  |  92.58          33.07  |  95.41          
                         fastest                  harbin                   45.99  |  86.45          29.72  |  89.08          



---------------------------------------------------------------------------------------------------------------------
                                                                              Precision                Recall                
---------------------------------------------------------------------------------------------------------------------
                         path_type                city                     LLM    |  PathGPT@9      LLM    |  PathGPT@9      
---------------------------------------------------------------------------------------------------------------------
                         most_used                beijing                  12.67  |  85.62          8.39  |  87.56           
                         most_used                chengdu                  37.49  |  83.66          32.76  |  86.63          
                         most_used                harbin                   48.16  |  77.22          30.92  |  84.52          
```



  
