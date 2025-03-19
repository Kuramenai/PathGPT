# PathGPT : Leveraging Large Language Models for Personalized Route Generation

This repository contains the implementation code of the paper PathGPT : Leveraging Large Language Models for Personalized Route Generation.

**Abstract**

The proliferation of GPS-enabled devices has led to the accumulation of a substantial corpus of historical trajectory data. By lever
aging these data for training machine learning models, researchers have devised novel data-driven methodologies that address the personalized route recommendation (PRR) problem. 
In contrast to conventional algorithms such as Dijkstra’s shortest path algorithm, these novel algorithms possess the capacity to discern and learn patterns within the data,
thereby facilitating the generation of more personalized paths. However, once these models have been trained, their application is constrained to 
the generation of routes that align with their training patterns. This limitation renders them less adaptable to novel scenarios and the deployment of
multiple machine learning models might be necessary to address new possible scenarios, which can be costly as each model must be trained separately. Inspired by recent advances in the field of Large Language
Models (LLMs), we leveraged their natural language understanding capabilities to develop a unified model to solve the PRR problem while being seamlessly adaptable to new scenarios without additional training.
To accomplish this, we combined the extensive knowledge LLMs acquired during training with further access to external hand-crafted context in
formation, similar to RAG (Retrieved Augmented Generation) systems, to enhance their ability to generate paths according to user-defined re quirements. 
Extensive experiments on different datasets show a considerable uplift in LLM performance on the PRR problem.
