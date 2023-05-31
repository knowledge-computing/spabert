# Tutorials for Testing and Fine-Tuning SpaBERT

This repository provides two Jupyter Notebooks for testing entity linking (one of the downstream tasks of SpaBERT) and fine-tuning procedure to train on geo-entitites from other knowledge bases (e.g., [World Historical Gazetteer](https://whgazetteer.org/))

Link to SpaBERT's original GitHub repository [https://github.com/zekun-li/spabert](https://github.com/zekun-li/spabert)

Run pip install requirements.txt before starting the jupyter notebooks to ensure you have all required packages

## Description of Jupyter Notebooks

### [spabert-fine-tuning.ipynb](https://github.com/Jina-Kim/spabert-tutorials/blob/main/spabert-fine-tuning.ipynb)
This Jupyter Notebook guides users on how to fine-tune spabert using point data from OpenStreetMap (OSM) in New York or Minnesota depending on the users choice. SpaBERT is pre-trained using data from California and London using OSM Point data. Instructions for pre-training your own model can be found on the spabert github
Here are the steps to run:

1. Define which dataset you want to use (New York or Minnesota)
2. Read data from csv file and construct KDTree for computing nearest neighbors
3. Create dataset using KDTree for fine-tuning SpaBERT using the dataset you chose
4. Load pre-trained model
5. Load dataset using the SpaBERT data loader
6. Train model for 1 epoch using fine-tuning model and save

### [spabert-entity-linking.ipynb](https://github.com/Jina-Kim/spabert-tutorials/blob/main/spabert-entity-linking.ipynb)
This Jupyter Notebook guides users on how to create an entity-linking dataset and how to perform entity-linking using SpaBERT. The dataset used here is a pre-matched dataset between World Historical Gazetteer (WHG) and Wikidata. The methods used to evaluate this model will be Hits@K and Mean Reciprocal Rank (MRR)
Here are the steps to run:

1. Load fine-tuned model from previous Jupyter notebook
2. Load datasets using the WHG data loader
3. Calculate embeddings for whg and wikidata entities using SpaBERT
4. Calculate hits@1, Hits@5, Hits@10, and MRR 
