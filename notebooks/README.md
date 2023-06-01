# Tutorials for Testing and Fine-Tuning SpaBERT

This repository provides two Jupyter Notebooks for testing entity linking (one of the downstream tasks of SpaBERT) and fine-tuning procedure to train on geo-entitites from other knowledge bases (e.g., [World Historical Gazetteer](https://whgazetteer.org/))

Link to SpaBERT's original GitHub repository [https://github.com/zekun-li/spabert](https://github.com/zekun-li/spabert)

The first step is cloning the spabert repo onto your machine. Run the following line of code to do this.

`git clone [this-repo]`

You will need to have IPython Kernel for Jupyter installed before running the code in this tutorial. Run the following line of code to ensure ipython is installed

`pip install ipykernel`

Before starting the jupyter notebooks run the following lines to make sure you have all required packages:

`pip install requirements.txt`

The requirements.txt file will be located in the spabert directory.
```
-spabert
 | - datasets
 | - experiments
 | - models
 | - models
 | - notebooks
 | - utils
 | - __init__.py
 | - README.md
 | - requirements.txt
 | - train_mlm.py
```

## Installing Model Weights

To install pre-trained & fine-tuned model weights run the following commands:

Make sure you have git-lfs installed (https://git-lfs.com windows & mac) (https://github.com/git-lfs/git-lfs/blob/main/INSTALLING.md linux)

run the follow commands separately in the order they appear

`git lfs install`
`git clone https://huggingface.co/knowledge-computing-lab/spabert-base-uncased`
`git clone https://huggingface.co/knowledge-computing-lab/spabert-base-uncased-finetuned-osm-mn`

Once the model weight is installed you'll see a file called `mlm_mem_keeppos_ep0_iter06000_0.2936.pth` and `spabert-base-uncased-finetuned-osm-mn.pth`
Move these files to the tutorial_datasets folder. After moving them you file structure should look like this
```
- notebooks
  | - tutorial_datasets
  |   | - mlm_mem_keeppos_ep0_iter06000_0.2936.pth
  |   | - osm_mn.csv
  |   | - SPABERT_finetuning_data.json
  |   | - spabert_whg_wikidata.json
  |   | - spabert_wikidata_sampled.json
  |   | - spabert-base-uncased-finetuned-osm-mn.pth
  | - README.md
  | - spabert-entity-linking.ipynb
  | - spabert-fine-tuning.ipynb
  | - WHGDataset.py
```
## Jupyter Notebook Descriptions

### [spabert-fine-tuning.ipynb](https://github.com/Jina-Kim/spabert/blob/main/notebooks/spabert-fine-tuning.ipynb)
This Jupyter Notebook provides on how to fine-tune spabert using point data from OpenStreetMap (OSM) in Minnesota. SpaBERT is pre-trained using data from California and London using OSM Point data. Instructions for pre-training your own model can be found on the spabert github
Here are the steps to run:

1. Define which dataset you want to use (e.g., OSM in New York or Minnesota)
2. Read data from csv file and construct KDTree for computing nearest neighbors
3. Create dataset using KDTree for fine-tuning SpaBERT using the dataset you chose
4. Load pre-trained model
5. Load dataset using the SpaBERT data loader
6. Train model for 1 epoch using fine-tuning model and save

### [spabert-entity-linking.ipynb](https://github.com/Jina-Kim/spabert/blob/main/notebooks/spabert-entity-linking.ipynb)
This Jupyter Notebook provides on how to create an entity-linking dataset and how to perform entity-linking using SpaBERT. The dataset used here is a pre-matched dataset between World Historical Gazetteer (WHG) and Wikidata. The methods used to evaluate this model will be Hits@K and Mean Reciprocal Rank (MRR)
Here are the steps to run:

1. Load fine-tuned model from previous Jupyter notebook
2. Load datasets using the WHG data loader
3. Calculate embeddings for whg and wikidata entities using SpaBERT
4. Calculate hits@1, Hits@5, Hits@10, and MRR 

## Dataset Descriptions

There are two types of tutorial datasets used for fine-tuning SpaBERT, CSV and JSON files.

- CSV file - sample taken from OpenStreetMap (OSM)
    - Minnesota State `./tutorial_datasets/osm_mn.csv`

    An example data structure:
  
    | row_id | name | latitude | longitude |
    | ------ | ---- | -------- | --------- |
    |    0   | Duluth | -92.1215 | 46.7729 |
    |    1   | Green Valley | -95.757 | 44.5269 | 

- JSON files - ready-to-use files for SpaBERT's data loader - [SpatialDataset](../datasets/dataset_loader.py)
    - OSM Minnesota State `./tutorial_datasets/spabert_osm_mn.json`
      - Generated from `./tutorial_datasets/osm_mn.csv` using spabert-fine-tuning.ipynb
    - WHG `./tutorial_datasets/spabert_whg_wikidata.json`
      - Geo-entities from WHG that have the link to Wikidata
    - Wikidata `./tutorial_datasets/spabert_wikidata_sampled.json`
      - Sampled from entities delivered by WHG. These entities have been linked between WHG and Wikidata by WHG prior to being delivered to us.
    
    
The file contains json objects on each line. Each json object describes the spatial context of an entity using nearby entities.

A sample json object looks like the following:
    
```json
{
   "info":{
      "name":"Duluth",
      "geometry":{
         "coordinates":[
            46.7729,
            -92.1215
         ]
      }
   },
   "neighbor_info":{
      "name_list":[
         "Duluth",
         "Chinese Peace Belle and Garden",
         ...
      ],
      "geometry_list":[
         {
            "coordinates":[
               46.7729,
               -92.1215
            ]
         },
         {
            "coordinates":[
               46.7770,
               -92.1241
            ]
         },
         ...
      ]
   }
}
```

To perform entity-linking on SpaBERT you must have a dataset structured similarly to the second dataset used for fine-tuning. 

A sample json object looks like the following: 


```json
{
   "info":{
      "name":"Duluth",
      "geometry":{
         "coordinates":[
            46.7729,
            -92.1215
         ]
      },
      "qid":"Q485708"
   },
   "neighbor_info":{
      ...
   }
}
```