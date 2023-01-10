# SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation

This repo contains code for [SpaBERT: A Pretrained Language Model from Geographic Data for Geo-Entity Representation](https://arxiv.org/abs/2210.12213). SpaBERT provides a general-purpose geo-entity representation based on neighboring entities in geospatial data. SpaBERT extends BERT to capture linearized spatial context, while incorporating a spatial coordinate embedding mechanism to preserve spatial relations of entities in the 2-dimensional space. SpaBERT is pretrained with masked language modeling and masked entity prediction tasks to learn spatial dependencies.

## Pretraining 
Pretrained model weights can be downloaded from the Google Drive for [SpaBERT-base](https://drive.google.com/file/d/1l44FY3DtDxzM_YVh3RR6PJwKnl80IYWB/view?usp=sharing) and [SpaBERT-large](https://drive.google.com/file/d/1LeZayTR92R5bu9gH_cGCwef7nnMX35cR/view?usp=share_link). 

Weights can also obtained from training from scratch using the following sample code. Data for pretraining can be downloaded [here](https://drive.google.com/drive/folders/1eaeVvUCcJVcNwnyTCk-1N1IKfukihk4j?usp=share_link). 

* Code to pretrain SpaBERT-base model:

  ```python3 train_mlm.py --lr=5e-5 --sep_between_neighbors --bert_option='bert-base'```

* Code to pretrain SpaBERT-large model:

  ```python3 train_mlm.py --lr=1e-6 --sep_between_neighbors --bert_option='bert-large```
  
## Downstream Tasks
### Supervised Geo-entity typing 
The goal is to predict a geo-entity’s semantic type (e.g., transportation and healthcare) given the target geo-entity name and spatial context (i.e. surrounding neighbors name and location). 

Models trained on OSM in London and California region can be downloaded from Google Drive for [SpaBERT-base](https://drive.google.com/file/d/1XFcA3sxC4wTlt7VjvMp1zNrWY5rjafzE/view?usp=share_link) and [SpaBERT-large](https://drive.google.com/file/d/12_FDVeSYkl_HQ61JmuMU6cRjQdKNpgR_/view?usp=share_link)

Data used for training can be downloaded [here](https://drive.google.com/drive/folders/1uyvGdiJdu-Cym4dOKhQLIkKpfgHvfo01?usp=share_link)

* Sample code for training SpaBERT-base typing model

```
python3 train_cls_spatialbert.py --lr=5e-5 --sep_between_neighbors --bert_option='bert-base'  --with_type --mlm_checkpoint_path='mlm_mem_keeppos_ep0_iter06000_0.2936.pth' 
```

* Sample code for training SpaBERT-large typing model

```
python3 train_cls_spatialbert.py --lr=1e-6 --sep_between_neighbors --bert_option='bert-large'  --with_type --mlm_checkpoint_path='mlm_mem_keeppos_ep1_iter02000_0.4400.pth' --epochs=20
```

### Unsupervised Geo-entity Linking

Geo-entity linking is to link geo-entities from a geographic information system (GIS) oriented dataset to a knowledge base (KB). This task unsupervised thus does not require any further training. Pretrained models can be directly used for this task. 



