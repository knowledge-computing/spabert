#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pdb
import json
import scipy.spatial as sp
import argparse


import torch
from torch.utils.data import DataLoader

from transformers import AdamW
from transformers import BertTokenizer
from tqdm import tqdm  # for our progress bar

sys.path.append('../../../')
from datasets.usgs_os_sample_loader import USGS_MapDataset
from datasets.wikidata_sample_loader import Wikidata_Geocoord_Dataset, Wikidata_Random_Dataset
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from utils.find_closest import find_ref_closest_match, sort_ref_closest_match
from utils.common_utils import load_spatial_bert_pretrained_weights, get_spatialbert_embedding, get_bert_embedding, write_to_csv
from utils.baseline_utils import get_baseline_model

from transformers import BertModel

sys.path.append('/home/zekun/spatial_bert/spatial_bert/datasets')
from dataset_loader import SpatialDataset
from osm_sample_loader import PbfMapDataset



MODEL_OPTIONS = ['spatial_bert-base','spatial_bert-large', 'bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large','luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large']


CANDSET_MODES = ['all_map'] # candidate set is constructed based on all maps or one map

def recall_at_k(rank_list, k = 1):
    
    total_query = len(rank_list)
    recall = np.sum(np.array(rank_list)<=k)
    recall = 1.0 * recall / total_query
    
    return recall

def reciprocal_rank(all_rank_list):
    
    recip_list = [1./rank for rank in all_rank_list]
    mean_recip = np.mean(recip_list)
    
    return mean_recip, recip_list

def link_to_itself(source_embedding_ogc_list, target_embedding_ogc_list):

    source_emb_list = [source_dict['emb'] for source_dict in source_embedding_ogc_list]
    source_ogc_list = [source_dict['ogc_fid'] for source_dict in source_embedding_ogc_list]
    
    target_emb_list = [target_dict['emb'] for target_dict in target_embedding_ogc_list]
    target_ogc_list = [target_dict['ogc_fid'] for target_dict in target_embedding_ogc_list]

    rank_list = []
    for source_emb, source_ogc in zip(source_emb_list, source_ogc_list):
        sim_matrix = 1 - sp.distance.cdist(np.array(target_emb_list), np.array([source_emb]), 'cosine')
        closest_match_ogc = sort_ref_closest_match(sim_matrix, target_ogc_list)

        closest_match_ogc = [a[0] for a in closest_match_ogc]
        rank = closest_match_ogc.index(source_ogc) +1
        rank_list.append(rank)

    
    mean_recip, recip_list = reciprocal_rank(rank_list)
    r1 = recall_at_k(rank_list, k = 1)
    r5 = recall_at_k(rank_list, k = 5)
    r10 = recall_at_k(rank_list, k = 10)
    
    return mean_recip , r1, r5, r10

def get_embedding_and_ogc(dataset, model_name, model):
    dict_list = []

    for source in dataset:
        if model_name == 'spatial_bert-base' or model_name ==  'spatial_bert-large':
            source_emb = get_spatialbert_embedding(source, model)
        else:
            source_emb = get_bert_embedding(source, model)
            
        source_dict = {}
        source_dict['emb'] = source_emb
        source_dict['ogc_fid'] = source['ogc_fid']
        #wikidata_dict['wikidata_des_list'] = [wikidata_cand['description']]

        dict_list.append(source_dict)

    return dict_list


def entity_linking_func(args):

    model_name = args.model_name
    candset_mode = args.candset_mode

    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill= args.spatial_dist_fill
    sep_between_neighbors = args.sep_between_neighbors

    spatial_bert_weight_dir = args.spatial_bert_weight_dir
    spatial_bert_weight_name = args.spatial_bert_weight_name

    if_no_spatial_distance = args.no_spatial_distance
    random_remove_neighbor = args.random_remove_neighbor


    assert model_name in MODEL_OPTIONS
    assert candset_mode in CANDSET_MODES


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    
        
    if model_name == 'spatial_bert-base':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        config = SpatialBertConfig()
        model = SpatialBertModel(config)

        model.to(device)
        model.eval()
        
        # load pretrained weights
        weight_path = os.path.join(spatial_bert_weight_dir, spatial_bert_weight_name)
        model = load_spatial_bert_pretrained_weights(model, weight_path)

    elif model_name == 'spatial_bert-large':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        
        config = SpatialBertConfig(hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24)
        model = SpatialBertModel(config)

        model.to(device)
        model.eval()
        
        # load pretrained weights
        weight_path = os.path.join(spatial_bert_weight_dir, spatial_bert_weight_name)
        model = load_spatial_bert_pretrained_weights(model, weight_path)

    else:
        model, tokenizer = get_baseline_model(model_name)
        model.to(device)
        model.eval()

    source_file_path = '../data/osm-point-minnesota-full.json'
    source_dataset = PbfMapDataset(data_file_path = source_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = 512, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = False,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        mode = None,
                                        random_remove_neighbor = random_remove_neighbor,
                                        )

    target_dataset = PbfMapDataset(data_file_path = source_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = 512, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = False,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        mode = None,
                                        random_remove_neighbor = 0., # keep all
                                        )

    # process candidates for each phrase
    

    source_embedding_ogc_list = get_embedding_and_ogc(source_dataset, model_name, model)
    target_embedding_ogc_list = get_embedding_and_ogc(target_dataset, model_name, model)

        
    mean_recip , r1, r5, r10 = link_to_itself(source_embedding_ogc_list, target_embedding_ogc_list)
    print('\n')
    print(random_remove_neighbor, mean_recip , r1, r5, r10)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='spatial_bert-base')
    parser.add_argument('--candset_mode', type=str, default='all_map')

    parser.add_argument('--distance_norm_factor', type=float, default = 0.0001)
    parser.add_argument('--spatial_dist_fill', type=float, default = 20)
                       
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--spatial_bert_weight_dir', type = str, default = None)
    parser.add_argument('--spatial_bert_weight_name', type = str, default = None)

    parser.add_argument('--random_remove_neighbor', type = float, default = 0.)
    
                        
    args = parser.parse_args()
    # print('\n')
    # print(args)
    # print('\n')

    entity_linking_func(args)

    # CUDA_VISIBLE_DEVICES='1' python3 linking_ablation.py  --sep_between_neighbors --model_name='spatial_bert-base'  --spatial_bert_weight_dir='/data/zekun/spatial_bert_weights/typing_lr5e-05_sep_bert-base_nofreeze_london_california_bsize12/ep0_iter06000_0.2936/' --spatial_bert_weight_name='keeppos_ep0_iter02000_0.4879.pth' --random_remove_neighbor=0.1


    # CUDA_VISIBLE_DEVICES='1' python3 linking_ablation.py  --sep_between_neighbors --model_name='spatial_bert-large'  --spatial_bert_weight_dir='/data/zekun/spatial_bert_weights/typing_lr1e-06_sep_bert-large_nofreeze_london_california_bsize12/ep2_iter02000_0.3921/' --spatial_bert_weight_name='keeppos_ep8_iter03568_0.2661_val0.2284.pth' --random_remove_neighbor=0.1


if __name__ == '__main__':

    main()

    
