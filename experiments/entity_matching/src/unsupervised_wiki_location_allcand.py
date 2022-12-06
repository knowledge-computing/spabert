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
#from transformers.models.bert.modeling_bert import BertForMaskedLM

from transformers import AdamW
from transformers import BertTokenizer
from tqdm import tqdm  # for our progress bar

sys.path.append('../../../')
from datasets.usgs_os_sample_loader import USGS_MapDataset
from datasets.wikidata_sample_loader import Wikidata_Geocoord_Dataset, Wikidata_Random_Dataset
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
#from models.spatial_bert_model import  SpatialBertForMaskedLM
from utils.find_closest import find_ref_closest_match, sort_ref_closest_match
from utils.common_utils import load_spatial_bert_pretrained_weights, get_spatialbert_embedding, get_bert_embedding, write_to_csv
from utils.baseline_utils import get_baseline_model

from transformers import BertModel


MODEL_OPTIONS = ['spatial_bert-base','spatial_bert-large', 'bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large','luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large']

MAP_TYPES = ['usgs']
CANDSET_MODES = ['all_map'] # candidate set is constructed based on all maps or one map


def disambiguify(model, model_name, usgs_dataset, wikidata_dict_list, candset_mode = 'all_map', if_use_distance = True, select_indices = None): 

    if select_indices is None: 
        select_indices = range(0, len(usgs_dataset))


    assert(candset_mode in ['all_map','per_map'])

    wikidata_emb_list = [wikidata_dict['wikidata_emb_list'] for wikidata_dict in wikidata_dict_list]
    wikidata_uri_list = [wikidata_dict['wikidata_uri_list'] for wikidata_dict in wikidata_dict_list]
    #wikidata_des_list = [wikidata_dict['wikidata_des_list'] for wikidata_dict in wikidata_dict_list]

    if candset_mode == 'all_map':
        wikidata_emb_list = [item for sublist in wikidata_emb_list for item in sublist] # flatten
        wikidata_uri_list = [item for sublist in wikidata_uri_list for item in sublist] # flatten
        #wikidata_des_list = [item for sublist in wikidata_des_list for item in sublist] # flatten

    
    ret_list = []
    for i in select_indices:

        if candset_mode == 'per_map':
            usgs_entity = usgs_dataset[i]
            wikidata_emb_list = wikidata_emb_list[i]
            wikidata_uri_list = wikidata_uri_list[i]
            #wikidata_des_list = wikidata_des_list[i]

        elif candset_mode == 'all_map':
            usgs_entity = usgs_dataset[i]
        else:
            raise NotImplementedError
        
        if model_name == 'spatial_bert-base' or model_name == 'spatial_bert-large':
            usgs_emb = get_spatialbert_embedding(usgs_entity, model, use_distance = if_use_distance)
        else:
            usgs_emb = get_bert_embedding(usgs_entity, model)

      
        sim_matrix = 1 - sp.distance.cdist(np.array(wikidata_emb_list), np.array([usgs_emb]), 'cosine')
        
        closest_match_uri = sort_ref_closest_match(sim_matrix, wikidata_uri_list)
        #closest_match_des = sort_ref_closest_match(sim_matrix, wikidata_des_list)

            
        sorted_sim_matrix = np.sort(sim_matrix, axis = 0)[::-1] # descending order

        ret_dict = dict()
        ret_dict['pivot_name'] = usgs_entity['pivot_name']
        ret_dict['sorted_match_uri'] = [a[0] for a in closest_match_uri]
        #ret_dict['sorted_match_des'] = [a[0] for a in closest_match_des]
        ret_dict['sorted_sim_matrix'] = [a[0] for a in sorted_sim_matrix]

        ret_list.append(ret_dict)

    return ret_list 

def entity_linking_func(args):

    model_name = args.model_name
    map_type = args.map_type 
    candset_mode = args.candset_mode

    usgs_distance_norm_factor = args.usgs_distance_norm_factor
    spatial_dist_fill= args.spatial_dist_fill
    sep_between_neighbors = args.sep_between_neighbors

    spatial_bert_weight_dir = args.spatial_bert_weight_dir
    spatial_bert_weight_name = args.spatial_bert_weight_name

    if_no_spatial_distance = args.no_spatial_distance


    assert model_name in MODEL_OPTIONS
    assert map_type in MAP_TYPES
    assert candset_mode in CANDSET_MODES


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.out_dir is None:

        if model_name == 'spatial_bert-base' or model_name == 'spatial_bert-large':

            if sep_between_neighbors:
                spatialbert_output_dir_str = 'dnorm' + str(usgs_distance_norm_factor ) + '_distfill' + str(spatial_dist_fill) + '_sep' 
            else:
                spatialbert_output_dir_str = 'dnorm' + str(usgs_distance_norm_factor ) + '_distfill' + str(spatial_dist_fill) + '_nosep' 


            checkpoint_ep = spatial_bert_weight_name.split('_')[3]
            checkpoint_iter = spatial_bert_weight_name.split('_')[4]
            loss_val = spatial_bert_weight_name.split('_')[5][:-4]

            if if_no_spatial_distance:
                linking_prediction_dir = 'linking_prediction_dir/abalation_no_distance/'
            else:
                linking_prediction_dir = 'linking_prediction_dir'

            if model_name == 'spatial_bert-base':
                out_dir  = os.path.join('/data2/zekun/', linking_prediction_dir,  spatialbert_output_dir_str) + '/' + map_type + '-' + model_name + '-' + checkpoint_ep + '-' + checkpoint_iter + '-' + loss_val
            elif model_name == 'spatial_bert-large':
                
                freeze_str = spatial_bert_weight_dir.split('/')[-2].split('_')[1] # either 'freeze' or 'nofreeze'
                out_dir  = os.path.join('/data2/zekun/', linking_prediction_dir,  spatialbert_output_dir_str) + '/' + map_type + '-' + model_name + '-' + checkpoint_ep + '-' + checkpoint_iter + '-' + loss_val + '-' + freeze_str

            

        else:
            out_dir  = '/data2/zekun/baseline_linking_prediction_dir/'  + map_type + '-' + model_name 

    else:
        out_dir = args.out_dir

    print('out_dir', out_dir)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    
        
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


    if map_type == 'usgs':
        map_name_list = ['USGS-15-CA-brawley-e1957-s1957-p1961',
                        'USGS-15-CA-paloalto-e1899-s1895-rp1911',
                        'USGS-15-CA-capesanmartin-e1921-s1917',
                        'USGS-15-CA-sanfrancisco-e1899-s1892-rp1911',
                        'USGS-30-CA-dardanelles-e1898-s1891-rp1912',
                        'USGS-30-CA-holtville-e1907-s1905-rp1946',
                        'USGS-30-CA-indiospecial-e1904-s1901-rp1910',
                        'USGS-30-CA-lompoc-e1943-s1903-ap1941-rv1941',
                        'USGS-30-CA-sanpedro-e1943-rv1944',
                        'USGS-60-CA-alturas-e1892-rp1904',
                        'USGS-60-CA-amboy-e1942',
                        'USGS-60-CA-amboy-e1943-rv1943',
                        'USGS-60-CA-modoclavabed-e1886-s1884',
                        'USGS-60-CA-saltonsea-e1943-ap1940-rv1942']
        
    print('processing wikidata...')

    wikidata_dict_list = []

    wikidata_random30k = Wikidata_Random_Dataset(
            data_file_path = '../data_processing/wikidata_sample30k/wikidata_30k_neighbor_reformat.json',
            #neighbor_file_path = '../data_processing/wikidata_sample30k/wikidata_30k_neighbor.json',
            tokenizer = tokenizer,
            max_token_len = 512, 
            distance_norm_factor = 0.0001, 
            spatial_dist_fill=100, 
            sep_between_neighbors = sep_between_neighbors,
        )
    
    # process candidates for each phrase
    for wikidata_cand in wikidata_random30k:
        if model_name == 'spatial_bert-base' or model_name ==  'spatial_bert-large':
            wikidata_emb = get_spatialbert_embedding(wikidata_cand, model)
        else:
            wikidata_emb = get_bert_embedding(wikidata_cand, model)
            
        wikidata_dict = {}
        wikidata_dict['wikidata_emb_list'] = [wikidata_emb]
        wikidata_dict['wikidata_uri_list'] = [wikidata_cand['uri']]

        wikidata_dict_list.append(wikidata_dict)



    for map_name in map_name_list:
        
        print(map_name)

        wikidata_dict_per_map = {}
        wikidata_dict_per_map['wikidata_emb_list'] = []
        wikidata_dict_per_map['wikidata_uri_list'] = []

        wikidata_dataset_permap =  Wikidata_Geocoord_Dataset(
            data_file_path = '../data_processing/outputs/wikidata_reformat/wikidata_' + map_name + '.json',
            tokenizer = tokenizer,
            max_token_len = 512, 
            distance_norm_factor = 0.0001, 
            spatial_dist_fill=100,
            sep_between_neighbors = sep_between_neighbors)



        for i in range(0, len(wikidata_dataset_permap)):
            # get all candiates for phrases within the map
            wikidata_candidates = wikidata_dataset_permap[i] # dataset for each map, list of [cand for each phrase] 


            # process candidates for each phrase
            for wikidata_cand in wikidata_candidates:
                if model_name == 'spatial_bert-base' or model_name ==  'spatial_bert-large':
                    wikidata_emb = get_spatialbert_embedding(wikidata_cand, model)
                else:
                    wikidata_emb = get_bert_embedding(wikidata_cand, model)

                wikidata_dict_per_map['wikidata_emb_list'].append(wikidata_emb)
                wikidata_dict_per_map['wikidata_uri_list'].append(wikidata_cand['uri'])

            
        wikidata_dict_list.append(wikidata_dict_per_map)
        


    for map_name in map_name_list:
        
        print(map_name)
        

        usgs_dataset =  USGS_MapDataset(
            data_file_path = '../data_processing/outputs/alignment_dir/map_' + map_name + '.json',
            tokenizer = tokenizer,
            distance_norm_factor = usgs_distance_norm_factor,
            spatial_dist_fill = spatial_dist_fill,
            sep_between_neighbors = sep_between_neighbors)
        
        
        ret_list = disambiguify(model, model_name, usgs_dataset, wikidata_dict_list, candset_mode= candset_mode, if_use_distance = not if_no_spatial_distance, select_indices = None)

        write_to_csv(out_dir, map_name, ret_list)

    print('Done')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='spatial_bert-base')
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--map_type', type=str, default='usgs')
    parser.add_argument('--candset_mode', type=str, default='all_map')

    parser.add_argument('--usgs_distance_norm_factor', type=float, default = 1)
    parser.add_argument('--spatial_dist_fill', type=float, default = 100)
                       
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--spatial_bert_weight_dir', type = str, default = None)
    parser.add_argument('--spatial_bert_weight_name', type = str, default = None)
                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    # out_dir not None, and out_dir does not exist, then create out_dir
    if args.out_dir is not None and not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    entity_linking_func(args)



if __name__ == '__main__':

    main()

    
