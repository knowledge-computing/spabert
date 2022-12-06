import sys
import numpy as np
import json 
import math

import torch
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import Dataset
sys.path.append('/home/zekun/spatial_bert/spatial_bert/datasets')
from dataset_loader import SpatialDataset

import pdb

'''Prepare candiate list given randomly sampled data and append to data_list'''
class Wikidata_Random_Dataset(SpatialDataset): 
    def __init__(self, data_file_path,  tokenizer=None, max_token_len = 512, distance_norm_factor = 0.0001, spatial_dist_fill=100, sep_between_neighbors = False):
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        self.max_token_len = max_token_len
        self.spatial_dist_fill = spatial_dist_fill # should be normalized distance fill, larger than all normalized neighbor distance
        self.sep_between_neighbors = sep_between_neighbors
        self.read_file(data_file_path)

        
        super(Wikidata_Random_Dataset, self).__init__(tokenizer , max_token_len , distance_norm_factor, sep_between_neighbors )

    def read_file(self, data_file_path):

        with open(data_file_path, 'r') as f:
            data = f.readlines()

        len_data = len(data)
        self.len_data = len_data
        self.data = data 

    def load_data(self, index):
        
        spatial_dist_fill = self.spatial_dist_fill
        line = self.data[index] # take one line from the input data according to the index

        line_data_dict = json.loads(line)

        # process pivot
        pivot_name = line_data_dict['info']['name']
        pivot_pos = line_data_dict['info']['geometry']['coordinates']
        pivot_uri = line_data_dict['info']['uri']

        neighbor_info = line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_geometry_list = neighbor_info['geometry_list']

        parsed_data = self.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill )
        parsed_data['uri'] = pivot_uri
        parsed_data['description'] = None # placeholder

        return parsed_data

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)



'''Prepare candiate list for each phrase and append to data_list'''

class Wikidata_Geocoord_Dataset(SpatialDataset): 

    #DEFAULT_CONFIG_CLS = SpatialBertConfig
    def __init__(self, data_file_path,  tokenizer=None, max_token_len = 512, distance_norm_factor = 0.0001, spatial_dist_fill=100, sep_between_neighbors = False):
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        self.max_token_len = max_token_len
        self.spatial_dist_fill = spatial_dist_fill # should be normalized distance fill, larger than all normalized neighbor distance
        self.sep_between_neighbors = sep_between_neighbors
        self.read_file(data_file_path)

        
        super(Wikidata_Geocoord_Dataset, self).__init__(tokenizer , max_token_len , distance_norm_factor, sep_between_neighbors )

    def read_file(self, data_file_path):

        with open(data_file_path, 'r') as f:
            data = f.readlines()

        len_data = len(data)
        self.len_data = len_data
        self.data = data 

    def load_data(self, index):
        
        spatial_dist_fill = self.spatial_dist_fill
        line = self.data[index] # take one line from the input data according to the index

        line_data = json.loads(line)
        parsed_data_list = []

        for line_data_dict in line_data:
            # process pivot
            pivot_name = line_data_dict['info']['name']
            pivot_pos = line_data_dict['info']['geometry']['coordinates']
            pivot_uri = line_data_dict['info']['uri']

            neighbor_info = line_data_dict['neighbor_info']
            neighbor_name_list = neighbor_info['name_list']
            neighbor_geometry_list = neighbor_info['geometry_list']

            parsed_data = self.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill )
            parsed_data['uri'] = pivot_uri
            parsed_data['description'] = None # placeholder
            parsed_data_list.append(parsed_data)

        return parsed_data_list


    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)
