import numpy as np
import torch
from torch.utils.data import Dataset
import json
import sys
sys.path.append("../")
from datasets.dataset_loader import SpatialDataset
from transformers import RobertaTokenizer, BertTokenizer

class WHGDataset(SpatialDataset):
    # initializes dataset loader and converts dataset python object
    def __init__(self, data_file_path, tokenizer=None,max_token_len = 512, distance_norm_factor = 1, spatial_dist_fill=100, sep_between_neighbors = False):
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        self.read_data(data_file_path)
        self.max_token_len = max_token_len
        self.distance_norm_factor = distance_norm_factor
        self.spatial_dist_fill = spatial_dist_fill
        self.sep_between_neighbors = sep_between_neighbors

    # returns a specific item from the dataset given an index
    def __getitem__(self, idx):
        return self.load_data(idx)
    
    # returns the length of the dataset loaded
    def __len__(self):
        return self.len_data

    def get_average_distance(self,idx):
        line = self.data[idx]
        line_data_dict = json.loads(line)
        pivot_pos = line_data_dict['info']['geometry']['coordinates']

        neighbor_geom_list = line_data_dict['neighbor_info']['geometry_list']
        lat_diff = 0
        lng_diff = 0
        for neighbor in neighbor_geom_list:
            coordinates = neighbor['coordinates']
            lat_diff = lat_diff + (abs(pivot_pos[0]-coordinates[0]))
            lng_diff = lng_diff + (abs(pivot_pos[1]-coordinates[1]))
        avg_lat_diff = lat_diff/len(neighbor_geom_list)
        avg_lng_diff = lng_diff/len(neighbor_geom_list)
        return (avg_lat_diff, avg_lng_diff)
            
        
    # reads dataset from given filepath, run on initilization
    def read_data(self, data_file_path):
        with open(data_file_path, 'r') as f:
            data = f.readlines()

        len_data = len(data)
        self.len_data = len_data
        self.data = data

    # loads and parses dataset
    def load_data(self, idx):
        line = self.data[idx]
        line_data_dict = json.loads(line)

        # get pivot info
        pivot_name = str(line_data_dict['info']['name'])
        pivot_pos = line_data_dict['info']['geometry']['coordinates']

        # get neighbor info
        neighbor_info = line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_geom_list = neighbor_info['geometry_list']

       
            
        parsed_data = self.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geom_list, self.spatial_dist_fill)
        parsed_data['qid'] = line_data_dict['info']['qid']

        return parsed_data

