import os
import sys
from tqdm import tqdm  # for our progress bar
import numpy as np
import argparse 
from sklearn.preprocessing import LabelEncoder
import pdb


import torch
from torch.utils.data import DataLoader
from transformers import AdamW
import torch.nn.functional as F

sys.path.append('../../../')
from datasets.osm_sample_loader import PbfMapDataset
from datasets.const import *
from utils.baseline_utils import get_baseline_model
from models.baseline_typing_model import BaselineForSemanticTyping

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import precision_recall_fscore_support

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)



MODEL_OPTIONS = ['bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large','luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large']

def testing(args):

    num_workers = args.num_workers
    batch_size = args.batch_size
    max_token_len = args.max_token_len

    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill
    with_type = args.with_type
    sep_between_neighbors = args.sep_between_neighbors
    freeze_backbone = args.freeze_backbone


    backbone_option = args.backbone_option

    checkpoint_path = args.checkpoint_path

    assert(backbone_option in MODEL_OPTIONS)
    
   
    london_file_path = '../data/sql_output/osm-point-london-typing.json'
    california_file_path = '../data/sql_output/osm-point-california-typing.json'


    
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_9_LIST)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    backbone_model, tokenizer = get_baseline_model(backbone_option)
    model = BaselineForSemanticTyping(backbone_model, backbone_model.config.hidden_size, len(CLASS_9_LIST))

    model.load_state_dict(torch.load(checkpoint_path) ) #, strict = False # load sentence position embedding weights as well

    model.to(device)
    model.train()



    london_dataset = PbfMapDataset(data_file_path = london_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = with_type,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        label_encoder = label_encoder,
                                        mode = 'test')

    
    california_dataset = PbfMapDataset(data_file_path = california_file_path, 
                                            tokenizer = tokenizer, 
                                            max_token_len = max_token_len, 
                                            distance_norm_factor = distance_norm_factor, 
                                            spatial_dist_fill = spatial_dist_fill, 
                                            with_type = with_type,
                                            sep_between_neighbors = sep_between_neighbors,
                                            label_encoder = label_encoder,
                                            mode = 'test')

    test_dataset = torch.utils.data.ConcatDataset([london_dataset, california_dataset])

    
    test_loader = DataLoader(test_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)
    

    

   
    print('start testing...')

    # setup loop with TQDM and dataloader
    loop = tqdm(test_loader, leave=True)

    mrr_total = 0.
    prec_total = 0.
    sample_cnt = 0

    gt_list = []
    pred_list = []

    for batch in loop:
        # initialize calculated gradients (from prev step)
        
        # pull all tensor batches required for training
        input_ids = batch['pseudo_sentence'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_ids = batch['sent_position_ids'].to(device)

        #labels = batch['pseudo_sentence'].to(device)
        labels = batch['pivot_type'].to(device)
        pivot_lens = batch['pivot_token_len'].to(device)

        outputs = model(input_ids, attention_mask = attention_mask, position_ids = position_ids,
                labels = labels, pivot_len_list = pivot_lens)


        onehot_labels = F.one_hot(labels, num_classes=9)
        
        gt_list.extend(onehot_labels.cpu().detach().numpy())
        pred_list.extend(outputs.logits.cpu().detach().numpy())

        mrr = label_ranking_average_precision_score(onehot_labels.cpu().detach().numpy(), outputs.logits.cpu().detach().numpy())
        mrr_total += mrr * input_ids.shape[0] 
        sample_cnt += input_ids.shape[0]

    precisions, recalls, fscores, supports = precision_recall_fscore_support(np.argmax(np.array(gt_list),axis=1), np.argmax(np.array(pred_list),axis=1), average=None)
    precision, recall, f1, _ = precision_recall_fscore_support(np.argmax(np.array(gt_list),axis=1), np.argmax(np.array(pred_list),axis=1), average='micro')
    print('precisions:\n', ["{:.3f}".format(prec) for prec in precisions])
    print('recalls:\n', ["{:.3f}".format(rec) for rec in recalls])
    print('fscores:\n', ["{:.3f}".format(f1) for f1 in fscores])
    print('supports:\n', supports)
    print('micro P, micro R, micro F1', "{:.3f}".format(precision), "{:.3f}".format(recall), "{:.3f}".format(f1))
    
    #pdb.set_trace()
    #print(mrr_total/sample_cnt)


       
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--max_token_len', type=int, default=300)
    

    parser.add_argument('--distance_norm_factor', type=float, default = 0.0001)
    parser.add_argument('--spatial_dist_fill', type=float, default = 20)

    parser.add_argument('--with_type', default=False, action='store_true')
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')

    parser.add_argument('--backbone_option', type=str, default='bert-base')
    parser.add_argument('--checkpoint_path', type=str, default=None)

                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')


    testing(args)


if __name__ == '__main__':

    main()

    