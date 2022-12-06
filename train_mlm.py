import os
import sys
from transformers import RobertaTokenizer, BertTokenizer
from tqdm import tqdm  # for our progress bar
from transformers import AdamW

import torch
from torch.utils.data import DataLoader

from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from models.spatial_bert_model import  SpatialBertForMaskedLM
from datasets.osm_sample_loader import PbfMapDataset
from transformers.models.bert.modeling_bert import BertForMaskedLM

import numpy as np
import argparse 
import pdb


DEBUG = False


def training(args):

    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr #1e-7 # 5e-5
    save_interval = args.save_interval
    max_token_len = args.max_token_len
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill
    with_type = args.with_type
    sep_between_neighbors = args.sep_between_neighbors
    freeze_backbone = args.freeze_backbone

    bert_option = args.bert_option
    if_no_spatial_distance = args.no_spatial_distance

    assert bert_option in ['bert-base','bert-large']

    london_file_path = '../data/sql_output/osm-point-london.json'
    california_file_path = '../data/sql_output/osm-point-california.json'

    if args.model_save_dir is None:
        sep_pathstr = '_sep' if sep_between_neighbors else '_nosep' 
        freeze_pathstr = '_freeze' if freeze_backbone else '_nofreeze'
        context_pathstr = '_nocontext' if if_no_spatial_distance else '_withcontext'
        model_save_dir = '/data2/zekun/spatial_bert_weights/mlm_mem_lr' + str("{:.0e}".format(lr)) + sep_pathstr + context_pathstr +'/'+bert_option+ freeze_pathstr + '_mlm_mem_london_california_bsize' + str(batch_size)

        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
    else:
        model_save_dir = args.model_save_dir

    
    print('model_save_dir', model_save_dir)
    print('\n')

    if bert_option == 'bert-base':
        bert_model = BertForMaskedLM.from_pretrained('bert-base-uncased')
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance)
    elif bert_option == 'bert-large':
        bert_model = BertForMaskedLM.from_pretrained('bert-large-uncased')
        tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance, hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24)
    else:
        raise NotImplementedError


    model = SpatialBertForMaskedLM(config)

    model.load_state_dict(bert_model.state_dict() , strict = False) # load sentence position embedding weights as well
    
    if bert_option == 'bert-large' and freeze_backbone:
        print('freezing backbone weights')
        for param in model.parameters():
            param.requires_grad = False

        for param in  model.cls.parameters(): 
            param.requires_grad = True

        for param in  model.bert.encoder.layer[21].parameters(): 
            param.requires_grad = True
        for param in  model.bert.encoder.layer[22].parameters(): 
            param.requires_grad = True
        for param in  model.bert.encoder.layer[23].parameters(): 
            param.requires_grad = True
        

    london_train_dataset = PbfMapDataset(data_file_path = london_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = with_type,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        label_encoder = None,
                                        mode = None)

    california_train_dataset = PbfMapDataset(data_file_path = california_file_path, 
                                            tokenizer = tokenizer, 
                                            max_token_len = max_token_len, 
                                            distance_norm_factor = distance_norm_factor, 
                                            spatial_dist_fill = spatial_dist_fill, 
                                            with_type = with_type,
                                            sep_between_neighbors = sep_between_neighbors,
                                            label_encoder = None,
                                            mode = None)

    train_dataset = torch.utils.data.ConcatDataset([london_train_dataset, california_train_dataset])


    if DEBUG:
        train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=True, pin_memory=True, drop_last=True)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()


    # initialize optimizer
    optim = AdamW(model.parameters(), lr = lr)

    print('start training...')

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        iter = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['masked_input'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_list_x = batch['norm_lng_list'].to(device)
            position_list_y = batch['norm_lat_list'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)

            labels = batch['pseudo_sentence'].to(device)

            outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
                position_list_x = position_list_x, position_list_y = position_list_y, labels = labels)


            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix({'loss':loss.item()})

            if DEBUG:
                print('ep'+str(epoch)+'_' + '_iter'+ str(iter).zfill(5), loss.item() )

            iter += 1

            if iter % save_interval == 0 or iter == loop.total:
                save_path = os.path.join(model_save_dir, 'mlm_mem_keeppos_ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                + '_' +str("{:.4f}".format(loss.item())) +'.pth' )
                torch.save(model.state_dict(), save_path)
                print('saving model checkpoint to', save_path)




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--max_token_len', type=int, default=300)
    

    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--distance_norm_factor', type=float, default = 0.0001)
    parser.add_argument('--spatial_dist_fill', type=float, default = 20)

    parser.add_argument('--with_type', default=False, action='store_true')
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--bert_option', type=str, default='bert-base')
    parser.add_argument('--model_save_dir', type=str, default=None)
    
                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')


    # out_dir not None, and out_dir does not exist, then create out_dir
    if args.model_save_dir is not None and not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    training(args)

    

if __name__ == '__main__':

    main()

    