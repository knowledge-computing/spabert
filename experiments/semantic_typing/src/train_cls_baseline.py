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

sys.path.append('../../../')
from datasets.osm_sample_loader import PbfMapDataset
from datasets.const import *
from utils.baseline_utils import get_baseline_model
from models.baseline_typing_model import BaselineForSemanticTyping


MODEL_OPTIONS = ['bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large','luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large']

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


    backbone_option = args.backbone_option

    assert(backbone_option in MODEL_OPTIONS)
    
   
    london_file_path = '../data/sql_output/osm-point-london-typing.json'
    california_file_path = '../data/sql_output/osm-point-california-typing.json'

    if args.model_save_dir is None:
        freeze_pathstr = '_freeze' if freeze_backbone else '_nofreeze'
        sep_pathstr = '_sep' if sep_between_neighbors else '_nosep'
        model_save_dir = '/data2/zekun/spatial_bert_baseline_weights/typing_lr' + str("{:.0e}".format(lr))  +'_'+backbone_option+ freeze_pathstr + sep_pathstr + '_london_california_bsize' + str(batch_size)
        
        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
    else:
        model_save_dir = args.model_save_dir

    print('model_save_dir', model_save_dir)
    print('\n')

    
    label_encoder = LabelEncoder()
    label_encoder.fit(CLASS_9_LIST)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    backbone_model, tokenizer = get_baseline_model(backbone_option)
    model = BaselineForSemanticTyping(backbone_model, backbone_model.config.hidden_size, len(CLASS_9_LIST))

    model.to(device)
    model.train()

    london_train_val_dataset = PbfMapDataset(data_file_path = london_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = with_type,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        label_encoder = label_encoder,
                                        mode = 'train')

    percent_80 = int(len(london_train_val_dataset) * 0.8)
    london_train_dataset, london_val_dataset = torch.utils.data.random_split(london_train_val_dataset, [percent_80, len(london_train_val_dataset) - percent_80])

    california_train_val_dataset = PbfMapDataset(data_file_path = california_file_path, 
                                            tokenizer = tokenizer, 
                                            max_token_len = max_token_len, 
                                            distance_norm_factor = distance_norm_factor, 
                                            spatial_dist_fill = spatial_dist_fill, 
                                            with_type = with_type,
                                            sep_between_neighbors = sep_between_neighbors,
                                            label_encoder = label_encoder,
                                            mode = 'train')
    percent_80 = int(len(california_train_val_dataset) * 0.8)
    california_train_dataset, california_val_dataset = torch.utils.data.random_split(california_train_val_dataset, [percent_80, len(california_train_val_dataset) - percent_80])

    train_dataset = torch.utils.data.ConcatDataset([london_train_dataset, california_train_dataset])
    val_dataset = torch.utils.data.ConcatDataset([london_val_dataset, california_val_dataset])


    
    train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=num_workers,
                            shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)



    

    

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
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['sent_position_ids'].to(device)

            #labels = batch['pseudo_sentence'].to(device)
            labels = batch['pivot_type'].to(device)
            pivot_lens = batch['pivot_token_len'].to(device)

            outputs = model(input_ids, attention_mask = attention_mask, position_ids = position_ids,
                labels = labels, pivot_len_list = pivot_lens)


            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix({'loss':loss.item()})
            
            
            iter += 1

            if iter % save_interval == 0 or iter == loop.total:
                loss_valid = validating(val_loader, model, device)
                print('validation loss', loss_valid)

                save_path = os.path.join(model_save_dir, 'ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                + '_' +str("{:.4f}".format(loss.item())) + '_val' + str("{:.4f}".format(loss_valid)) +'.pth' )

                torch.save(model.state_dict(), save_path)
                print('saving model checkpoint to', save_path)

                
                
def validating(val_loader, model, device):

    with torch.no_grad():

        loss_valid = 0
        loop = tqdm(val_loader, leave=True)

        for batch in loop:
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_ids = batch['sent_position_ids'].to(device)

            labels = batch['pivot_type'].to(device)
            pivot_lens = batch['pivot_token_len'].to(device)

            outputs = model(input_ids, attention_mask = attention_mask, position_ids = position_ids,
                 labels = labels, pivot_len_list = pivot_lens)

            loss_valid  += outputs.loss

        loss_valid /= len(val_loader)

        return loss_valid


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

    parser.add_argument('--backbone_option', type=str, default='bert-base')
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

    