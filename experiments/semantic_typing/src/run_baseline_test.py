import os 
import pdb
import argparse
import numpy as np
import time

MODEL_OPTIONS = ['bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large','luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large']

def execute_command(command, if_print_command):
    t1 = time.time()

    if if_print_command:
        print(command)
    os.system(command)

    t2 = time.time()
    time_usage = t2 - t1 
    return time_usage

def run_test(args):
    weight_dir = args.weight_dir
    backbone_option = args.backbone_option
    gpu_id = str(args.gpu_id)
    if_print_command = args.print_command
    sep_between_neighbors =  args.sep_between_neighbors

    assert backbone_option in MODEL_OPTIONS

    if sep_between_neighbors:
        sep_str = '_sep'
    else:
        sep_str = ''
    
    if 'large' in backbone_option:
        checkpoint_dir = os.path.join(weight_dir, 'typing_lr1e-06_%s_nofreeze%s_london_california_bsize12'% (backbone_option, sep_str))
    else:
        checkpoint_dir = os.path.join(weight_dir, 'typing_lr5e-05_%s_nofreeze%s_london_california_bsize12'% (backbone_option, sep_str))
    weight_files = os.listdir(checkpoint_dir)

    val_loss_list = [weight_file.split('_')[-1] for weight_file in weight_files]
    min_loss_weight = weight_files[np.argmin(val_loss_list)]

    checkpoint_path = os.path.join(checkpoint_dir, min_loss_weight)

    if sep_between_neighbors:
        command = 'CUDA_VISIBLE_DEVICES=%s python3 test_cls_baseline.py  --sep_between_neighbors --backbone_option=%s --batch_size=8 --with_type --checkpoint_path=%s ' % (gpu_id, backbone_option, checkpoint_path)
    else:
        command = 'CUDA_VISIBLE_DEVICES=%s python3 test_cls_baseline.py  --backbone_option=%s --batch_size=8 --with_type --checkpoint_path=%s ' % (gpu_id, backbone_option, checkpoint_path)


    execute_command(command, if_print_command)




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight_dir', type=str, default='/data2/zekun/spatial_bert_baseline_weights/')
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--backbone_option', type=str, default=None)
    parser.add_argument('--gpu_id', type=int, default=0) # output prefix 

    parser.add_argument('--print_command', default=False, action='store_true')

                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    run_test(args)


if __name__ == '__main__':

    main()

    

