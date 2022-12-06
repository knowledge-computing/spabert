#!/usr/bin/env python
# coding: utf-8


import sys
import os
import glob
import json
import numpy as np
import pandas as pd
import pdb


prediction_dir = sys.argv[1]

print(prediction_dir)

gt_dir = '../data_processing/outputs/alignment_gt_dir/'
prediction_path_list = sorted(os.listdir(prediction_dir))

DISPLAY = False
DETAIL = False

if DISPLAY:
    from IPython.display import display

def recall_at_k_all_map(all_rank_list, k = 1):
    
    rank_list = [item for sublist in all_rank_list for item in sublist]
    total_query = len(rank_list)
    prec = np.sum(np.array(rank_list)<=k)
    prec = 1.0 * prec / total_query
    
    return prec

def recall_at_k_permap(all_rank_list, k = 1):
    
    prec_list = []
    for rank_list in all_rank_list:
        total_query = len(rank_list)
        prec = np.sum(np.array(rank_list)<=k)
        prec = 1.0 * prec / total_query
        prec_list.append(prec)
    
    return prec_list




def reciprocal_rank(all_rank_list):
    
    recip_list = [1./rank for rank in all_rank_list]
    mean_recip = np.mean(recip_list)
    
    return mean_recip, recip_list




count_hist_list = []

all_rank_list = []

all_recip_list = []

permap_recip_list = []

for map_path in prediction_path_list:
    
    pred_path = os.path.join(prediction_dir, map_path)
    gt_path = os.path.join(gt_dir, map_path.split('.json')[0] + '.csv')
    
    if DETAIL:
        print(pred_path)
    
    
    with open(gt_path, 'r') as f:
        gt_data = f.readlines()
        
    gt_dict = dict()
    for line in gt_data:
        line = line.split(',')
        pivot_name = line[0]
        gt_uri = line[1]
        gt_dict[pivot_name] = gt_uri
        
    rank_list = []
    pivot_name_list = []
    with open(pred_path, 'r') as f:
        pred_data = f.readlines()
        for line in pred_data:
            pred_dict = json.loads(line)
            #print(pred_dict.keys())
            pivot_name = pred_dict['pivot_name']
            sorted_match_uri = pred_dict['sorted_match_uri']
            #sorted_match_des = pred_dict['sorted_match_des']
            sorted_sim_matrix = pred_dict['sorted_sim_matrix']
            
            
            total = len(sorted_match_uri)
            if total == 1: 
                continue
            
            if pivot_name in gt_dict:
                
                gt_uri = gt_dict[pivot_name]
                
                try:
                    assert gt_uri in sorted_match_uri
                except Exception as e:
                    #print(e)
                    continue
                
                pivot_name_list.append(pivot_name)
                count_hist_list.append(total)
                rank = sorted_match_uri.index(gt_uri) +1
                
                rank_list.append(rank)
                #print(rank,'/',total)
                
    all_rank_list.append(rank_list)
    
    mean_recip, recip_list = reciprocal_rank(rank_list)

    all_recip_list.extend(recip_list)
    permap_recip_list.append(recip_list)
    
    d = {'pivot': pivot_name_list + ['AVG'], 'rank':rank_list + [' '] ,'recip rank': recip_list + [str(mean_recip)]}
    if DETAIL:
        print(pivot_name_list, rank_list, recip_list)
    
    if DISPLAY:
        df = pd.DataFrame(data=d)

        display(df)
    
    

print('all mrr, micro', np.mean(all_recip_list))


if DETAIL:

    len(rank_list)



    print(recall_at_k_all_map(all_rank_list, k = 1))
    print(recall_at_k_all_map(all_rank_list, k = 2))
    print(recall_at_k_all_map(all_rank_list, k = 5))
    print(recall_at_k_all_map(all_rank_list, k = 10))


    print(prediction_path_list)


prec_list_1 = recall_at_k_permap(all_rank_list, k = 1)
prec_list_2 = recall_at_k_permap(all_rank_list, k = 2)
prec_list_5 = recall_at_k_permap(all_rank_list, k = 5)
prec_list_10 = recall_at_k_permap(all_rank_list, k = 10)

if DETAIL:

    print(np.mean(prec_list_1))
    print(prec_list_1)
    print('\n')

    print(np.mean(prec_list_2))
    print(prec_list_2)
    print('\n')

    print(np.mean(prec_list_5))
    print(prec_list_5)
    print('\n')

    print(np.mean(prec_list_10))
    print(prec_list_10)
    print('\n')








import pandas as pd



map_name_list = [name.split('.json')[0].split('USGS-')[1] for name in prediction_path_list]
d = {'map_name': map_name_list,'recall@1': prec_list_1, 'recall@2': prec_list_2, 'recall@5': prec_list_5, 'recall@10': prec_list_10 }
df = pd.DataFrame(data=d)


if DETAIL:
    print(df)





category = ['15-CA','30-CA','60-CA']
col_1 = [np.mean(prec_list_1[0:4]), np.mean(prec_list_1[4:9]), np.mean(prec_list_1[9:])]
col_2 = [np.mean(prec_list_2[0:4]), np.mean(prec_list_2[4:9]), np.mean(prec_list_2[9:])]
col_3 = [np.mean(prec_list_5[0:4]), np.mean(prec_list_5[4:9]), np.mean(prec_list_5[9:])]
col_4 = [np.mean(prec_list_10[0:4]), np.mean(prec_list_10[4:9]), np.mean(prec_list_10[9:])]



mrr_15 = permap_recip_list[0] + permap_recip_list[1] + permap_recip_list[2] + permap_recip_list[3]
mrr_30 = permap_recip_list[4] + permap_recip_list[5] + permap_recip_list[6] + permap_recip_list[7] + permap_recip_list[8] 
mrr_60 = permap_recip_list[9] + permap_recip_list[10] + permap_recip_list[11] + permap_recip_list[12] + permap_recip_list[13] 



column_5 = [np.mean(mrr_15), np.mean(mrr_30), np.mean(mrr_60)]


d = {'map set': category, 'mrr': column_5, 'prec@1': col_1, 'prec@2': col_2, 'prec@5': col_3, 'prec@10': col_4 }
df = pd.DataFrame(data=d)

print(df)




print('all mrr, micro', np.mean(all_recip_list))

print('\n')



print(recall_at_k_all_map(all_rank_list, k = 1))
print(recall_at_k_all_map(all_rank_list, k = 2))
print(recall_at_k_all_map(all_rank_list, k = 5))
print(recall_at_k_all_map(all_rank_list, k = 10))




if DISPLAY:

    import seaborn 

    p = seaborn.histplot(data = count_hist_list, color = 'blue', alpha=0.2)
    p.set_xlabel("Number of Candiates")
    p.set_title("Candidate Distribution in USGS")




    len(count_hist_list)







