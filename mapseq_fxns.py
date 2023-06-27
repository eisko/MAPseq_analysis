# script with fxns

import pandas as pd
import numpy as np
from M194_M220_metadata import *




def sort_by_celltype(proj, it_areas=["OMCc", "AUD", "STR"], ct_areas=["TH"], pt_areas=["AMY","SNr","SCm","PG","PAG","RN"]):
    """
    Function takes in projection matrix and outputs matrix sorted by the 3 major celltypes:
    - IT = intratelencephalic (projects to cortical and/or Striatum), type = 10
    - CT = corticalthalamic (projects to thalamus w/o projection to brainstem), type = 100
    - PT = pyramidal tract (projects to brainstem += other areas), type = 1000
    Returns single dataframe with cells sorted and labelled by 3 cell types (IT/CT/PT)
    
    default areas:
    - it_areas=["OMCc", "AUD", "STR"]
    - ct_areas=["TH"]
    - pt_areas=["AMY","SNr","SCm","PG","PAG","RN"]
    """
    
    
    ds=proj
 
    

    # 1. isolate PT cells
    pt_counts = ds[pt_areas].sum(axis=1)
    pt_idx = ds[pt_counts>0].index
    ds_pt = ds.loc[pt_idx,:]
    ds_pt = ds_pt.sort_values(['PAG','AMY'], ascending=False)
    ds_pt['type'] = 1000

    # Isolate remaining non-PT cells
    ds_npt = ds.drop(pt_idx)

    # Identify CT cells by thalamus projection
    th_idx = ds_npt['TH'] > 0
    ds_th = ds_npt[th_idx]
    ds_th = ds_th.sort_values('TH', ascending=False)
    ds_th['type'] = 100

    # Identify IT cells by the remaining cells (non-PT, non-CT)
    ds_nth = ds_npt[~th_idx]
    ds_nth = ds_nth.sort_values(['OMCc','AUD','STR'],ascending=False)
    ds_nth['type'] = 10

    # combine IT and CT cells
    ds_npt = pd.concat([ds_nth, ds_th])

    # combine IT/CT and PT cells
    sorted = pd.concat([ds_npt,ds_pt],ignore_index=True)

    sorted=sorted.reset_index(drop=True)
    
    return sorted

def est_proj_prob(bin_proj, reps=1000, sample_size=300):
    """
    funtion takes subsample of projection matrix and repetitions as input.
    Funtion outputs vector where dim0 = projection probability of subsample, and dim1 = repetition
    """
    
    est_probs = []
    for i in range(reps):
        new = bin_proj.sample(sample_size)
        est_probs.append(new.sum()/sample_size)
        
#         if i%100 == 0:
#             print('finished simulation', i)
        
    return np.array(est_probs)


def clean_up_data(df_dirty, to_drop = ['OB', 'ACAi', 'ACAc', 'HIP']):

    # drop unused areas
    dropped = df_dirty.drop(to_drop, axis=1)

    # change RN to bs
    replaced = dropped.rename(columns={'RN':'BS'})

    # drop neurons w/ 0 projections after removing negative regions
    nodes = replaced.drop(["OMCi"], axis=1).sum(axis=1)
    n_idx = nodes > 0 # non-zero projecting neurons
    clean = replaced[n_idx]
    

    return clean


def df_list_to_nodes(df_list, drop = ["OMCi", "type"], species=None, meta=metadata):
    """
    Function to turn list of binarized dataframes per animal to dataframe 
    containing node proportions

    df_list = list of binarized dataframes
    drop = list of column names to drop
    mice = list of mouse names
    species = string of species name
    returns dataframe of node proportions
    """

    # determine which species
    # seperate metadata by species
    meta_mm = metadata[metadata["species"]=='MMus'].reset_index(drop=True)
    meta_st = metadata[metadata["species"]=='STeg'].reset_index(drop=True)
    if species == "MMus":
        meta = meta_mm
    elif species == "STeg":
        meta = meta_st

    nodes_list = []
    for i in range(len(df_list)):
        if drop == []:
            int_df = df_list[i]
        else:
            int_df = df_list[i].drop(drop, axis=1)
        nodes = int_df.sum(axis=1)
        node_counts = nodes.value_counts().sort_index()
        node_proportion = node_counts/node_counts.sum()
        total = node_counts.sum()

        

        df_save = pd.DataFrame(node_proportion, columns=["Normalized Frequency"]).reset_index(names="Node Degree")
        df_save["Total of cell type"] = total
        df_save["Species"] = species
        df_save["mouse"] = meta.loc[i,'mice']
        df_save["Dataset"] = meta.loc[i,'dataset']
        nodes_list.append(df_save)

    node_all = pd.concat(nodes_list)

    return node_all

def df_to_nodes(df, drop=['OMCi']):
    int_df = df.drop(drop, axis=1)
    nodes = int_df.sum(axis=1)
    node_counts = nodes.value_counts().sort_index()
    node_proportion = node_counts/node_counts.sum()

    return node_proportion