# MAPseq_processing.py
# 230714

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from M194_M220_metadata import *

def clean_up_data(df_dirty, to_drop = ['OB', 'ACAi', 'ACAc', 'HIP']):
    """Clean up datasets so all matrices are in the same format. Function 
        (1) drops unwanted columns, e.g. negative controls or dissections of other injection sites. 
        (2) renames RN (allen acronym) as BS (my brainstem acronym)
        (3) drops neurons w/ 0 projections after dropped columns

    Args:
        df_dirty (DataFrame)        | Pandas dataframe (neurons x area) that needs to be processed
        to_drop (list, optional)    | columns to drop, DEFAULT: ['OB', 'ACAi', 'ACAc', 'HIP']

    Returns:
        clean (DataFrame)   | Cleaned up data in dataframe
    """

    # 1. drop unused areas
    dropped = df_dirty.drop(to_drop, axis=1)

    # 2. change RN to bs
    replaced = dropped.rename(columns={'RN':'BS'})

    # 3. drop neurons w/ 0 projections after removing negative regions
    nodes = replaced.drop(["OMCi"], axis=1).sum(axis=1)
    n_idx = nodes > 0 # non-zero projecting neurons
    clean = replaced[n_idx]
    
    return clean


def dfs_preprocess_counts(df_list, drop=["OMCi", "type"]):
    """INPUT: Takes list of dataframe(s) that are normalized counts (counts normalized to spike-in RNA),
                disregards values in volumns specified by `drop`
       OUTPUT: Returns list of dataframe(s) where each dataframe is normalized to its dataframe median

    Args:
        df_list (list)  | List of dataframes of ncounts
        drop (list)     | List of columns to drop and not account for when determining median

    returns:
        out_list (list): List of dataframes normalized to dataframe median
    """
    out_list = []
    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        vals = df.values.flatten()
        idx = vals.nonzero() # only use non-zero ncounts for determining median
        plot = vals[idx]
        median = np.median(plot)
        out_df = df/median

        for j in range(len(drop)):
            out_df[drop[j]] = df_list[i][drop[j]] # add dropped columns back in, can preserve labels
        
        out_list.append(out_df)

    return out_list


def sort_by_celltype(proj, it_areas=["OMCc", "AUD", "STR"], ct_areas=["TH"], pt_areas=["AMY","SNr","SCm","PG","PAG","BS"]):
    """
    Function takes in projection matrix and outputs matrix sorted by the 3 major celltypes:
    - IT = intratelencephalic (projects to cortical and/or Striatum), type = 10
    - CT = corticalthalamic (projects to thalamus w/o projection to brainstem), type = 100
    - PT = pyramidal tract (projects to brainstem += other areas), type = 1000
    Returns single dataframe with cells sorted and labelled by 3 cell types (IT/CT/PT)
    
    default areas:
    - it_areas=["OMCc", "AUD", "STR"]
    - ct_areas=["TH"]
    - pt_areas=["AMY","SNr","SCm","PG","PAG","BS"]
    """
    
    ds=proj.copy()
 
    # 1. isolate PT cells
    pt_counts = ds[pt_areas].sum(axis=1)
    pt_idx = ds[pt_counts>0].index
    ds_pt = ds.loc[pt_idx,:]
    ds_pt = ds_pt.sort_values(['PAG','AMY'], ascending=False)
    ds_pt['type'] = "PT"

    # Isolate remaining non-PT cells
    ds_npt = ds.drop(pt_idx)

    # Identify CT cells by thalamus projection
    th_idx = ds_npt['TH'] > 0
    ds_th = ds_npt[th_idx]
    ds_th = ds_th.sort_values('TH', ascending=False)
    ds_th['type'] = "CT"

    # Identify IT cells by the remaining cells (non-PT, non-CT)
    ds_nth = ds_npt[~th_idx]
    ds_nth = ds_nth.sort_values(['OMCc','AUD','STR'],ascending=False)
    ds_nth['type'] = "IT"

    # combine IT and CT cells
    ds_npt = pd.concat([ds_nth, ds_th])

    # combine IT/CT and PT cells
    sorted = pd.concat([ds_npt,ds_pt],ignore_index=True)

    sorted=sorted.reset_index(drop=True)
    
    return sorted


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

def dfs_to_proportions(df_list, drop=["OMCi", "type"], cell_type=None, meta=metadata):
    """Output dataframe of proportions in format that can be plotted with seaborn

    Args:
        df_list (list): 
            - List of dataframes of neurons/BC by areas
        drop (list, optional): 
            - Defaults to ["OMCi", "type"]
            - list of areas/columns to drop before calculating proportions
        cell_type (string, optional): 
            - Specify cell types in df, either IT, CT or PT
            - Defaults to None

    Returns:
        plot_df (pandas_dataframe):
            - returns dataframe in format for seaborn plotting
            - columns = areas, and other metadata
    """

    plot_df = pd.DataFrame(columns=["area", "proportion", "mice", "species", "dataset"])

    if cell_type == "IT":
        drop = ["OMCi", 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = ["OMCi",'OMCc', 'AUD']

    mice = meta["mice"]
    species = meta["species"]
    dataset = meta["dataset"]

    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        bc_sum = df.sum()
        proportion = bc_sum/df.shape[0]
        df_add = pd.DataFrame({"area":proportion.index.values, "proportion":proportion.values, 
        "mice":mice[i], "species":species[i], "dataset":dataset[i]})
        plot_df = pd.concat([plot_df, df_add])
    
    return plot_df