# MAPseq Clustered Maps
# 220907
# ECI - for committee meeting on 220922
# to be used with conda environment banerjeelab

####### 1. load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_ds.pkl', 'rb') as f:
    omc_ds = pickle.load(f)    
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_dsN.pkl', 'rb') as f:
    omc_dsN = pickle.load(f)
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_ds.pkl', 'rb') as f:
    acc_ds = pickle.load(f)    
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_dsN.pkl', 'rb') as f:
    acc_dsN = pickle.load(f)

#### function to seperate dataframe into 3 dataframes of 3 different celltypes (IT, CT, PT)
def sort_by_celltype(proj):
    """
    Function takes in projection matrix and outputs 3 matrices corresponding to the 3 major celltypes:
    - IT = intratelencephalic (projects to cortical and/or Striatum)
    - CT = corticalthalamic (projects to thalamus w/o projection to brainstem)
    - PT = pyramidal tract (projects to brainstem += other areas)
    Returns 3 dataframes containing cells for the 3 cell types
    """
    # neurons to projections to any of the following areas are considered PT cells
    pt_areas = ["SNr","SCm","PG","PAG","RN"]
 
    ds = proj

    # 1. isolate PT cells
    pt_counts = ds[pt_areas].sum(axis=1)
    pt_idx = ds[pt_counts>0].index
    ds_pt = ds.loc[pt_idx,:]
    ds_pt = ds_pt.sort_values('PAG', ascending=False)
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


####### figure of heatmaps for each mouse
mice = ["mm1", "mm2","mm3", "st1","st2","st3"]

# OMC
fig, axs = plt.subplots(2,3,figsize=(12,10))
i = 0
for ax in axs.flat:
    plot = sort_by_celltype(omc_dsN[i])
    plot = plot.drop(['AMY','HIP'], axis=1)
    
    sns.heatmap(plot, norm=LogNorm(), cbar_kws={'label': 'barcode count'}, ax=ax)
    ax.set_title("OMC - "+mice[i])
    i+=1

fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_OMC_individual.jpg',dpi=300, bbox_inches='tight')

# ACC
fig, axs = plt.subplots(2,3,figsize=(12,10))
i = 0
for ax in axs.flat:
    plot = sort_by_celltype(acc_dsN[i])
    plot = plot.drop(['AMY','HIP'], axis=1)
    
    sns.heatmap(plot, norm=LogNorm(), cbar_kws={'label': 'barcode count'}, ax=ax)
    ax.set_title("ACC - "+mice[i])
    i+=1

fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_ACC_individual.jpg',dpi=300, bbox_inches='tight')

######## figure of heatmpas for each species

# combine species datasets
all_mm_omc = pd.concat([omc_dsN[0],omc_dsN[1],omc_dsN[2]], ignore_index=True)
all_st_omc = pd.concat([omc_dsN[3],omc_dsN[4],omc_dsN[5]], ignore_index=True)
all_mm_acc = pd.concat([acc_dsN[0],acc_dsN[1],acc_dsN[2]], ignore_index=True)
all_st_acc = pd.concat([acc_dsN[3],acc_dsN[4],acc_dsN[5]], ignore_index=True)

# OMC
fig, axs = plt.subplots(1,2,figsize=(12,6))
titles = ["Lab Mouse", "Singing Mouse"]
all_ds_omc = [all_mm_omc, all_st_omc]

i = 0
for ax in axs.flat:
    plot = sort_by_celltype(all_ds_omc[i])
    plot = plot.drop(['AMY','HIP','ACA-i','ACA-c'], axis=1)
    plot['type'] = plot['type']/10000
    
    sns.heatmap(plot, ax=ax, norm = LogNorm(vmin=0.0001, vmax=1), cbar_kws={'label': 'Log Normalized Barcode Count'})
    ax.set_title(titles[i]+" OMC")
    i+=1


fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_OMC_all.jpg',dpi=300, bbox_inches='tight')

# ACC
fig, axs = plt.subplots(1,2,figsize=(12,6))
all_ds_acc = [all_mm_acc, all_st_acc]

i = 0
for ax in axs.flat:
    plot = sort_by_celltype(all_ds_acc[i])
    plot = plot.drop(['AMY','HIP','OMCi','OMCc'], axis=1)
    plot['type'] = plot['type']/10000
    
    sns.heatmap(plot, ax=ax, norm = LogNorm(vmin=0.0001, vmax=1), cbar_kws={'label': 'Log Normalized Barcode Count'})
    ax.set_title(titles[i]+" ACC")
    i+=1


fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_ACC_all.jpg',dpi=300, bbox_inches='tight')