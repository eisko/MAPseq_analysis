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
from mapseq_fxns import sort_by_celltype

with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_ds.pkl', 'rb') as f:
    omc_ds = pickle.load(f)    
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_dsN.pkl', 'rb') as f:
    omc_dsN = pickle.load(f)
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_ds.pkl', 'rb') as f:
    acc_ds = pickle.load(f)    
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_dsN.pkl', 'rb') as f:
    acc_dsN = pickle.load(f)


####### figure of heatmaps for each mouse
mice = ["mm1", "mm2","mm3", "st1","st2","st3"]

# OMC
fig, axs = plt.subplots(2,3,figsize=(13,10))
i = 0
for ax in axs.flat:
    plot = sort_by_celltype(omc_dsN[i])
    plot['type'] = plot['type']/10000
    # plot = plot.drop(['type'], axis=1)
    
    sns.heatmap(plot, norm=LogNorm(), cmap="flare", cbar_kws={'label': 'barcode count'}, ax=ax)
    ax.set_title("OMC - "+mice[i])
    i+=1

fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_OMC_individual.jpg',dpi=300, bbox_inches='tight')

# ACA
fig, axs = plt.subplots(2,3,figsize=(12,10))
i = 0
for ax in axs.flat:
    plot = sort_by_celltype(acc_dsN[i])
    plot['type'] = plot['type']/10000
    # plot = plot.drop(['type'], axis=1)

    sns.heatmap(plot, norm=LogNorm(), cmap="flare", cbar_kws={'label': 'barcode count'}, ax=ax)
    ax.set_title("ACA - "+mice[i])
    i+=1

fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_ACA_individual.jpg',dpi=300, bbox_inches='tight')

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
sns.set(font_scale=1.5)

i = 0
for ax in axs.flat:
    plot = sort_by_celltype(all_ds_omc[i])
    plot = plot.drop(['ACA-i','ACA-c'], axis=1)
    plot['type'] = plot['type']/10000
    plot = plot.drop(['type'], axis=1)

    sns.heatmap(plot, ax=ax, norm = LogNorm(vmin=0.0001, vmax=1), cmap="flare", cbar_kws={'label': 'Log Normalized Barcode Count'})
    ax.set_title(titles[i]+" OMC")
    i+=1


fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_OMC_all.jpg',dpi=300, bbox_inches='tight')

# ACA
fig, axs = plt.subplots(1,2,figsize=(12,6))
all_ds_acc = [all_mm_acc, all_st_acc]
sns.set(font_scale=1.5)


i = 0
for ax in axs.flat:
    plot = sort_by_celltype(all_ds_acc[i])
    plot = plot.drop(['OMCi','OMCc'], axis=1)
    plot['type'] = plot['type']/10000
    plot = plot.drop(['type'], axis=1)

    sns.heatmap(plot, ax=ax, norm = LogNorm(vmin=0.0001, vmax=1), cmap="flare", cbar_kws={'label': 'Log Normalized Barcode Count'})
    ax.set_title(titles[i]+" ACA")
    i+=1


fig.savefig('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/sortedN_ACA_all.jpg',dpi=300, bbox_inches='tight')