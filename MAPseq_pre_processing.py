# MAPseq pre-processing
# 220907
# ECI - for committee meeting on 220922
# to be used with conda environment banerjeelab

####### 1. load packages
import pandas as pd
import numpy as np
import scipy.io as sio
import pickle
import mat73

# Load data and target area names
data = mat73.loadmat('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/MAPseq_data_June_2022/M194BarcodeMatrix.mat')
cols = pd.read_csv('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/MAPseq_data_June_2022/cols_to_brain_area.csv', header=None, names=['brain_areas', 'sample'])
brain_areas = cols.brain_areas.to_list()

sample = ['Bm1', 'Bm2', 'Bm3', 'Bs1', 'Bs2', 'Bs3']
mice = ['mm-1', 'mm-2', 'mm-3', 'steg-1', 'steg-2', 'steg-3']


######### 2. Pre-process data
# convert data into list of dataframes of raw and normalized counts
datasets = [] # list of raw count dataframes
datasetsN = [] # list of normalized count dataframes
for s in sample:
    datasets.append(data[s])
    datasetsN.append(data[s+'norm'])

# restrict datasets to columns relevant to each sample
# original data has columns for every sample collected across all brains
ctl = cols[cols['sample'] == 'control'] # controls are common across all brains, keep track of control columns
for i in range(6):
    b_idx = pd.concat([cols[cols['sample'] == mice[i]], ctl])
    datasets[i] = datasets[i][:, b_idx.index]
    datasets[i] = pd.DataFrame(datasets[i], columns=b_idx.brain_areas.to_list())
    datasetsN[i] = datasetsN[i][:, b_idx.index]
    datasetsN[i] = pd.DataFrame(datasetsN[i], columns=b_idx.brain_areas.to_list())

######## 3. Threshold & Filter data
# only keep neurons/barcodes that match thresholds
# max projection site >= 5, injection site >= 50 ==> thresholds used by Xaoyin (?)
# In between Xaoyin and Justus' theshold thresholds, Justus's thresholds were much stricter (max projection >= 30, injection site >= 100)
# set thresholds
# note: injection site = columns[-6:-3], target sites = colummns[:-6]
thr_proj = 5
thr_inj = 50
ds_thr = []
for i in range(6):
    ds = datasets[i]
    
    proj = ds.iloc[:,0:-6]
    max_proj = proj.max(axis=1)
    idx_proj = max_proj>=thr_proj
    ds_new = ds[idx_proj]
    
    injs=ds_new.iloc[:,-6:-3]
    max_inj = injs.max(axis=1)
    inj_idx = max_inj>=thr_inj
    ds_new2 = ds_new[inj_idx]
    ds_thr.append(ds_new2)


# filter out neurons that have barcode detected in negative controls (OB and ctl)
ds_filt = []
ds_filtN = []
for i in range(6):
    ds_filt.append(ds_thr[i])
    ds_filtN.append(datasetsN[i])

for i in range(6):
    # OB
    idx = ds_filt[i].OB == 0
    ds_filt[i] = ds_filt[i][idx]
    
    # ctl
    ctl = ds_filt[i].iloc[:, -3:]
    idx = ctl.sum(axis=1) == 0
    ds_filt[i] = ds_filt[i][idx]

# drop negative control columns
for i in range(6):
    ds_filt[i] = ds_filt[i].drop(['OB', 'L1 Tar ctl', 'H2O Tar ctl',
       'H2O Inj ctl'], axis=1)

# filter out barcodes where max value is not injection site
for i in range(6):
    idx = ds_filt[i].idxmax(axis=1).str.contains("OMC-Ai|OMC-Pi|ACA-i")
    ds_filt[i] = ds_filt[i][idx]

# apply filters to normalized dataset and reset indices
for i in range(6):
    idx = ds_filt[i].index
    ds_filtN[i] = ds_filtN[i].loc[idx].reset_index(drop=True)
    ds_filt[i] = ds_filt[i].reset_index(drop=True)



######## 4. Rearrange columns for analysis
fin_cols = ['OMC-Ai', 'OMC-Pi', 'ACA-i','OMC-Ac', 'OMC-Pc', 'ACA-c', 'AUD', 'STR-d', 'STR-v', 'TH', 'HY', 'AMY',
       'HIP', 'SNr', 'SCm', 'PG', 'PAG-Ad', 'PAG-Av', 'PAG-Pd', 'PAG-Pv', 'RN']
for i in range(6):
    ds_filt[i] = ds_filt[i][fin_cols]
    ds_filtN[i] = ds_filtN[i][fin_cols]

######## 5. Save datasets as pickle file (compressed objects) to be used in later analyses
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/datasets.pkl', 'wb') as f:
    pickle.dump(ds_filt, f, protocol=-1)
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/datasetsN.pkl', 'wb') as f:
    pickle.dump(ds_filtN, f, protocol=-1)

######## 6. Combine divided areas for initial analysis?
datasets_fin = ds_filt
datasetsN_fin = ds_filtN

# combine regions in count data
for i in range(6):
    ds = datasets_fin[i]
    ds["OMCi"] = ds["OMC-Ai"] + ds["OMC-Pi"]
    ds["OMCc"] = ds["OMC-Ac"] + ds["OMC-Pc"]
    ds["STR"] = ds["STR-d"] + ds["STR-v"]
    ds["PAG"] = ds["PAG-Ad"] + ds["PAG-Av"] + ds["PAG-Pd"] + ds["PAG-Pv"]
    datasets_fin[i] = ds

# combine regions in normalized data
for i in range(6):
    ds = datasetsN_fin[i]
    ds["OMCi"] = ds["OMC-Ai"] + ds["OMC-Pi"]
    ds["OMCc"] = ds["OMC-Ac"] + ds["OMC-Pc"]
    ds["STR"] = ds["STR-d"] + ds["STR-v"]
    ds["PAG"] = ds["PAG-Ad"] + ds["PAG-Av"] + ds["PAG-Pd"] + ds["PAG-Pv"]
    datasetsN_fin[i] = ds


######## 7. seperate by injection site (OMC and ACA/ACC)
### Note: doing this before excluding/rearranging columns, b/c counts could be not equalized after combining?
omc_ds = []
omc_dsN = []
acc_ds = []
acc_dsN = []
for i in range(6):
    inj = datasets_fin[i][["OMC-Ai","OMC-Pi","ACA-i"]].idxmax(axis=1)
    idx_omc = inj.str.contains("OMC-Ai|OMC-Pi")
    idx_acc = inj == "ACA-i"
    omc_ds.append(datasets_fin[i][idx_omc])
    acc_ds.append(datasets_fin[i][idx_acc])
    omc_dsN.append(datasetsN_fin[i][idx_omc])
    acc_dsN.append(datasetsN_fin[i][idx_acc])


# only keep combined columns
fin_cols = ['OMCi', 'OMCc', 'ACA-i', 'ACA-c', 'AUD', 'STR', 'TH', 'HY', 'AMY',
       'HIP', 'SNr', 'SCm', 'PG', 'PAG', 'RN']
for i in range(6):
    omc_ds[i] = omc_ds[i][fin_cols]
    omc_dsN[i] = omc_dsN[i][fin_cols]
    acc_ds[i] = acc_ds[i][fin_cols]
    acc_dsN[i] = acc_dsN[i][fin_cols]



with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_ds.pkl', 'wb') as f:
    pickle.dump(omc_ds, f, protocol=-1)
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_dsN.pkl', 'wb') as f:
    pickle.dump(omc_dsN, f, protocol=-1)
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_ds.pkl', 'wb') as f:
    pickle.dump(acc_ds, f, protocol=-1)
with open('/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_dsN.pkl', 'wb') as f:
    pickle.dump(acc_dsN, f, protocol=-1)

######## 7. Combine divided areas for initial analysis? -> will be taken care of in individual analyses

