# MAPseq Projection Strength - modifiable
# 220919
# ECI - for committee meeting on 220922
# to be used with conda environment banerjeelab
# make plots to test indpendence of co-prejections

########## input variables
in_f = '/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/omc_ds.pkl'
out_f = '/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/omc_PT_'
omc = True
cell_type = 1000 # IT == 10, CT == 100, PT == 1000, all_cells == None

###### load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
import pickle
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize
from mapseq_fxns import sort_by_celltype

# set random seed
np.random.seed(10)

###### load data
with open(in_f, 'rb') as f:
    ds = pickle.load(f)    

# binarize data
bin = []
for i in range(6):
    bin.append(pd.DataFrame(binarize(ds[i], threshold=1), columns=ds[i].columns))

# restrict to cell type if specified
if cell_type:
    for i in range(6):
        temp = sort_by_celltype(bin[i])
        idx = temp['type']==cell_type
        bin[i] = temp[idx]
        bin[i] = bin[i].drop(['type'], axis=1)

###### functions
def shuffle_cols(bin_proj):
    
    """
    function that takes  proj matrix and returns shuffled version
    will shuffle rows w/in columns, i.e. preserve total number of projections to
    area but disrupt dependence b/w columns
    bin_proj = binarized projection matrix (row = cells, columns = brain areas)
    """
    random = bin_proj.copy()
    for i in range(random.shape[1]):
        random.iloc[:,i] = np.random.permutation(random.iloc[:,i].values)
    
    return random

def joint_prob(bin_proj):
    
    """
    funtion that takes projection matrix and return joint probabilities of 
    cell/neuron projecting to 2 areas
    bin_proj = binarized projection matrix (row = cells, columns = brain areas)
    """
    
    ds = bin_proj
    n_areas = ds.shape[1]
    result = np.zeros((n_areas,n_areas))
    
    for i in range(n_areas):
        for j in range(n_areas):
            a1 = ds.iloc[:,i]
            a2 = ds.iloc[:,j]
            union = a1 + a2
            counts2 = sum(union == 2)
            
            result[i,j] = counts2/ds.shape[0]
    
    return result

def shuffle_sims(bin_proj, reps=1000):
    """
    function takes binary projection matrix and number of repetitions (n) as input
    funtion outputs ndarray of joint probability matrices for n repetitions
    note: takes ~ 2.5 mins per 1000 reps (scales linearly?)
    """

    shuffle_probs = []
    for i in range(reps):
        new = shuffle_cols(bin_proj)
        shuffle_probs.append(joint_prob(new))
        
        if i%100 == 0:
            print('finished simulation', i)
        
    return np.array(shuffle_probs)


##### calculate observed, expected, and delta joint probabilities
# expected based on multiplying individual projection proportions between each region
# combine into lab mouse/singing mouse aggregate
bin_mm = pd.concat([bin[0],bin[1],bin[2]])
bin_st = pd.concat([bin[3],bin[4],bin[5]])

# lab mouse
proj_probm = bin_mm.sum(axis=0)/bin_mm.shape[0]
exp_mm = np.outer(proj_probm,proj_probm)
exp_mm = np.tril(exp_mm, -1)
exp_mm[exp_mm==0] = np.nan

obs_mm = joint_prob(bin_mm)
obs_mm = np.tril(obs_mm, -1)
obs_mm[obs_mm==0] = np.nan    

delta_mm = obs_mm-exp_mm
delta_mm = np.tril(delta_mm, -1)
delta_mm[delta_mm==0] = np.nan

# Singing mouse
proj_probs = bin_st.sum(axis=0)/bin_st.shape[0]
exp_st = np.outer(proj_probs,proj_probs)
exp_st = np.tril(exp_st, -1)
exp_st[exp_st==0] = np.nan

obs_st = joint_prob(bin_st)
obs_st = np.tril(obs_st, -1)
obs_st[obs_st==0] = np.nan    

delta_st = obs_st-exp_st
delta_st = np.tril(delta_st, -1)
delta_st[delta_st==0] = np.nan

######### plot of correlation heatmaps of joint probabilities

plot_df = [obs_mm, exp_mm, delta_mm, obs_st, exp_st, delta_st]
plot_titles = ['Observed - lab mouse', 'Expected - lab mouse', 'Observed-Expected - lab mouse',
                'Observed - singing mouse', 'Expected - singing mouse', 'Observed-Expected - singing mouse']

fig, axs = plt.subplots(2,3, figsize=[25,14])

i=0
for ax in axs.flat:
    sns.heatmap(plot_df[i],
                annot=True,
                annot_kws={"size":7.5},
                yticklabels=bin[0].columns,
                xticklabels=bin[0].columns,
                cbar_kws={'label': 'joint prob'},
                cmap='bwr',
                center=0.00,
                fmt='.1%',
                ax=ax)
    ax.set_title(plot_titles[i])
    i+=1

fig.savefig(out_f + 'joint_prob.jpg',dpi=300, bbox_inches='tight')


####### run simulation
from datetime import datetime
start = datetime.now()
shuffle = shuffle_sims(bin_mm)
end = datetime.now()
print('\n','run time:',end-start, '\n')

n = bin_mm.shape[1]
p_vals = np.zeros([n,n])
sums = np.zeros([n,n])

for i in range(n):
    for j in range(n):
        obs = obs_mm[i,j]
        test = shuffle[:,i,j]
        mean = test.mean()
        if obs >= mean:
            above = sum(test >= obs)
            sums[i,j] = above
        else:
            below = sum(test <= obs)
            sums[i,j] = below
        
        p_vals[i,j] = (sums[i,j]+1)/(1000+1)
        


