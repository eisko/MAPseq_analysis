# MAPseq Projection Strength - modifiable
# 220919
# ECI - for committee meeting on 220922
# to be used with conda environment banerjeelab
# make plots to compare number of neurons projecting to different areas using bootstrapping

########## input variables
in_f = '/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/data_obj/acc_ds.pkl'
out_f = '/Volumes/Data/Emily/MAPseq/MAPseq_June_2022/python_clean/figs/acc_'
omc = False

###### load packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import binarize
import pickle
import seaborn as sns
from matplotlib.colors import LogNorm, Normalize

# set random seed
np.random.seed(10)

###### load data
with open(in_f, 'rb') as f:
    ds = pickle.load(f)    

# binarize data
bin = []
for i in range(6):
    bin.append(pd.DataFrame(binarize(ds[i], threshold=1), columns=ds[i].columns))

    

###### functions
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


# create distribution by randomly sampling MMus data
mm_probs = []
for i in range(3):
    mm_probs.append(est_proj_prob(bin[i]))

# calculate fraction of projections in singing mice
steg_prob = [0,0,0]
for i in range(3,6):
    probs = (bin[i].sum())/bin[i].shape[0]
    steg_prob.append(probs)


# plot distributions and compare w/ STeg individual values
# sample from concatenated lab mouse data all
mice = ["mm1", "mm2","mm3", "st1","st2","st3"]
colors = ["#8E4B66", "#EF5BA1", "#FBDDEA", "#3B3E5D", "#5168AD", "#D3D5EA"]
areas = ['OMCi', 'OMCc', 'ACA-i', 'ACA-c', 'AUD', 'STR','TH','HY','AMY', 'HIP', 'SNr', 'SCm', 'PG', 'PAG', 'RN']


fig, axs = plt.subplots(3,5, figsize=(25,20))

for i in range(15):
    ax= axs.flat[i]
    sns.histplot(mm_probs[0][:,i], ax=ax, color=colors[0])
    sns.histplot(mm_probs[1][:,i], ax=ax, color=colors[1])
    sns.histplot(mm_probs[2][:,i], ax=ax, color=colors[2])
    ax.axvline(steg_prob[3][i], color=colors[3]) # steg1 marker
    ax.axvline(steg_prob[4][i], color=colors[4]) # steg2 marker
    ax.axvline(steg_prob[5][i], color=colors[5]) # steg3 marker
    ax.set_title(areas[i])
    ax.set_ylabel('')
    # legend so know which color corresponds to which mouse
    ax.text(0.05, 0.95, mice[0], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[1])
    ax.text(0.05, 0.9, mice[1], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[2])
    ax.text(0.05, 0.85, mice[2], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[3])
    ax.text(0.05, 0.80, mice[3], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[3])
    ax.text(0.05, 0.75, mice[4], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[4])
    ax.text(0.05, 0.70, mice[5], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[5])

fig.supxlabel('Projection Probability', fontsize=16)
fig.supylabel('Count',fontsize=16)

fig.savefig(out_f+'proj_prob_all.jpg',dpi=300, bbox_inches='tight')

############### create example histograms
# AUD[4], OMCc (1), ACCc (3)
# use mm_probs to plot
fig, axs = plt.subplots(1,2, figsize=(10,5))
if omc:
    regions = [areas[1],areas[4]]
    area_loc = [1,4]
else: # if acc
    regions = [areas[3], areas[4]]
    area_loc = [3,4]

for i in range(2):
    ax= axs.flat[i]
    sns.histplot(mm_probs[0][:,area_loc[i]], ax=ax, color=colors[0])
    sns.histplot(mm_probs[1][:,area_loc[i]], ax=ax, color=colors[1])
    sns.histplot(mm_probs[2][:,area_loc[i]], ax=ax, color=colors[2])
    ax.axvline(steg_prob[3][area_loc[i]], color=colors[3])
    ax.axvline(steg_prob[4][area_loc[i]], color=colors[4])
    ax.axvline(steg_prob[5][area_loc[i]], color=colors[5])
    ax.set_title(regions[i])
#     ax.set_xlabel('projection probability')
    ax.set_ylabel('')
    ax.text(0.05, 0.95, mice[0], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[1])
    ax.text(0.05, 0.9, mice[1], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[2])
    ax.text(0.05, 0.85, mice[2], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[3])
    ax.text(0.05, 0.80, mice[3], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[3])
    ax.text(0.05, 0.75, mice[4], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[4])
    ax.text(0.05, 0.70, mice[5], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[5])

fig.supxlabel('Projection Proportion', fontsize=16)
fig.supylabel('Count',fontsize=16)

fig.savefig(out_f+'contra_aud.jpg',dpi=300, bbox_inches='tight')



###############
# SCm (11), PAG (13)
fig, axs = plt.subplots(1,2, figsize=(10,5))
regions = [areas[11],areas[13]]
area_loc = [11,13]

for i in range(2):
    ax= axs.flat[i]
    sns.histplot(mm_probs[0][:,area_loc[i]], ax=ax, color=colors[0])
    sns.histplot(mm_probs[1][:,area_loc[i]], ax=ax, color=colors[1])
    sns.histplot(mm_probs[2][:,area_loc[i]], ax=ax, color=colors[2])
    ax.axvline(steg_prob[3][area_loc[i]], color=colors[3])
    ax.axvline(steg_prob[4][area_loc[i]], color=colors[4])
    ax.axvline(steg_prob[5][area_loc[i]], color=colors[5])
    ax.set_title(regions[i])
#     ax.set_xlabel('projection probability')
    ax.set_ylabel('')
    ax.text(0.05, 0.95, mice[0], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[1])
    ax.text(0.05, 0.9, mice[1], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[2])
    ax.text(0.05, 0.85, mice[2], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[3])
    ax.text(0.05, 0.80, mice[3], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[3])
    ax.text(0.05, 0.75, mice[4], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[4])
    ax.text(0.05, 0.70, mice[5], transform=ax.transAxes, fontsize=14,
            verticalalignment='top', color=colors[5])

fig.supxlabel('Projection Proportion', fontsize=16)
fig.supylabel('Count',fontsize=16)

fig.savefig(out_f+'scm_pag.jpg',dpi=300, bbox_inches='tight')


###### create scatterplot

# calcualte mean projection strength for lab and singing mouse
proj = []

for i in range(6):
    df_t = bin[i]
    proj_d = df_t.sum()/df_t.shape[0]
    proj.append(proj_d)
    

mm_proj_means = (proj[0] + proj[1] + proj[2])/3
st_proj_means = (proj[3] + proj[4] + proj[5])/3

# scatterplot
x = np.log10(mm_proj_means)
y = np.log10(st_proj_means)
df = pd.DataFrame({'G':x, 'GA':y})
plt.figure(2)
plt.scatter(x, y)
plt.axline((0, 0), (1, 1),linestyle='--')
for i in range(df.shape[0]):
 plt.text(x=df.G[i]+0.1,y=df.GA[i]+0.1,s=df.index[i], 
          fontdict=dict(color='red',size=10))
plt.title('log10(projection probability)')
plt.xlabel('lab mouse')
plt.ylabel('singing mouse')
plt.savefig(out_f+'proj_prob_scatter.jpg',dpi=300, bbox_inches='tight')

###### calculate p-values

# aggregate sampled mouse projection probs
mm_probs_all = np.concatenate((mm_probs[0],mm_probs[1],mm_probs[2]))


# calculate p-value == ((number of mmus simulated >= steg observed mean) + 1)/((number of non-zeros)+1)
# added +1 to numerator and denominator of p-value due to finite sampling
p_values = []
for i in range(15):
    # 1. sum up number of simulations >= to singing mouse mean
    if st_proj_means[i] >= mm_proj_means[i]:
        above = (mm_probs_all[:,i] >= st_proj_means[i]).sum()
        p_value = (above+1)/(3000+1)
    else:
        below = (mm_probs_all[:,i] <= st_proj_means[i]).sum()
        p_value = (below+1)/(3000+1)
    p_values.append(p_value)

# create volcano plot w/ p-values
# calculate sums
# x-axis = effect size
x = (st_proj_means-mm_proj_means)/(mm_proj_means+st_proj_means)
y = -np.log10(p_values)

# restrict to regions of interest (i.e. not OMCi or ACAi/c)
if omc:
    x2 = pd.concat([x.take([1]),x[4:]])
    y2 = np.delete(y, [0,2,3])
else: # acc
    x2 = x[3:]
    y2 = np.delete(y, [0,1,2])

# create dataframe so can plot region labels for points
df = pd.DataFrame({'effect_size':x2, 'nlog10_p':y2})

plt.figure(3)
plt.scatter(x2,y2)
plt.axline((0, 0), (0, 1),linestyle='--')
plt.axline((0, 0), (1, 0),linestyle='--')
# plt.axline((0, -np.log10(0.05)), (1,  -np.log10(0.05)),linestyle='--', color='r', alpha=0.75)
plt.axline((0, -np.log10(0.01)), (1,  -np.log10(0.01)),linestyle='--', color='r', alpha=0.75)
plt.text(0.9, -np.log10(0.01)+.01, 'p<0.01', color='r', alpha=0.75)
plt.axline((0, -np.log10(0.001)), (1,  -np.log10(0.001)),linestyle='--', color='r', alpha=0.5)
plt.text(0.9, -np.log10(0.001)+.01, 'p<0.001', color='r', alpha=0.75)


for i in range(df.shape[0]):
 plt.text(x=df.effect_size[i]+0.01,y=df.nlog10_p[i]+0.01,s=df.index[i], 
          fontdict=dict(color='black',size=10))
plt.title('Volcano Plot')
plt.xlabel('(st_mean-mm_mean)/(mm_mean+st_mean)')
plt.ylabel('-log10(p-value)')
plt.savefig(out_f+'proj_prob_volcano.jpg',dpi=300, bbox_inches='tight')
