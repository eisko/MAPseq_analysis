# MAPseq_processing.py
# 230714

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from M194_M220_metadata import *
from scipy import stats

from itertools import combinations, chain
from upsetplot import from_memberships
from math import comb
import math

from statsmodels.distributions.empirical_distribution import ECDF # for generating cdfs


def clean_up_data(df_dirty, to_drop = ['OB', 'ACAi', 'ACAc', 'HIP'], inj_site="OMCi"):
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
    nodes = replaced.drop([inj_site], axis=1).sum(axis=1)
    n_idx = nodes > 0 # non-zero projecting neurons
    clean = replaced[n_idx]
    
    return clean


def dfs_preprocess_counts(df_list, drop=["OMCi", "type"]):
    """INPUT: Takes list of dataframe(s) that are normalized counts (counts normalized to spike-in RNA),
                disregards values in columns specified by `drop`
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


def sort_by_celltype(proj, it_areas=["OMCc", "AUD", "STR"], ct_areas=["TH"], pt_areas=["AMY","HY","SNr","SCm","PG","PAG","BS"]):
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
    # ds_pt = ds_pt.sort_values(['PAG','AMY'], ascending=False)
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
    ds_nth = ds_nth.sort_values(it_areas,ascending=False)
    ds_nth['type'] = "IT"

    # combine IT and CT cells
    ds_npt = pd.concat([ds_nth, ds_th])

    # combine IT/CT and PT cells
    sorted = pd.concat([ds_npt,ds_pt],ignore_index=True)

    sorted=sorted.reset_index(drop=True)
    
    return sorted


def dfs_to_node_proportions(df_list, drop=["OMCi", "type"], keep=None, cell_type=None, meta=metadata, inj_site="OMCi"):
    """Output dataframe of proportions of each node degree in format that can be plotted with seaborn

    Args:
        df_list (list): 
            - List of dataframes of neurons/BC by areas
        drop (list, optional): 
            - Defaults to ["OMCi", "type"]
            - list of areas/columns to drop before calculating proportions
        keep (list, optionl):
            - if present, only keep selected columns
            - Defaults to None.
        cell_type (string, optional): 
            - Specify cell types in df, either IT, CT or PT
            - Defaults to None

    Returns:
        plot_df (pandas_dataframe):
            - returns dataframe in format for seaborn plotting
            - columns = node_degree, node degree proportion, total cell count, percentage, and other metadata
    """

    plot_df = pd.DataFrame(columns=["node_degree", "proportion", "count", "percentage", "mice", "species", "dataset"])

    if cell_type == "IT":
        drop = [inj_site, 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = [inj_site,inj_site[:-1]+"c", 'AUD']
    
    if keep:
        drop = [] # if only selecting few columns, don't need drop

    mice = meta["mice"]
    species = meta["species"]
    dataset = meta["dataset"]

    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        if keep:
            df = df.loc[:, keep]
        nodes = df.sum(axis=1)
        node_counts = nodes.value_counts().sort_index()
        node_proportion = node_counts/node_counts.sum()
        node_percentage = node_proportion*100
        # total = node_counts.sum()
        df_add = pd.DataFrame({"node_degree":node_counts.index.values, "proportion":node_proportion.values, 
        "count":node_counts, "percentage":node_percentage.values, "mice":mice[i], "species":species[i], "dataset":dataset[i]})
        plot_df = pd.concat([plot_df, df_add])
    
    return plot_df

def dfs_to_proportions(df_list, drop=["OMCi", "type"], keep=None, cell_type=None, meta=metadata, inj_site="OMCi"):
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
        drop = [inj_site, 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = [inj_site,inj_site[:-1]+"c", 'AUD', "STR"]

    if keep:
        drop = []

    mice = meta["mice"]
    species = meta["species"]
    dataset = meta["dataset"]

    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        if keep:
            df = df.loc[:, keep] # just subset keep columns
        bc_sum = df.sum()
        proportion = bc_sum/df.shape[0]
        df_add = pd.DataFrame({"area":proportion.index.values, "proportion":proportion.values, 
        "mice":mice[i], "species":species[i], "dataset":dataset[i]})
        plot_df = pd.concat([plot_df, df_add])
    
    return plot_df

def dfs_to_total_BCs(df_list, drop=['type'], keep=None, cell_type=None, meta=metadata, inj_site="OMCi"):
    """Output dataframe of proportions in format that can be plotted with seaborn. Not for binary data!

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

    plot_df = pd.DataFrame(columns=["area", "BC_total", "mice", "species", "dataset"])

    if cell_type == "IT":
        drop = [inj_site, 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = [inj_site,inj_site[:-1]+"c", 'AUD', "STR"]

    if keep:
        drop = []

    mice = meta["mice"]
    species = meta["species"]
    dataset = meta["dataset"]

    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        if keep:
            df = df.loc[:, keep] # just subset keep columns
        bc_sum = df.sum()
        # proportion = bc_sum/df.shape[0]
        df_add = pd.DataFrame({"area":bc_sum.index.values, "BC_total":bc_sum.values, 
        "mice":mice[i], "species":species[i], "dataset":dataset[i]})
        plot_df = pd.concat([plot_df, df_add])
    
    return plot_df

def dfs_to_mean_BC_per_neuron(df_list, drop=['type'], keep=None, cell_type=None, meta=metadata, inj_site="OMCi"):
    """Output dataframe of mean BC/neuron per area in format that can be plotted with seaborn. Not for binary data!

    Args:
        df_list (list): 
            - List of dataframes of neurons/BC by areas
        drop (list, optional): 
            - Defaults to ["type"]
            - list of areas/columns to drop before calculating proportions
        cell_type (string, optional): 
            - Specify cell types in df, either IT, CT or PT
            - Defaults to None

    Returns:
        plot_df (pandas_dataframe):
            - returns dataframe in format for seaborn plotting
            - columns = areas, and other metadata
    """

    plot_df = pd.DataFrame(columns=["area", "mean_BC_neuron", "mice", "species", "dataset"])

    if cell_type == "IT":
        drop = [inj_site, 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = [inj_site,inj_site[:-1]+"c", 'AUD', "STR"]

    if keep:
        drop = []

    mice = meta["mice"]
    species = meta["species"]
    dataset = meta["dataset"]

    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        if keep:
            df = df.loc[:, keep] # just subset keep columns

        areas = []
        BC_neuron = []
        for area in df.columns:
            areas.append(area)
            idx = df[area] > 0
            BC_neuron.append(df[idx][area].mean())


        df_add = pd.DataFrame({"area":areas, "mean_BC_neuron":BC_neuron, 
        "mice":mice[i], "species":species[i], "dataset":dataset[i]})
        plot_df = pd.concat([plot_df, df_add])
    
    return plot_df

def proportion_ttest(df, sp1="MMus", sp2="STeg", to_plot='proportion'):
    """output dataframe based on comparison of species proportional means
        output dataframe can be used for making volcano plot

    Args:
        df (pd.DataFrame): output of dfs_to_proportions
    """

    areas = sorted(df['area'].unique())

    # for area in areas:
    #     area_df = df[df['area']==area]
    #     mean = df.groupby('area', sort = False, as_index=False)['proportion'].mean()

    # sp1
    sp1_df = df[df["species"]==sp1]
    sp1_array = sp1_df.pivot(columns='mice', values=to_plot, index='area').values

    sp2_df = df[df["species"]==sp2]
    sp2_array = sp2_df.pivot(columns='mice', values=to_plot, index='area').values

    results = stats.ttest_ind(sp1_array, sp2_array, axis=1)
    p_vals = results[1]
    plot = pd.DataFrame({"area":areas, "p-value":p_vals})
    plot[sp1+"_mean"] = sp1_array.mean(axis=1)
    plot[sp2+"_mean"] = sp2_array.mean(axis=1)
    # plot["effect_size"] = (plot["st_mean"]-plot["mm_mean"]) / (plot["st_mean"] + plot["mm_mean"]) # modulation index
    plot["fold_change"] = plot[sp2+"_mean"]/(plot[sp1+"_mean"])
    plot["log2_fc"] = np.log2(plot["fold_change"])
    plot["nlog10_p"] = -np.log10(plot["p-value"])
    plot["p<0.05"] = plot["p-value"]<0.05

    return(plot)

def calc_PAB(df, drop=["OMCi", "type"], cell_type=None, inj_site="OMCi"):
    """Calculate probability of neuron projecting to A given it projects to B
        Output is array that can be turned into heatmap, and list of areas to use as axis labels

    Args:
        df (pd.DataFrame): Binary df, BC x area
        drop (list, optional): List of areas to not include in PAB. Defaults to ["OMCi", "type"].
    """

    if cell_type == "IT":
        drop = [inj_site, 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = [inj_site,inj_site[:-1]+"c", 'AUD']

    drop_df = df.drop(drop, axis=1)
    areas = drop_df.columns

    PAB = np.zeros((len(areas), len(areas)))

    for i in range(len(areas)):
        for j in range(len(areas)):
            areai = areas[i]
            areaj = areas[j]
            if drop_df[areaj].sum() != 0: # create conditional so avoid divide by zero

                union = drop_df[areai] + drop_df[areaj]
                overlap = union==2
                n_overlap = overlap.sum()
                total = drop_df[areaj].sum()
                PAB[i,j] = n_overlap/total
    return(PAB, areas)

def df_to_motif_proportion(df, areas, proportion=True):
    """Plot upset plot based on given data and area list

    Args:
        df (pd.DataFrame): df containing bc x area
        areas (list): List of areas (str) to be plotted
    """

    # generate all combinations of areas in true/false list
    area_comb = []
    for i in range(len(areas)):
        n = i+1
        area_comb.append(list(combinations(areas, n)))

    area_comb_list = list(chain.from_iterable(area_comb)) # flatten list
    memberships = from_memberships(area_comb_list) # generate true/false
    area_comb_TF = memberships.index.values # extract array w/ true/false values
    area_comb_names = memberships.index.names # get order of areas

    # calculate number of neurons of each motif
    comb_count = []
    for tf in area_comb_TF:
        neurons = df
        for i in range(len(area_comb_names)):
            # subset dataset on presence/absence of proj
            neurons = neurons[neurons[area_comb_names[i]]==tf[i]]
        if proportion: # return proportion
            comb_count.append(neurons.shape[0]/df.shape[0])
        else: # return count
            comb_count.append(neurons.shape[0])

    plot_s = from_memberships(area_comb_list, data=comb_count)

    return(plot_s)

def fold_change_calc(df_type, meta=metadata, drop=["OMCi","type"], inj_site="OMCi"):
    """Take df_list with cells label by type, calculate fold change b/w 2 groups.
        return dataframe to be plotted by fold change and colored by cell type

    Args:
        df_type (list): List of bc x area
        meta (DataFrame): dataframe of metadata of df_list
    """

    # get it cells
    cell_type = ["IT", "PT"]
    plot_fin = []
    for c in cell_type:
        df_ct = [df[df["type"]==c].drop("type", axis=1) for df in df_type]
        plot_ct = dfs_to_proportions(df_ct, cell_type=c, drop=drop, inj_site=inj_site)
        vplot_ct = proportion_ttest(plot_ct)
        # exclude str and th
        vplot_ct = vplot_ct[~(vplot_ct['area']=="STR") & ~(vplot_ct['area']=="TH")]
        vplot_ct['type'] = c
        plot_fin.append(vplot_ct)

    plot = pd.concat(plot_fin)
    plot = plot.sort_values("log2_fc", ascending=False).reset_index(drop=True)
    return plot

    
def stvmm_calc_stats(data, to_plot="proportion", species=["STeg", "MMus"]):
    """calculate statistics on to_plot column per species

    Args:
        data (pd.DataFrame): _description_
        to_plot (str, optional): Column to calculate statistics on. Defaults to "proportion".
    """

    # separate by species
    stats_sp = []
    for sp in species:
        data_sp = data[data["species"]==sp]
        agg_sp = data_sp.groupby(['area'])[to_plot].agg(['mean', 'count', 'std', 'sem'])

        ci95 = []
        for i in agg_sp.index:
            c = agg_sp.loc[i, 'count']
            sd = agg_sp.loc[i, 'std']
            ci95.append(1.96*sd/math.sqrt(c))
        agg_sp['ci95'] = ci95

        agg_sp['species'] = sp

        # change area from index to column
        agg_sp = agg_sp.reset_index()

        # assign cell type
        for i in range(agg_sp.shape[0]):
            area = agg_sp.loc[i, 'area']
            area_df = data_sp[data_sp['area']==area].reset_index()
            agg_sp.loc[i, 'type'] = area_df.loc[0, 'type']

        stats_sp.append(agg_sp)

    return(pd.concat(stats_sp))

def stvmm_calc_ttest(data, sp1="MMus", sp2="STeg", to_plot='proportion'):
    """Given dataset w/ labeled cell type and species, calculate ttest p-values b/w species replicates

    Args:
        data (DataFrame): _description_
    """
    

    # separate by cell type
    data_it = data[data["type"]=="IT"]
    data_pt = data[data["type"]=="PT"]

    # # calculate means
    # st_mean = data_st.groupby("area").mean(numeric_only=True)
    # mm_mean = data_mm.groupby("area").mean(numeric_only=True)

    
    it_tt = proportion_ttest(data_it, sp1=sp1, sp2=sp2, to_plot=to_plot)
    it_tt['type'] = "IT"
    pt_tt = proportion_ttest(data_pt, sp1=sp1, sp2=sp2, to_plot=to_plot)
    pt_tt['type'] = "PT"

    omc_tt = pd.concat([it_tt, pt_tt])
    omc_tt['p<0.05'] = omc_tt['p-value']<0.05
    omc_tt = omc_tt.reset_index(drop=True)
    
    return(omc_tt)

def sample_mm_all(data, metadata=metadata, random_state=10):
    """Given list of dataframe, sample from combined lab mouse cells (without replacement), in equivalent numbers
    to singing mouse cells/brain. Make sure that each simulated 'brain' does not have overlapping/reused cells.

    Args:
        data (list): List of pandas dataframes where each row is a different cell/neuron
        metadata (DataFrame, optional): Metadata used to determine which df in data is lab/singing mouse. 
                                        Defaults to metadata.
        random_state (int, optional): Set random state to use for repeatable sampling.
                                        Defaults to 10.
    """

    mm_all = [data[i] for i in range(metadata.shape[0]) if metadata.loc[i,'species']=="MMus"]
    mm_all = pd.concat(mm_all).reset_index(drop=True)

    # print("mm_all.shape[0]", mm_all.shape[0])
    mm_pool = mm_all.copy()
    mm_samp = []

    for i in range(metadata.shape[0]):
        if metadata.loc[i,'species'] == "STeg":
            n = data[i].shape[0]
            int = mm_pool.sample(n, random_state=10)
            idx = int.index

            mm_samp.append(int.reset_index(drop=True))

            # update mm_pool so don't resample neurons
            # print("sampled:", len(idx))
            mm_pool = mm_pool.drop(idx)
            # print("update mm_pool.shape[0]", mm_pool.shape[0])

    return(mm_samp)

def proportion_comparisons(data, species1="MMus", species2="STeg", label="inter"):
    """Given output of dfs_to_proportions calculate differences between individual points among/within species

    Args:
        data (DataFrame): pd.DataFrame, output of dfs_to_proportions()
        group1 (str, optional): Species to compare to. Defaults to "MMus".
        group2 (str, optional): Other species to compare. Defaults to "STeg".
        label (str, optional): Label of intra/inter comparison. Defautls to "inter".
    """

    df1 = data[data['species']==species1]
    df2 = data[data['species']==species2]

    areas = data['area'].unique()

    out_df = pd.DataFrame(columns=['area', 'prop_diff', 'species', 'mice', 'label'])

    for area in areas:
        
        df1a = df1[df1['area']==area].reset_index(drop=True)
        df2a = df2[df2['area']==area].reset_index(drop=True)
        
        for a in range(df1a.shape[0]):
            for b in range(df2a.shape[0]):
                diff = df1a.loc[a, 'proportion'] - df2a.loc[b, 'proportion']
                abs_diff = math.sqrt(diff**2)
                row = [area, abs_diff, species1+"_"+species2, df1a.loc[a, 'mice']+"_"+df2a.loc[b,'mice'], label]
                i = out_df.shape[0]
                out_df.loc[i, :] = row
                
    return(out_df)


def dfs_to_cdf(df_list, plot_areas, resolution=1000, metadata=metadata):
    """Takes in list of DFs of count(N) data and returns dataframe w/ cdf data that can be plotted.
    Returned Dataframe includes metadata

    Args:
        df_list (list): list of DataFrames of count(N) data.
        plot_areas (list): List of strings of areas to calculate cdfs.
        resolution (int, optional): Used to determine resolution of cdf line. Defaults to 1000.
        medatadata (df, optional): Metadata where row corresponds to df_list indices. Defaults to metadata.
    """

    # combine all DFs into one df labelled w/ metadata
    all_bc = pd.DataFrame(columns=list(df_list[0].columns)+["mice", "species", "dataset"])
    for i in range(metadata.shape[0]):
        df = df_list[i].copy(deep=True)
        df['mice'] = metadata.loc[i, 'mice']
        df['species'] = metadata.loc[i, "species"]
        df['dataset'] = metadata.loc[i, "dataset"]
        all_bc = pd.concat([all_bc, df])

    all_bc = all_bc.reset_index(drop=True)

    cdf_df = pd.DataFrame(columns=["x", "cdf", "mice", "species", "dataset", "area"])

    # calculate cdf by area, then add by mouse
    for area in plot_areas:
        # just use nonzero BC
        area_idx = all_bc[area] > 0
        area_bc = all_bc.loc[area_idx, [area, "mice", "species", "dataset"]]
        # get min/max for each area to set cdf bounds
        area_min = area_bc[area].min()
        area_max = area_bc[area].max()

        for i in range(metadata.shape[0]):
            micei = metadata.loc[i, 'mice']
            mice_bc = area_bc[area_bc['mice']==micei]
            if mice_bc[area].sum()==0:
                print("NO BARCODES, cannot compute ECDF for", area, metadata.loc[i,'mice'])
            else:
                # print(area, metadata.loc[i,"mice"])
                ecdf = ECDF(mice_bc[area])
                x = np.logspace(np.log10(area_min), np.log10(area_max), num=resolution)
                y = ecdf(x)

                int = pd.DataFrame({"x":x, "cdf":y, "mice":metadata.loc[i,"mice"], "species":metadata.loc[i,"species"], 
                                    "dataset":metadata.loc[i,"dataset"], "area":area})
                cdf_df = pd.concat([cdf_df, int])

    return(cdf_df)

def dfs_to_medians(df_list, drop=["AOMCi", "POMCi", "ACAi", "ACAc", "OB", "HIP", "inj_site", 
                                  'L1_ctl', 'H2O_inj_ctl', 'H2O_targ_ctl'], 
                   keep=None, cell_type=None, meta=metadata, inj_site="OMCi"):
    """Output dataframe of medians in format that can be plotted with seaborn

    Args:
        df_list (list): 
            - List of dataframes (normalized counts) of neurons/BC by areas
        drop (list, optional): 
            - Defaults to ["AOMCi", "POMCi", "inj_site", 'L1_ctl', 'H2O_inj_ctl', 'H2O_targ_ctl']
            - list of areas/columns to drop before calculating proportions
        cell_type (string, optional): 
            - Specify cell types in df, either IT, CT or PT
            - Defaults to None

    Returns:
        plot_df (pandas_dataframe):
            - returns dataframe in format for seaborn plotting
            - columns = areas, and other metadata
    """

    plot_df = pd.DataFrame(columns=["area", "median", "mice", "species", "dataset"])

    if cell_type == "IT":
        drop = [inj_site, 'TH', 'HY', 'AMY', 'SNr', 'SCm', 'PG',
       'PAG', 'BS']
    elif cell_type == "PT":
        drop = [inj_site,inj_site[:-1]+"c", 'AUD', "STR"]

    if keep:
        drop = []

    mice = meta["mice"]
    species = meta["species"]
    dataset = meta["dataset"]

    for i in range(len(df_list)):
        df = df_list[i].drop(drop, axis=1)
        if keep:
            df = df.loc[:, keep] # just subset keep columns

        # only keep nonzero BC to calculate median
        df.replace(0, np.nan, inplace=True)
        medians = df.median(axis=0)
        df_add = pd.DataFrame({"area":medians.index.values, "median":medians.values, 
        "mice":mice[i], "species":species[i], "dataset":dataset[i]})
        plot_df = pd.concat([plot_df, df_add])
    
    return plot_df