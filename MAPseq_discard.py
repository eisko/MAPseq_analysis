# discarded functions from plotting/processing scripts

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from M194_M220_metadata import *
from colormaps import *
from MAPseq_processing import *
from matplotlib.colors import LogNorm
import matplotlib.lines as mlines # needed for custom legend
from matplotlib.patches import Patch # needed for custom legend
from scipy import stats
import matplotlib.patches as mpatches  # Import patches to create custom legend markers
import math
from matplotlib.lines import Line2D # for custom legend
from matplotlib.markers import MarkerStyle # used to plot open circles


# for upset plots
import upsetplot

def single_neuron_heatmap(df, neuron, figsize=(6.4, 0.5), label=None, 
                          sort_by=['type'], drop=["OMCi", "type"], cmap=orange_cmp):
    """_summary_

    Args:
        neuron (int): index of neuron to plot
        figsize (tuple, optional): Fig size to look right. Defaults to (6.4, 0.5).
        title (string, optional): title. Defaults to None.
    """
    fig = plt.figure(figsize=figsize)

    plot = df.replace({"IT":0.25, "CT":0.5, "PT":0.75})
    plot = plot.sort_values(by=sort_by).reset_index(drop=True)

    ineuron = plot.iloc[neuron,:]
    plotn = pd.DataFrame(ineuron).T
    sns.heatmap(plotn.drop(drop, axis=1), cmap=cmap, cbar=False)
    plt.gca().get_yaxis().set_visible(False)
    plt.text(-0.3, 0.5, label, va="center_baseline", size=12)

    return(fig)

def single_neuron_line(df, neuron, figsize=(6.4, 0.5), label=None, ylim=1400,
                          sort_by=['type'], drop=["OMCi", "type"], cmap=orange_cmp):
    """_summary_

    Args:
        df (dataframe): must be normalized count data
        neuron (int): index of neuron to plot
        figsize (tuple, optional): _description_. Defaults to (6.4, 0.5).
        label (string, optional): _description_. Defaults to None.
        sort_by (list, optional): _description_. Defaults to ['type'].
        drop (list, optional): _description_. Defaults to ["OMCi", "type"].
        cmap (colormap, optional): _description_. Defaults to orange_cmp.
    """
    fig = plt.figure(figsize=figsize)

    plot = df.copy()

    plot = df.replace({"IT":0.25, "CT":0.5, "PT":0.75})
    plot = plot.sort_values(by=sort_by).reset_index(drop=True)
    plot = plot.drop(drop, axis=1)


    ineuron = plot.iloc[neuron,:]
    plotn = pd.DataFrame(ineuron).T

    areas = plotn.columns.values
    values = plotn.iloc[0].values
    values = values/values.max() # row normalized
    plt.plot(areas, values, color=cmap.colors[255])
    plt.text(-2, 0.55, label, va="center_baseline")
    # plt.ylim(0,ylim)
    
    return(fig)

def single_neuron_plot(df, neuron, figsize=(6.4, 0.5), label=None, ylim=None,
                          sort_by=['type'], drop=["OMCi", "type"], cmap=orange_cmp,
                          kind="bar", ylog=False, row_norm=True):
    """Plot bar/line/heatmap plot of single neuron from n=1000 random sample neurons.
        Used for figure 2 to give example neurons of each cell type in heatmap.

    Args:
        df (dataframe): must be normalized count data
        neuron (int): index of neuron to plot
        figsize (tuple, optional): _description_. Defaults to (6.4, 0.5).
        label (string, optional): _description_. Defaults to None.
        sort_by (list, optional): _description_. Defaults to ['type'].
        drop (list, optional): _description_. Defaults to ["OMCi", "type"].
        cmap (colormap, optional): _description_. Defaults to orange_cmp.
        kind (str, optional): What kind of plot to make, can be "bar", "line", or "heatmap". Defaults to "bar".
        ylog (boolean, optional): Whether to plot y axis on log scale or not. Defautls to False.
        row_norm (boolean, optional): Whether to row normalize the expression values before plotting. Defaults to True.
    """
    fig = plt.figure(figsize=figsize)

    plot = df.copy()

    plot = df.replace({"IT":0.25, "CT":0.5, "PT":0.75})
    plot = plot.sort_values(by=sort_by).reset_index(drop=True)
    plot = plot.drop(drop, axis=1)


    ineuron = plot.iloc[neuron,:]
    plotn = pd.DataFrame(ineuron).T

    areas = plotn.columns.values
    values = plotn.iloc[0].values

    # row normalize if desired
    if row_norm:
        values = values/values.max() # row normalized

    if kind=="bar":
        plt.bar(areas, values, color=cmap.colors[255])
    elif kind=="line":
        plt.plot(areas, values, color=cmap.colors[255])
    elif kind=="heatmap":
        sns.heatmap(plotn.drop(drop, axis=1), cmap=cmap, cbar=False)
        plt.gca().get_yaxis().set_visible(False)

    plt.text(-2, 0.55, label, va="center_baseline")

    # set yaxis limits if given
    if ylim:
        plt.ylim(ylim)

    # set y axis on log scale if given
    if ylog:
        plt.yscale("log")
    
    return(fig)

def proportion_polar_plot(df_list, plot_individuals=False, title=None,
                          drop=["OMCi","STR", "TH", "type"], keep=None, cell_type=None, 
                          meta=metadata, log_norm=True, inj_site="OMCi"): # these arguments for proportion fxn
    """Plot polar plot where have mean +/- sem and/or indiviudal mice in circular histogram

    Args:
        df_list (list)
            * list of pandas dataframes in BC x area format
        plot_individuals (bool, optional): 
            * Determine whether to plot lines for individual mice (vs. summary graph). 
            * Defaults to False.
        drop (list, optional): 
            * list of brain areas NOT to include when calculating proportion, input to 
                dfs_to_proportions function
            * Defaults to ["OMCi","type"].
        keep (list, optionl):
            * if present, only keep selected columns
            * Defaults to None.
        cell_type (string, optional): 
            * specify whether looking at all cells vs. just IT/CT/PT
            * Defaults to None. inputs can be "IT", "CT", or "PT"
        meta (DataFrame, optional): 
            * contains metadata from df_list, each row corresponds to element in df_list
            * Defaults to metadata. - i.e. all 12 samples
    """
    # calcualte proportion of neurons project to each area of interest
    plot_df = dfs_to_proportions(df_list, drop=drop, keep=keep, cell_type=cell_type, meta=meta, inj_site=inj_site)

    # # set-up angles used to plot
    N = plot_df.area.unique().shape[0] # number of areas to plot
    angles = [n / float(N) * 2 * np.pi for n in range(N)] # in radians
    angles.append(2*np.pi) # extra angle ensures connected circle

    sp_list = plot_df.species.unique() # list of species in data

    fig = plt.subplot(projection='polar')

    for sp in sp_list:
        df = plot_df[plot_df["species"] == sp]

        if sp=="MMus":
            sp_cmp = blue_cmp
        elif sp=="STeg":
            sp_cmp = orange_cmp

        # plot individual mice
        mice = meta[meta["species"] == sp].mice

        if plot_individuals:
            for i in range(len(mice)):
                int = df[df['mice']== mice[i]]
                values = int.proportion
                values.loc[len(values)+1] = values.loc[0]
                if log_norm:
                    values = np.log10(values)
                plt.polar(angles, values, color=sp_cmp.colors[50], label=mice[i])

       

        mean = df.groupby('area', sort = False, as_index=False)['proportion'].mean()
        sem = df.groupby('area', sort = False, as_index=False)['proportion'].sem()
        mean_sem = mean['proportion'] + sem['proportion']
        sem_mean = mean['proportion'] - sem['proportion']
        mean = mean['proportion']

        # add first entry to end to complete polar cirle
        mean.loc[len(mean)+1] = mean.loc[0]
        mean_sem.loc[len(mean_sem)+1] = mean_sem.loc[0]
        sem_mean.loc[len(sem_mean)+1] = sem_mean.loc[0]

        # put proportions on log scale
        if log_norm:
            mean = np.log10(mean)
            mean_sem = np.log10(mean_sem)
            sem_mean = np.log10(sem_mean)

        plt.polar(angles, mean, color=sp_cmp.colors[255], label = 'mean')
        plt.polar(angles, mean_sem, color=sp_cmp.colors[100], linestyle = '--', linewidth=0.9, label = 'sem')
        plt.polar(angles, sem_mean, color=sp_cmp.colors[100], linestyle = '--', linewidth=0.9)
    
    plt.xticks(angles[:-1], plot_df.area.unique())
    if log_norm:
        plt.yticks([-2,-1.5,-1,-0.5],['$10^{-2}$','$10^{-1.5}$','$10^{-1}$','$10^{-0.5}$'])
    plt.legend(bbox_to_anchor=(1.3, 1.05))
    plt.title(title)

    return(fig)

def area_proportion_dot_plot(data, area=None, title=None, err="se", add_legend=True,
                              to_plot="proportion", ylim=(0), resample=False):
    """Given area proportions labeled by area and species, plot dot plots of area proportions.

    Args:
        df (DataFrame): Output from dfs_to_proportion
        area (str): area to plot
        err (str): error bar to plot for sns.pointplot(), can be "ci", "pi", "se", or "sd". Defaults to "se".
        title (str): Title to apply to plot
        add_legend (bool, optional): Specify whether to include legend or not. Defaults to True.
        to_plot (str, optional): what column name to plot. Defaults to "propotion".
        ylim (int, optional): Lower bound for yaxis. Defaults to (0).
        resample (bool, optional): Determines whether to plot resampled propotions or not. Defaults to False.
    """

    if area:
        df = data[data["area"]==area]
    else:
        df = data.copy()

    # means = area_df.groupby('species')['proportion'].mean() # need means for plotting lines

    fig, ax = plt.subplots()

    strip = sns.stripplot(data=df, x="species", y=to_plot, hue="species", size=10, ax=ax)
    # violin = sns.violinplot(area_df, x='species',y="proportion",
    #             split=True, hue ='species', inner = None, 
    #             palette="pastel",legend=False)
    point = sns.pointplot(data=df, x="species", y=to_plot, hue="species", units='mice', 
                          color='black', markers='+', ax=ax, errorbar=err) # plots mean and 95 confidence interval:

    # mm_line = mlines.Line2D([0, 1], [means["MMus"], means["MMus"]], color=blue_cmp.colors[150])
    # st_line = mlines.Line2D([0, 1], [means["STeg"], means["STeg"]], color=orange_cmp.colors[150])
    
    # ax.add_line(mm_line)
    # ax.add_line(st_line)

    plt.title(title, size=20)
    plt.ylim(ylim) # make sure y axis starts at 0
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if add_legend:
        legend = mlines.Line2D([], [], color="black", marker="+", linewidth=0, label="mean, "+err)
        plt.legend(handles=[legend], loc="lower right")
    else:
        plt.legend([],[], frameon=False)


    return(fig)





