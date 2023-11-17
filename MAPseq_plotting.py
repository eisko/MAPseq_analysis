# MAPseq_plotting
# 230714

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


def dot_bar_plot(df, title="", xaxis="Node Degree", yaxis="Normalized Frequency", 
                 hueaxis="Species", errorbar="se"):
    """
    Function to take pandas dataframe and plot individual values and mean/sem values
    Intent to use for plotting nodes by frequency (in fraction of neurons)

    Args:
        df (pandas.core.frame.DataFrame): pandas dataframe where rows are nodes and columns are:
         'Node Degreee', 'Normalized Frequency', 'Species', and 'mouse'
         - See output of df_to_nodes
        title (str): plot title
    """
    fig = plt.subplot()
    sns.stripplot(df, x=xaxis, y=yaxis, hue=hueaxis, dodge=True, jitter=False, size=3)
    t_ax = sns.barplot(df, x=xaxis, y=yaxis, hue=hueaxis, errorbar=errorbar, errwidth=1)
    for patch in t_ax.patches:
        clr = patch.get_facecolor()
        patch.set_edgecolor(clr)
        patch.set_facecolor((0,0,0,0))
    plt.setp(t_ax.patches, linewidth=1)
    plt.title(title, size=18)

    return(fig)

def individ_node_plot(df, title="", xaxis="Node Degree", yaxis="Normalized Frequency"):
    """Plot connected dots for individual mice colored by species

    Args:
        df (pandas dataframe): - should contain column called "Species" with data labelled
                                as "MMus" or "STeg"
                               - should also contain column called "Dataset w/ data labelled
                                as "M194" or "M220"
                               - should also contain column labelled "mouse" w/ mouse identiy
    """
    fig = plt.subplot()
    sns.scatterplot(df[df['Species']=="MMus"], x=xaxis, y=yaxis, style="Dataset")
    sns.lineplot(df[df['Species']=="MMus"], x=xaxis, y=yaxis, style="mouse", alpha=0.5)
    sns.scatterplot(df[df['Species']=="STeg"], x=xaxis, y=yaxis, style="Dataset", color="orange", legend=False)
    sns.lineplot(df[df['Species']=="STeg"], x=xaxis, y=yaxis, style="mouse", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    plt.text(2.75, 0.8, "MMus", color=sns.color_palette("tab10")[0]) # match blue to seaborn
    plt.text(2.75, 0.76, "STeg", color=sns.color_palette("tab10")[1]) # match orange to default seaborn colors
    plt.title(title)

    return(fig)

def sorted_heatmap(df, title=None, sort_by=['type'], sort_ascend=True, drop=['type'], nsample=None, 
                   random_state=10, norm=None, cmap=orange_cmp, cbar=False,
                   label_neurons=None, col_order=None):
    """_summary_

    Args:
        df (DataFrame): Dataframe to plot heatmap
        sort_by (list, optional): How to sort neurons. Defaults to ['type'].
        nsample (integer, optional): If prsent, down sample dataframe. Defaults to None.
        random_state (int, optional): If downsample, what random state to use. Defaults to 10.
        label_neurons (dict, option): Dictionary of label:index of neurons to label
        col_order (list, optional): Order to plot columns. Defaults to None.
    """

    if nsample:
        plot = df.sample(nsample, random_state=random_state)
    else:
        plot = df.copy()

    plot = plot.replace({"IT":0.25, "CT":0.5, "PT":0.75})
    plot = plot.sort_values(by=sort_by, ascending=sort_ascend)
    idx = plot.index
    plot = plot.reset_index(drop=True)

    # reorder cols if given
    if col_order:
        plot = plot[col_order]

    fig=plt.subplot()
    sns.heatmap(plot.drop(drop, axis=1), norm=norm, cmap=cmap, cbar=cbar)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(title)
    if label_neurons:
        for key in label_neurons.keys():
            plt.text(-0.3,label_neurons[key], key+"-", va="center_baseline", size=12)
    return(idx, fig)
    
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

def single_neuron_bar(df, neuron, figsize=(6.4, 0.5), label=None, ylim=1400,
                          sort_by=['type'], drop=["OMCi", "type"], cmap=orange_cmp, col_order=None):
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

    if col_order:
        plotn = plotn[col_order]

    areas = plotn.columns.values
    values = plotn.iloc[0].values
    values = values/values.max() # row normalized
    plt.bar(areas, values, color=cmap.colors[255])
    plt.text(-2, 0.55, label, va="center_baseline", size=12)
    # plt.ylim(0,ylim)

    # hide top and right axes
    ax = plt.gca()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    return(fig)

def single_neuron_plot(df, neuron, figsize=(6.4, 0.5), label=None, ylim=None,
                          sort_by=['type'], drop=["OMCi", "type"], cmap=orange_cmp,
                          kind="bar", ylog=False):
    """_summary_

    Args:
        df (dataframe): must be normalized count data
        neuron (int): index of neuron to plot
        figsize (tuple, optional): _description_. Defaults to (6.4, 0.5).
        label (string, optional): _description_. Defaults to None.
        sort_by (list, optional): _description_. Defaults to ['type'].
        drop (list, optional): _description_. Defaults to ["OMCi", "type"].
        cmap (colormap, optional): _description_. Defaults to orange_cmp.
        kind (str, optional): What kind of plot to make, can be "bar", "line", or "heatmap". Defaults to "bar".
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


# def area_proportion_dot_plot(data, area=None, title=None, err="se", add_legend=True,
#                               to_plot="proportion", ylim=(0), resample=False):
#     """Given area proportions labeled by area and species, plot dot plots of area proportions.

#     Args:
#         df (DataFrame): Output from dfs_to_proportion
#         area (str): area to plot
#         err (str): error bar to plot for sns.pointplot(), can be "ci", "pi", "se", or "sd". Defaults to "se".
#         title (str): Title to apply to plot
#         add_legend (bool, optional): Specify whether to include legend or not. Defaults to True.
#         to_plot (str, optional): what column name to plot. Defaults to "propotion".
#         ylim (int, optional): Lower bound for yaxis. Defaults to (0).
#         resample (bool, optional): Determines whether to plot resampled propotions or not. Defaults to False.
#     """

#     if area:
#         df = data[data["area"]==area]
#     else:
#         df = data.copy()

#     # means = area_df.groupby('species')['proportion'].mean() # need means for plotting lines

#     fig, ax = plt.subplots()

#     strip = sns.stripplot(data=df, x="species", y=to_plot, hue="species", size=10, ax=ax)
#     # violin = sns.violinplot(area_df, x='species',y="proportion",
#     #             split=True, hue ='species', inner = None, 
#     #             palette="pastel",legend=False)
#     point = sns.pointplot(data=df, x="species", y=to_plot, hue="species", units='mice', 
#                           color='black', markers='+', ax=ax, errorbar=err) # plots mean and 95 confidence interval:

#     # mm_line = mlines.Line2D([0, 1], [means["MMus"], means["MMus"]], color=blue_cmp.colors[150])
#     # st_line = mlines.Line2D([0, 1], [means["STeg"], means["STeg"]], color=orange_cmp.colors[150])
    
#     # ax.add_line(mm_line)
#     # ax.add_line(st_line)

#     plt.title(title, size=20)
#     plt.ylim(ylim) # make sure y axis starts at 0
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)

#     if add_legend:
#         legend = mlines.Line2D([], [], color="black", marker="+", linewidth=0, label="mean, "+err)
#         plt.legend(handles=[legend], loc="lower right")
#     else:
#         plt.legend([],[], frameon=False)


#     return(fig)

def dot_plot(data, area=None, title=None, err="se", add_legend=False,
                              to_plot="proportion", ylim=(0), fig_size=(3.5,3.5)):
    """Plot open/closed circle of value per area for data and resampled data.

    Args:
        data (DataFrame): Dataframe of proportions, included resampled data.
        area (str, optional): Area to plot. Defaults to None.
        title (str, optional): Title for plot. Defaults to None.
        err (str, optional): Error to plot, can be "ci", "pi", "se", or "sd". Defaults to "se".
        add_legend (bool, optional): Whether to add legend labeling mean/err. Defaults to False.
        to_plot (str, optional): Column to plot. Defaults to "proportion".
        ylim (tuple, optional): lower bound for yaxis. Defaults to (0).
    """

    if area:
        df = data[data["area"]==area]
    else:
        df = data.copy()


    # add column for xaxis plotting
    # df['xaxis'] = df['species'].replace({"MMus":0, "STeg":1, "MMus_resampled":2, "STeg_resampled":3})


    fig, ax = plt.subplots()

    # plot individual value
    sns.stripplot(data=df, x="species", y=to_plot, hue="species", size=10, ax=ax)
    # plot mean and error bar for each species
    sns.pointplot(data=df, x="species", y=to_plot, hue="species", units='mice', 
                          color='black', markers='+', errorbar=err, ax=ax) # plots mean and 95 confidence interval:
    # necessary to put mean/error markers on top
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")


    ax.set_xlabel("")

    plt.title(title, size=20)
    plt.ylim(ylim) # make sure y axis starts at 0
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0],fig_size[1])

    if add_legend:
        legend = mlines.Line2D([], [], color="black", marker="+", linewidth=0, label="mean, "+err)
        plt.legend(handles=[legend], loc="lower right")
    else:
        plt.legend([],[], frameon=False)


    return(fig)


def proportion_volcano_plot(df, title=None, labels="area", p_05=True, p_01=True, p_bf=None):
    """output volcano plot based on comparison of species proportional means

    Args:
        df (pd.DataFrame): output of proprotion_ttest
    """

    # areas = sorted(df['area'].unique())

    fig = plt.subplot()

    x=df.log2_fc
    y=df.nlog10_p

    plt.scatter(x,y, s=25)
    # plt.xlim([-1,1])
    # plt.ylim([-0.1,4])
    # plot 0 axes
    plt.axline((0, 0), (0, 1),linestyle='--', linewidth=0.5)
    plt.axline((0, 0), (1, 0),linestyle='--', linewidth=0.5)

    # p_05
    if p_05:
        plt.axline((0, -np.log10(0.05)), (1,  -np.log10(0.05)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
        plt.text(-0.1, -np.log10(0.05)+.015, 'p<0.05', color='r', alpha=0.75)
    if p_01:
        plt.axline((0, -np.log10(0.01)), (1,  -np.log10(0.01)),linestyle='--', color='r', alpha=0.5, linewidth=0.5)
        plt.text(-0.1, -np.log10(0.01)+.015, 'p<0.01', color='r', alpha=0.75)
    if p_bf:
        plt.axline((0, -np.log10(p_bf)), (1,  -np.log10(p_bf)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
        plt.text(-0.1, -np.log10(p_bf)+.015, 'p<bf_01', color='r', alpha=0.75)


    for i in range(df.shape[0]):
        plt.text(x=df.log2_fc[i]+0.01,y=df.nlog10_p[i]+0.01,s=df.loc[i, labels], 
            fontdict=dict(color='black',size=10))


    plt.title(title)
    plt.xlabel('log2(fold change)')
    plt.ylabel('-log10(p-value)')

    return(fig)

def plot_volcano(df, x="log2_fc", y="nlog10_p", title=None, labels="area", shape=None,
                 p_05=True, p_01=True, p_bf=None, xlim=(-2,2), legend_loc="upper left",
                 fig_size=(4,4)):
    """output volcano plot based on comparison of species proportional means

    Args:
        df (pd.DataFrame): output of proprotion_ttest
        x (str): column name to put on x axis
        y (str): column name to put on y axis
        title (str, optional): title of plot. Defaults to None
        labels (str, optional): column used to label points. Defaults to 'area'.
        shape (str, optional): Column used to determine shape (e.g. 'type'). Defaults to None.
        p_05 (bool, optionl): used to toggle p<05 line on/off. Defaults to Ture.
        p_01 (bool, optionl): used to toggle p<01 line on/off. Defaults to Ture.
        p_bf (float, optional): Boferoni correction cut-off to plot. Defaults to None.
        xlim (tuple, optional): tuple of numbers to set x-axis limits. Defaults to (-2,2).
    """

    # areas = sorted(df['area'].unique())

    fig = plt.subplot()

    marker_order = ['o', 'D', 'v']
    color_order = ["#2ca02c", "#9467bd"]
    if shape:
        nshapes = df[shape].unique()
        for i in range(nshapes.shape[0]):
            dfn = df[df[shape]==nshapes[i]]
            plt.scatter(dfn[x], dfn[y], label=nshapes[i],
                        marker=marker_order[i], s=25, c=color_order[i])
        

        # #  Create custom legend labels
        # point1 = line2D([0], [0], label=)

        # # Create custom legend markers using patches
        # circle_marker = mpatches.Patch(color='black', marker='o', label=legend_labels[0])
        # diamond_marker = mpatches.Patch(color='black', marker='D', label=legend_labels[1])

        # # Create a custom legend using handles and labels
        # plt.legend(handles=[circle_marker, diamond_marker], labels=legend_labels, loc='upper right')

    else:
        plt.scatter(df[x],df[y], s=25)


    # plt.xlim([-1,1])
    # plt.ylim([-0.1,4])
    # plot 0 axes
    plt.axline((0, 0), (0, 1),linestyle='--', linewidth=0.5)
    plt.axline((0, 0), (1, 0),linestyle='--', linewidth=0.5)

    # p_05
    if p_05:
        plt.axline((0, -np.log10(0.05)), (1,  -np.log10(0.05)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
        plt.text(-0.1, -np.log10(0.05)+.015, 'p<0.05', color='r', alpha=0.75)
    if p_01:
        plt.axline((0, -np.log10(0.01)), (1,  -np.log10(0.01)),linestyle='--', color='r', alpha=0.5, linewidth=0.5)
        plt.text(-0.1, -np.log10(0.01)+.015, 'p<0.01', color='r', alpha=0.75)
    if p_bf:
        plt.axline((0, -np.log10(p_bf)), (1,  -np.log10(p_bf)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
        plt.text(-0.1, -np.log10(p_bf)+.015, 'p<bf_01', color='r', alpha=0.75)


    for i in range(df.shape[0]):
        plt.text(x=df.loc[i,"log2_fc"]+0.01,y=df.loc[i,"nlog10_p"]+0.01,s=df.loc[i, labels], 
            fontdict=dict(color='black',size=10))


    plt.title(title, pad=12)
    # plt.xlabel('log2(fold change)')
    plt.xlabel('$log_{2}$($\dfrac{STeg}{MMus}$)')
    plt.ylabel('$-log_{10}(p\ value)$')

    if legend_loc:
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(handles=handles, loc=legend_loc)


    # apply axis limits
    plt.xlim(xlim)

    plt.rcParams.update({'font.size': 12})

    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0],fig_size[1])


    return(fig)


def proportion_node_stacked_bars(df, title=None):
    """given dataframe, plot stacked barchart of proportion of each node degree by species

    Args:
        df (pd.DataFrame): Output of dfs_to_node_proportions
        title (str, optional): title of plot. Defaults to None.
    """
    # fig = plt.subplot()

    plot_df = pd.DataFrame(columns=df[df["species"]=="MMus"]['node_degree'], index=["Mmus", "STeg"])
    plot_df.loc["Mmus",:] = df[df["species"]=="MMus"]["proportion"]
    plot_df.loc["STeg",:] = df[df["species"]=="STeg"]["proportion"]

    plot_df.plot(kind='bar', stacked=True)
    plt.legend(bbox_to_anchor=(1.05, 0.95), loc='upper left', borderaxespad=0, title="Node degree")
    plt.gca().set_aspect(5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.ylabel("proportion")
    plt.title(title)

    # return(fig)

def pab_heatmap(array, areas, title=None, cmap=None):
    """Make P(A|B) heatmap, give output from calc_PAB()
        return heatmap

    Args:
        array (np.Array): array of P(A|B) where columns=A and rows=B
        title (str, optional): Plot title. Defaults to None.
    """

    fig = plt.subplot()

    sns.heatmap(array, yticklabels=areas, xticklabels=areas, cmap=cmap, cbar_kws={'label': 'P(B|A)'})
    plt.ylabel("B")
    plt.xlabel("A")
    plt.title(title)

    return(fig)


def upset_plot(plot_s, title=None, suptitle=None, facecolor="tab:blue", shading_color="lightgray",
               ymin=0, ymax=0.5, sortby="input", sort_cat="input", with_lines=True, show_counts=False):
    """upset plot with specialized parameters

    Args:
        plot_s (Pandas series?): output of from_membership() from upsetplot package
        face_color (str, optional): Color for bars, only takes string colors. Defaults to "tab:blue".
        shading_color (str, optional): only takes strings. Defaults to "lightgray".
        ymin (float/int, optional): min for y axis. Defaults to 0.
        ymax (float, optional): max for y axis. Defaults to 0.5.
        sortby (str, optional): way to sort motifs, can be: {'cardinality', 'degree', 
                                '-cardinality', '-degree', 'input', '-input'}. Defaults to "input".
        sort_cat (str, optional): Way to sort categoreis, can be: ["cardinality", "-cardinality",
                                    "input", "-input"]. Defaults to "input".
        with_lines (bool, optional): whether to plot motif lines. Defaults to True.
        show_counts (bool, optional): whether plot counts or not. Defaults to False.
    """

    # fig = plt.subplot()
    upsetplot.plot(plot_s, facecolor=facecolor, shading_color=shading_color, sort_by=sortby,
                   sort_categories_by=sort_cat, with_lines=with_lines, show_counts=show_counts)
    plt.title(title, fontsize=18)
    plt.suptitle(suptitle, x=0.56, y=0.88, fontsize=10)
    plt.ylim(ymin, ymax)
    
    # return(fig)


def fold_change_ranked(plot, title=None, suptitle=None,
                       x="area", y="log2_fc", it_color="tan", pt_color="dimgrey"):
    
    fig = plt.subplot()
    cols = []
    for type in plot['type']:
        if type=="IT":
            cols.append("tan")
        else:
            cols.append("dimgrey")

    legend_elements = [Patch(facecolor=it_color, label='IT Cells'),
                    Patch(facecolor=pt_color, label='PT Cells')]

    sns.barplot(plot, x=x, y=y, palette=cols)
    plt.legend(handles=legend_elements)
    plt.suptitle(suptitle, size=18)
    plt.title(title, size=10)
    plt.ylabel("Log2(r'\frac{Steg}{\MMus}$')")
    return(fig)

def stvmm_area_scatter(data, title="", to_plot="proportion", log=False, 
                       err="sem", ax_limits=None,
                       sp1="STeg", sp2="MMus", line_up_limit=1):
    """Plots lab mouse v. singing mouse scatter w/ unity line

    Args:
        data (pandas.dataframe): output of calc_fluor
        to_plot (str, optional): Label of column in data to plot. Defaults to "proportion".
    """

    # separate by species
    data_st = data[data["species"]==sp1]
    data_mm = data[data["species"]==sp2]

    # calculate means
    st_mean = data_st.groupby("area").mean(numeric_only=True)
    mm_mean = data_mm.groupby("area").mean(numeric_only=True)

    # calculate error
    if err=="sem":
        st_err = data_st.groupby("area").sem(numeric_only=True)
        mm_err = data_mm.groupby("area").sem(numeric_only=True)
    elif err=="std":
        st_err = data_st.groupby("area").std(numeric_only=True)
        mm_err = data_mm.groupby("area").std(numeric_only=True)


    fig = plt.subplot()

    plt.errorbar(st_mean[to_plot], mm_mean[to_plot], 
            xerr=st_err[to_plot], fmt='|', color="orange")
    plt.errorbar(st_mean[to_plot], mm_mean[to_plot], 
            yerr=mm_err[to_plot], fmt='|')

    # add area labels
    labels = list(st_mean.index)
    for i in range(len(labels)):
        plt.annotate(labels[i], (st_mean[to_plot][i], mm_mean[to_plot][i]))
    
    # set x and y lims so that starts at 0,0
    if ax_limits:
        plt.xlim(ax_limits)
        plt.ylim(ax_limits)


    # adjust scale
    if log:
        plt.xscale("log")
        plt.yscale("log")

    # plot unity line
    x = np.linspace(0,line_up_limit, 5)
    y = x
    plt.plot(x, y, color='red', linestyle="--", linewidth=0.5)


    # add axis labels
    plt.xlabel(sp1+" "+to_plot, color="tab:orange")
    plt.ylabel(sp2+" "+to_plot, color="tab:blue")

    # add title
    plt.title(title)

    return(fig)

def stvmm_area_scatter_type(data, x="STeg", y="MMus", title="", log=False, 
                       err="sem", ax_limits=None, axis_label="Proportion", species=None):
    """Plots lab mouse v. singing moues scatter w/ unity line

    Args:
        data (pandas.dataframe): output of calc_fluor
        x (str): species to plot on x axis. Defaults to "STeg".
        y (str): species to plot on y axis. Defaults to "MMUs".
        title (Str): string to use for plot title. Defaults to "".
        log (bool): determine whether to plot axes on log scale. Defaults to False.
        err (str): Used to determine what error to plot. Can be "sem", "std", or "ci95
        ax_limits (list): list of 2 intergers (or tuple) to set axis limits. Defaults to None.
        axis_label (str): String to add to axis labels. Defaults to "Proportion".
        # species (str): used to determine whether plotting st v mm or sp vs sp. Defaults to None.
    """


    data_x = data[data['species']==x]
    x_stats = data_x.copy().reset_index()
    print(x_stats.shape)
    x_label = x
    data_y = data[data['species']==y]
    y_stats = data_y.copy().reset_index()
    print(y_stats.shape)
    y_label = y

    fig = plt.subplot()
    

    # plot errorbars
    plt.errorbar(x_stats['mean'], y_stats['mean'], 
            xerr=x_stats[err], fmt='|', color="black")
    plt.errorbar(x_stats['mean'], y_stats['mean'], 
            yerr=y_stats[err], fmt='|', color="black")

    # plot by cell type
    # it cells
    x_it = x_stats[x_stats['type']=="IT"]
    y_it = y_stats[y_stats['type']=="IT"]
    plt.scatter(x=x_it['mean'], y=y_it['mean'], marker="o", c="#2ca02c", label="IT")

    x_pt = x_stats[x_stats['type']=="PT"]
    y_pt = y_stats[y_stats['type']=="PT"]
    plt.scatter(x=x_pt['mean'], y=y_pt['mean'], marker="D", c="#9467bd", label="PT")

    # add area labels
    labels = list(x_stats['area'])
    for i in range(len(labels)):
        plt.annotate(labels[i], (x_stats['mean'][i]+0.01, y_stats['mean'][i]+0.01))
        
    
    # set x and y lims so that starts at 0,0
    if ax_limits:
        plt.xlim(ax_limits)
        plt.ylim(ax_limits)


    # adjust scale
    if log:
        plt.xscale("log")
        plt.yscale("log")

    # plot unity line
    x = np.linspace(0,1, 5)
    y = x
    plt.plot(x, y, color='grey', linestyle="--", linewidth=0.5)


    # add axis labels
    plt.xlabel(x_label+" "+axis_label)
    plt.ylabel(y_label+" "+axis_label)

    # add legend
    plt.legend()

    # add title
    plt.title(title)

    return(fig)

def stvmm_area_scatter_individ(data, data_prop, title="", log=False, 
                       err="sem", ax_limits=None, axis_label="Proportion", species=None):
    """Plots lab mouse v. singing moues scatter w/ unity line

    Args:
        data (pandas.dataframe): Input = proportions
        title (Str): string to use for plot title. Defaults to "".
        log (bool): determine whether to plot axes on log scale. Defaults to False.
        err (str): Used to determine what error to plot. Can be "sem", "std", or "ci95
        ax_limits (list): list of 2 intergers (or tuple) to set axis limits. Defaults to None.
        axis_label (str): String to add to axis labels. Defaults to "Proportion".
        species (str): used to determine whether plotting st v mm or sp vs sp. Defaults to None.
    """

    if species:
        data_sp = data[data['species']==species]
        x_stats = data_sp.copy().reset_index(drop=True)
        y_stats = data_sp.copy().reset_index(drop=True)
        x_label = species
        y_label = species

        x_indv = data_prop.copy()
        y_indv = data_prop.copy()

    else:
        x_stats = data[data['species']=="STeg"]
        x_indv = data_prop[data_prop['species']=="STeg"].copy()
        x_label = "Singing Mouse"
        y_stats = data[data['species']=="MMus"]
        y_indv = data_prop[data_prop['species']=="MMus"]
        y_label = "Lab Mouse"

    fig = plt.subplot()
    

    # # plot errorbars
    # plt.errorbar(x_stats['mean'], y_stats['mean'], 
    #         xerr=x_stats[err], fmt='|', color="black")
    # plt.errorbar(x_stats['mean'], y_stats['mean'], 
    #         yerr=y_stats[err], fmt='|', color="black")

    # plot individs
    # set colors for each area
    colors = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
    "#aec7e8",  # Light Blue
    "#ffbb78",  # Light Orange
    "#98df8a",  # Light Green
    "#ff9896",  # Light Red
    "#c5b0d5",  # Light Purple
    "#c49c94",  # Light Brown
    ]


    for i in range(x_stats['area'].shape[0]):
        area = x_stats.loc[i,'area']
        x_st_area = x_stats[x_stats['area']==area]
        y_st_area = y_stats[y_stats['area']==area]
        x_in_area = x_indv[x_indv['area']==area]
        nx = x_in_area.shape[0]

        y_in_area = y_indv[y_indv['area']==area]
        ny = y_in_area.shape[0]

        plt.scatter(x=[x_st_area['mean']]*nx, y=y_in_area['proportion'], color=colors[i])
        plt.scatter(x=x_in_area['proportion'], y=[y_st_area['mean']]*ny, color=colors[i])


    # plot by cell type
    # it cells
    x_it = x_stats[x_stats['type']=="IT"]
    y_it = y_stats[y_stats['type']=="IT"]

    x_pt = x_stats[x_stats['type']=="PT"]
    y_pt = y_stats[y_stats['type']=="PT"]

    # plot means
    plt.scatter(x=x_it['mean'], y=y_it['mean'], marker="o", c="black", label="IT")
    plt.scatter(x=x_pt['mean'], y=y_pt['mean'], marker="D", c="black", label="PT")


    # add area labels
    labels = list(x_stats['area'])
    for i in range(len(labels)):
        plt.annotate(labels[i], (x_stats.loc[i,'mean']+0.01, y_stats.loc[i,'mean']+0.01), c=colors[i])
        
    
    # set x and y lims so that starts at 0,0
    if ax_limits:
        plt.xlim(ax_limits)
        plt.ylim(ax_limits)


    # adjust scale
    if log:
        plt.xscale("log")
        plt.yscale("log")

    # plot unity line
    x = np.linspace(0,1, 5)
    y = x
    plt.plot(x, y, color='grey', linestyle="--", linewidth=0.5)


    # add axis labels
    plt.xlabel(x_label+" "+axis_label)
    plt.ylabel(y_label+" "+axis_label)

    # add legend
    plt.legend()

    # add title
    plt.title(title)

    return(fig)


def plot_cdf(data, plot_areas, log=True, title="", color_by="species", colors=[blue_cmp.colors[255], orange_cmp.colors[255]],
             individual=True, meta=metadata, legend=True, fig_size=(3,3)):
    """Takes in countN data and returns cdf plots

    Args:
        data (list): list of DataFrames of count(N), where each element in animal
        plot_areas (list): list of strings of areas to include in final output
        log (bool, optional): Whether to use log on axis scale. Defaults to True.
        title (str, optional): figure title. Defaults to "".
        color_by (str, optional): Can be "mice", "species", or "dataset", what to label as metadata. Defaults to "species".
        colors (list, optional): colors used to label cdfs. Defaults to [blue_cmp.colors[255], orange_cmp.colors[255]].
        individual (bool, optional): _description_. Defaults to True.
        meta (_type_, optional): _description_. Defaults to metadata.
    """


    # calculate ecdf per animal and put into dataframe
    cdf_df, foo = dfs_to_cdf(data, plot_areas=plot_areas, metadata=meta)

    # calculate number of axes needed
    n = math.ceil(len(plot_areas)/5) # round up divide by 4 = axs rows

    if len(plot_areas)==1:
        fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax_list = [ax]
    else:
        fig, axs = plt.subplots(n, 5, figsize=(20, 5*n))
        ax_list = axs.flat

    i = 0
    for ax in ax_list:

        if i < len(plot_areas):
            area = plot_areas[i]

            plot = cdf_df[cdf_df['area']==area]
            
            groups = plot[color_by].unique()

            plot_1 = plot[plot[color_by] ==groups[0]]
            plot_2 = plot[plot[color_by] ==groups[1]]

            if individual:
                sns.lineplot(plot_1, x="x", y="cdf", estimator=None, units="mice", color=colors[0], ax=ax) # plots individual mice
                sns.lineplot(plot_2, x="x", y="cdf", estimator=None, units="mice", color=colors[1], ax=ax) # plots individual mice
            else: 
                sns.lineplot(plot_1, x="x", y="cdf", ax=ax) # plots mean ci95
                sns.lineplot(plot_2, x="x", y="cdf", ax=ax) # plots mean ci95
            
            if log:
                ax.set_xscale("log")
            ax.set_xlabel("Normalized Counts")

            ax.set_title(area)
            i+=1
        else:
            ax.axis('off')

    # create cutom legend
    if legend:
        colors = [colors[0], colors[1]]
        lines = [Line2D([0], [0], color=c, linewidth=3) for c in colors]
        labels = [groups[0], groups[1]]
        fig.legend(lines,labels, bbox_to_anchor=(0.75, 0.935))

    # increase text size
    plt.rcParams.update({'font.size': 12})

    if title!="":
        plt.suptitle(title, y=0.93, size=20)
    elif title=="":
        plt.suptitle("By "+color_by, y=0.93, size=20)

    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0],fig_size[1])

    return(fig)


def fancy_upsetplot(data, plot_areas, reps=500, title="", subset=None, color="tab:orange",
                    ymax=0.6, plot_legend=True, plot_sim=True):
    """Given data of BCxAreas, generate upset plot with simulated/permutated data and counts

    Args:
        data (DataFrame): Dataframe of BC x areas
        plot_areas (list): List of string of areas to generate motifs
        reps (int): Number of simulations/permutations to run. Defaults to 500.
        title (str, optional): String to use as title. Defaults to "".
        subset (str, optional): area wish to subset motifs on. Defaults to None.
        color (str, optional): Color for bar graphs. Defaults to "tab:orange".
        ymax (float, optional): max on yaxis. Defaults to 0.6.
        plot_legend (bool, optional): Whether to plot legend or not. Defaults to True.
    """
    motifs, simulations = motif_simulation(data, plot_areas=plot_areas, reps=reps)

    # 5. plot motif means in upset plot

    # generate proportions for motifs (counts encounter problems in figure size???)
    motif_prop = df_to_motif_proportion(data, plot_areas, proportion=True)
    # generate motif counts
    motif_counts =df_to_motif_proportion(data, plot_areas, proportion=False)

    # if subset specified, extract motifs that project to area specified
    if subset:
        motif_areas = motif_prop.index.names
        subset_idx = motif_areas.index(subset)
        idx = [i for i, x in enumerate(motif_prop.index) if x[subset_idx]]
    else:
        idx = range(motif_prop.shape[0])


    # generate mean/std from permuted data
    means = simulations.mean(axis=0)
    x = list(range(len(idx)))
    y = means
    yerr = simulations.std(axis=0)


    upset_plot(motif_prop[idx],  facecolor=color, ymax=ymax)
    # upsetplot.add_catplot(kind="point", y)

    # plot simulation mean and std
    if plot_sim:
        plt.scatter(x,y[idx], color="black", marker="_", s=50, zorder=10)
        plt.errorbar(x, y[idx], 
                    yerr=yerr[idx], fmt='|', color="black", zorder=11)

    # label counts for each motif
    for i in x:
        plt.text(x=i, y=motif_prop[idx][i]+0.04, s=str(motif_counts[idx][i]), ha="center", fontsize=8, zorder=12)

    if plot_legend:
        legend = mlines.Line2D([], [], color="black", marker="+", linewidth=0, label="sim mean/std")
        plt.legend(handles=[legend], bbox_to_anchor=(-0.25, 1))
    plt.title("Total Neurons="+str(data.shape[0]), pad=10, fontsize=10)
    plt.suptitle(title, y=1, fontsize=18)
    # plt.show()


def dot_plot_resample(data, area=None, title=None, err="se", add_legend=False,
                              to_plot="proportion", ylabel="Proportion", ylim=(0), fig_size=(3.5,3.5)):
    """Plot open/closed circle of proportions per area for data and resampled data.

    Args:
        data (DataFrame): Dataframe of proportions, included resampled data.
        area (str, optional): Area to plot. Defaults to None.
        title (str, optional): Title for plot. Defaults to None.
        err (str, optional): Error to plot, can be "ci", "pi", "se", or "sd". Defaults to "se".
        add_legend (bool, optional): Whether to add legend labeling mean/err. Defaults to False.
        to_plot (str, optional): Column to plot. Defaults to "proportion".
        ylim (tuple, optional): lower bound for yaxis. Defaults to (0).
    """

    if area:
        df = data[data["area"]==area]
    else:
        df = data.copy()


    # add column for xaxis plotting
    df['xaxis'] = df['species'].replace({"MMus":0, "STeg":1, "MMus_resampled":2, "STeg_resampled":3})

    fig, ax = plt.subplots()

    # plot mean and error bar for each species
    ax = sns.pointplot(data=df, x="xaxis", y=to_plot, hue="species", units='mice', 
                          color='black', markers='+', ax=ax, errorbar=err) # plots mean and 95 confidence interval:
    # necessary to put mean/error markers on top
    plt.setp(ax.lines, zorder=100)
    plt.setp(ax.collections, zorder=100, label="")

    # needed to prevent cut-off of dots near top of graph
    plt.margins(0.15)

    # plot proportions w/ closed/open circles
    colors=["tab:blue", "tab:orange", "tab:blue", "tab:orange"]
    markers = ["o", "o", MarkerStyle('o', fillstyle="none"), MarkerStyle('o', fillstyle="none"), ]
    for i in range(4):
        df_temp = df[df['xaxis']==i]
        ax.scatter(x=df_temp["xaxis"], y=df_temp["proportion"], c=colors[i], marker=markers[i], s=100)



    xtick_labels = ["MMus\nfull data", "STeg\nfull data", "MMus\ndownsampled", "STeg\ndownsampled"]
    ax.set_xticklabels(xtick_labels)

    plt.title(title, size=20)
    plt.ylim(ylim) # make sure y axis starts at 0
    plt.xlabel(None)
    plt.ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # rotate xtick labels
    plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
    
    # increase text size
    plt.rcParams.update({'font.size': 12})
    
    # set figure size
    fig = plt.gcf()
    fig.set_size_inches(fig_size[0],fig_size[1])

    if add_legend:
        legend = mlines.Line2D([], [], color="black", marker="+", linewidth=0, label="mean, "+err)
        plt.legend(handles=[legend], loc="lower right")
    else:
        plt.legend([],[], frameon=False)


    return(fig)

def plot_motif_hist_prop(data, plot_areas=["OMCc", "AUD", "STR"], 
                         title=None, color="tab:orange", reps=500, subset_area=None,
                         subset_idx=None, plot_pab=False):
    """Create plots of histograms w/ 

    Args:
        data (_type_): _description_
        to_plot (list, optional): _description_. Defaults to ["OMCc", "AUD", "STR"].
        color (str, optional): _description_. Defaults to "tab:orange".
        reps (int, optional): _description_. Defaults to 500.
    """

    motifs, simulations = motif_simulation(data, plot_areas=plot_areas, reps=reps, subset=subset_area)
    motif_prop = df_to_motif_proportion(data, areas=plot_areas, proportion=True, subset=subset_area)
    pab_proportions = df_to_calc_pab_proportions(data, motif_prop.index)

    # subset by idx if given
    if subset_idx:
        start = subset_idx[0]
        end = subset_idx[1]
        motifs = motifs[start:end]
        simulations = simulations[:,start:end]
        motif_prop = motif_prop[start:end]
        pab_proportions = pab_proportions[start:end]


    # calculate number of axes needed
    n = math.ceil(len(motifs)/5) # round up divide by 4 = axs rows


    fig, axs = plt.subplots(n, 5, figsize=(20, 5*n))

    i = 0
    for ax in axs.flat:

        if i < len(motifs):
            ax.hist(simulations[:,i], color=color)
            ax.set_title(motifs[i])
            ax.axline((motif_prop[i], 0),(motif_prop[i], 10),
            color="grey", linestyle="--")
            if plot_pab:
                ax.axline((pab_proportions[i], 0),(pab_proportions[i], 10),
            color="grey", linestyle="-")
        else:
            ax.axis('off')
        
        i+=1

    if title:
        plt.suptitle(title, fontsize=20)
    
    if plot_pab:
        # if plot_pab true, plot legend
        linestyles = ["--", "-"]
        lines = [Line2D([0], [0], color="grey", linestyle=l, linewidth=3) for l in linestyles]
        labels = ["Observed Proportion", "Calculated Proportion - P(A)*P(B)"]
        fig.legend(lines,labels, bbox_to_anchor=(0.75, 0.935))
    
    return(fig)