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

# for upset plots
import upsetplot


def dot_bar_plot(df, title="", xaxis="Node Degree", yaxis="Normalized Frequency", hueaxis="Species"):
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
    t_ax = sns.barplot(df, x=xaxis, y=yaxis, hue=hueaxis, errorbar="se", errwidth=1)
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

def sorted_heatmap(df, title=None, sort_by=['type'], drop=['type'], nsample=None, 
                   random_state=10, norm=None, cmap=orange_cmp, cbar=False,
                   label_neurons=None):
    """_summary_

    Args:
        df (DataFrame): Dataframe to plot heatmap
        sort_by (list, optional): How to sort neurons. Defaults to ['type'].
        nsample (integer, optional): If prsent, down sample dataframe. Defaults to None.
        random_state (int, optional): If downsample, what random state to use. Defaults to 10.
        label_neurons (dict, option): Dictionary of label:index of neurons to label
    """

    if nsample:
        plot = df.sample(nsample, random_state=random_state)
    else:
        plot = df.copy()

    plot = plot.replace({"IT":0.25, "CT":0.5, "PT":0.75})
    plot = plot.sort_values(by=sort_by).reset_index(drop=True)

    fig=plt.subplot()
    sns.heatmap(plot.drop(drop, axis=1), norm=norm, cmap=cmap, cbar=cbar)
    plt.gca().get_yaxis().set_visible(False)
    plt.title(title)
    if label_neurons:
        for key in label_neurons.keys():
            plt.text(-0.3,label_neurons[key], key+"-", va="center_baseline")
    return(fig)
    
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
    plt.text(-0.3, 0.5, label, va="center_baseline")

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


def area_proportion_dot_plot(df, area, title=None):
    """_summary_

    Args:
        df (DataFrame): Output from dfs_to_proportion
        area (string): area to plot
    """

    area_df = df[df["area"]==area]

    means = area_df.groupby('species')['proportion'].mean() # need means for plotting lines

    fig, ax = plt.subplots()

    strip = sns.stripplot(data=area_df, x="species", y="proportion", hue="species", size=10, ax=ax)
    violin = sns.violinplot(area_df, x='species',y="proportion",
                split=True, hue ='species', inner = None, 
                palette="pastel",legend=False)
    point = sns.pointplot(data=area_df, x="species", y="proportion", hue="species", units='mice', color='black', markers='+', ax=ax) # plots mean and 95 confidence interval:

    mm_line = mlines.Line2D([0, 1], [means["MMus"], means["MMus"]], color=blue_cmp.colors[150])
    st_line = mlines.Line2D([0, 1], [means["STeg"], means["STeg"]], color=orange_cmp.colors[150])
    
    ax.add_line(mm_line)
    ax.add_line(st_line)

    plt.title(title, size=20)
    plt.ylim((0)) # make sure y axis starts at 0
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    plt.legend(loc="upper left")
    legend = mlines.Line2D([], [], color="black", marker="+", linewidth=0, label="mean, 95 CI")
    plt.legend(handles=[legend])

    return(fig)


def proportion_volcano_plot(df, title=None):
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
    plt.axline((0, 0), (0, 1),linestyle='--', linewidth=0.5)
    plt.axline((0, 0), (1, 0),linestyle='--', linewidth=0.5)
    plt.axline((0, -np.log10(0.05)), (1,  -np.log10(0.05)),linestyle='--', color='r', alpha=0.75, linewidth=0.5)
    plt.text(-0.1, -np.log10(0.05)+.015, 'p<0.05', color='r', alpha=0.75)
    plt.axline((0, -np.log10(0.01)), (1,  -np.log10(0.01)),linestyle='--', color='r', alpha=0.5, linewidth=0.5)
    plt.text(-0.1, -np.log10(0.01)+.015, 'p<0.01', color='r', alpha=0.75)


    for i in range(df.shape[0]):
        plt.text(x=df.log2_fc[i]+0.01,y=df.nlog10_p[i]+0.01,s=df.area[i], 
            fontdict=dict(color='black',size=10))


    plt.title(title)
    plt.xlabel('log2(fold change)')
    plt.ylabel('-log10(p-value)')

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
               ymin=0, ymax=0.5):
    """upset plot with specialized parameters

    Args:
        plot_s (Pandas series?): output of from_membership() from upsetplot package
        face_color (str, optional): Color for bars, only takes string colors. Defaults to "tab:blue".
        shading_color (str, optional): only takes strings. Defaults to "lightgray".
    """

    # fig = plt.subplot()
    upsetplot.plot(plot_s, facecolor=facecolor, shading_color=shading_color)
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
