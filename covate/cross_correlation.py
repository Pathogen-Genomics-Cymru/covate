import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import matplotlib.pylab as pl
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import os
from .utils import getdate, getenddate

def crosscorrelation(timeseries, lineagelist, regionlist, enddate, output, primaryregion):

    laggedcorr(timeseries, lineagelist, regionlist, enddate, output, primaryregion)


def laggedcorr(timeseries, lineagelist, regionlist, enddate, output, primaryregion):

    path = os.path.join(output, str(getenddate(enddate)))

    # raise error if more than two regions
    if len(regionlist) > 2:
        raise ValueError('The cross-correlation analysis is currently only supported for two regions.')

    # get secondary region
    seclist = [s for s in regionlist if not str(primaryregion) in s]
    secondregion = str(seclist[0])

    appended_data = []
    for lineage in lineagelist:
        primcol = str(lineage) + "_" + str(primaryregion)
        seccol = str(lineage) + "_" + str(secondregion)
        data = pd.concat([timeseries[seccol], timeseries[primcol]], axis=1)
        lagged_correlation = pd.DataFrame.from_dict({x: [data[seccol].corr(data[x].shift(t), method='kendall') for t in range(-30, 31, 1) ] for x in data.columns})
        appended_data.append(lagged_correlation.iloc[:, 1])

    appended_data = pd.concat(appended_data, axis=1)
    appended_data = appended_data.T

    day_list = list(range(-30, 31, 1))
    appended_data.columns = day_list

    max_corr_time = appended_data.idxmax(axis=1)
    max_corr_value = appended_data.max(axis=1)

    max_combine = pd.concat([max_corr_time, max_corr_value], axis=1)
    max_combine_03 = max_combine[ max_combine.loc[:,1]>= 0.3]
    max_combine_05 = max_combine[ max_combine.loc[:,1]>= 0.5]
    max_combine_07 = max_combine[ max_combine.loc[:,1]>= 0.7]


    bins_list = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

    # Create 2x2 sub plots
    gs = gridspec.GridSpec(2, 3)

    fig = plt.figure()
    fig.set_size_inches(16, 8)

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.hist(max_corr_time, bins = bins_list, color = "slategrey")
    ax1.set_ylabel('Lineage Count', fontsize=12)
    ax1.set_xlabel('Lag (days)', fontsize=12)
    ax1.set_xlim(left=-32, right=32)
    ax1.annotate('(b)', xy=(-30, 7.5), fontsize=14)

    ax2 = fig.add_subplot(gs[1, 1:])
    ax2.hist(max_combine_05.iloc[:,0], bins = bins_list, color = "slategrey")
    x_ticks=[-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25]
    ax2.set_ylabel('Lineage Count', fontsize=14)
    ax2.set_xlabel('Lag (days)', fontsize=14)
    ax2.set_xticks(x_ticks)
    ax2.set_xlim(left=-31, right=26)
    ax2.annotate('(c)',  xy=(-30, 5.5), fontsize=14)

    ax3 = fig.add_subplot(gs[0, :])
    appended_data.boxplot()
    ax3.set_xticks(ax3.get_xticks()[::2])
    ax3.set_xlabel("Lag (days)", fontsize='14')
    ax3.set_ylabel("Correlation coefficient", fontsize='14')
    at = AnchoredText("(a)", frameon=False,
                  prop=dict(size=14), loc='upper left',
                  )
    ax3.add_artist(at)

    plt.rcParams['font.size'] = '10'
    plt.savefig(path + '/' + "crosscorrelation.png", format="png")
