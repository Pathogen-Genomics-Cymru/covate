#!/usr/bin/env python3

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .utils import getenddate
from .build_time_series import gettimeperiod


def buildhbseries(metadata, adm, lineagetype, timeperiod, enddate,
                  output, nsteps, primaryregion, hbdata):
    """ Build HB time series for lineages in primary region (WLS)"""

    # set path
    path = os.path.join(output, str(getenddate(enddate)), 'healthboard')

    # load metadata and index by date
    df = pd.read_csv(metadata, usecols=['cog_id', adm, 'sample_date',
                                        lineagetype, 'phylotype'],
                     parse_dates=['sample_date'], index_col='sample_date',
                     dayfirst=True, dtype={'cog_id': str, adm: str,
                                           lineagetype: str, 'phylotype': str})

    # select time period
    df, enddate = gettimeperiod(df, timeperiod, enddate, False, False)

    # only keep primary region, e.g. UK-WLS
    wal = df[df[adm].str.match(primaryregion, na=False)]

    # get date lineage was first and last observed
    firstdate = wal.drop_duplicates(lineagetype, keep='first')
    firstdate.drop('phylotype', axis=1, inplace=True)
    firstdate = firstdate.rename_axis('first_observed').reset_index()
    firstdate.drop(['cog_id', adm], axis=1, inplace=True)

    lastdate = wal.drop_duplicates(lineagetype, keep='last')
    lastdate.drop('phylotype', axis=1, inplace=True)
    lastdate = lastdate.rename_axis('last_observed').reset_index()
    lastdate.drop(['cog_id', adm], axis=1, inplace=True)

    firstlastdate = pd.merge(firstdate, lastdate, on=[lineagetype], how='left')
    firstlastdate.to_csv(path + '/waleslineagedates.csv', sep=',', index=False)

    # reset index
    wal = wal.rename_axis('sample_date').reset_index()

    # count lineages and phylotypes
    lineage_group_count = wal.groupby([lineagetype]).size()
    lineage_group_count.to_csv(path + '/waleslineagecount.csv', sep=',')

    phylotype_group_count = wal.groupby(['phylotype']).size()
    phylotype_group_count.to_csv(path + '/walesphylotypecount.csv', sep=',')

    # drop phylotype column
    wal = wal.drop('phylotype', axis=1)

    # drop na
    wal.replace(['None', 'nan'], np.nan, inplace=True)
    wal.dropna(inplace=True)

    # load hb metadata
    hb = pd.read_csv(hbdata, usecols=['cog_id', 'HB'],
                     dtype={'cog_id': str, 'HB': str})

    hb.replace(['None', 'nan' 'NA'], np.nan, inplace=True)
    hb.dropna(inplace=True)

    # combine with hb
    walhb = pd.merge(wal, hb, on=['cog_id'], how='left')
    walhb.dropna(inplace=True)

    # dataframe to csv
    walhb.to_csv(path + '/hblineage.csv', sep=',', index=False)

    # plot unique lineages observed in a healthboard on a given day
    plotuniquehb(walhb, lineagetype, output, enddate)

    return (walhb)


def plotuniquehb(wal, lineagetype, output, enddate):
    """ Plot unique lineages by HB in primary region (WLS)"""

    # set path
    path = os.path.join(output, str(getenddate(enddate)), 'healthboard')

    # set scalepoint for plots
    scalepoint = 10

    # drop duplicates (we want number of unique lineages in a day)
    wal.drop_duplicates(subset=['sample_date', lineagetype, 'HB'],
                        keep='first', inplace=True)

    # define healthboard dictionary
    di = {"Powys Teaching Health Board": "Powys Teaching",
          "Hywel Dda University Health Board": "Hywel Dda University",
          "Cardiff and Vale University Health Board": "Cardiff & Vale University",
          "Swansea Bay University Health Board": "Swansea Bay University",
          "Cwm Taf Morgannwg University Health Board": "Cwm Taf Morgannwg University",
          "Aneurin Bevan University Health Board": "Aneurin Bevan University",
          "Betsi Cadwaladr University Health Board": "Betsi Cadwaladr University"}

    healthb = ["Hywel Dda University", "Swansea Bay University", "Cwm Taf Morgannwg University",
               "Cardiff & Vale University", "Aneurin Bevan University", "Powys Teaching",
               "Betsi Cadwaladr University"]

    wal = wal.replace({"HB": di})

    wal["HB"] = pd.Categorical(wal["HB"], categories=healthb)
    wal.sort_values('HB', inplace=True)
    wal.HB = wal.HB.astype(str)
    wal['sample_date'] = wal['sample_date'].astype(str)

    wal['combine'] = wal['sample_date'] + " " + wal['HB']
    wal['frequency'] = wal['combine'].map(wal['combine'].value_counts())
    cw = wal['frequency'].to_numpy()
    sw = lambda x: scalepoint*x

    wal.sample_date = pd.to_datetime(wal['sample_date'], format='%Y-%m-%d')

    wal.to_csv(path + '/hbplot.csv', sep=',', index=False)

    # set colours
    colors = {"Hywel Dda University": "#CCCC33", "Swansea Bay University": "#FF6600",
              "Cwm Taf Morgannwg University": "#993300", "Cardiff & Vale University": "#333399",
              "Aneurin Bevan University": "#006633", "Powys Teaching": "#339900",
              "Betsi Cadwaladr University": "#CC3333"}

    # plot
    # open wales map
    im = Image.open('hb.png')
    height = im.size[1]

    # need a float array between 0-1
    im = np.array(im).astype(float) / 255

    fig, ax = plt.subplots()
    fig.set_size_inches(18, 6)
    plt.rcParams['font.size'] = '12'
    plt.scatter(wal.sample_date, wal.HB, s=sw(cw), c=wal['HB']
                .apply(lambda x: colors[x]), marker='o')
    ax.set_xlabel("Date collected", fontsize='14')
    plt.yticks(rotation=30)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(14)
    ax.grid()
    msizes = [1, 10, 20, 30]
    markers = []
    for size in msizes:
        markers.append(plt.scatter([], [], s=scalepoint*size, label=size,
                       c='grey'))
    plt.legend(handles=markers, labelspacing=3,
               title="Number of individual lineages",
               bbox_to_anchor=(1.005, 1), frameon=False)
    fig.figimage(im, fig.bbox.xmax + 3450.0, height - 700.0)

    plt.savefig(path + "/healthboardDate.svg", format="svg", bbox_inches='tight')
    plt.savefig(path + "/healthboardDate.png", format="png",
                bbox_inches='tight', dpi=300)
