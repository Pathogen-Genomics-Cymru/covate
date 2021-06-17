import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def buildseries(metadata, regions, adm, lineagetype):

    # load metadata and index by date
    df = pd.read_csv(metadata, usecols=['central_sample_id', adm, 'sample_date', lineagetype], parse_dates=['sample_date'], index_col='sample_date', dayfirst=True)
    df.dropna(inplace=True)
    df[lineagetype] = df[lineagetype].astype(str)
    df[adm] = df[adm].astype(str)

    # get region list
    region_list = [str(region) for region in regions.split(', ')]

    # only keep regions in regions list
    df = df[df[adm].isin(region_list)]

    # create array of lineages
    lineagearr = df[lineagetype].unique()

    # rename lineage by region (e.g UKX_UK-WLS)
    df['lineage_adm'] = df[[lineagetype, adm]].apply(lambda x: '_'.join(x), axis=1)

    # get the lineage count by date in each region
    groupdf = df.groupby('lineage_adm').resample('D')['central_sample_id'].count().reset_index(name="count")
    countbydate = groupdf.pivot_table('count', ['sample_date'], 'lineage_adm')
    countbydate.replace([np.nan], '0', inplace=True)

    # get column names for lineages and convert to numeric
    lineage_adm_cols=[item for item in countbydate.columns if item not in ["sample_date"]]
    for col in lineage_adm_cols:
        countbydate[col]=pd.to_numeric(countbydate[col])

    # only keep lineages found in at least two regions
    lineagecommon = []
    numcountry = len(region_list)
    for lineage in lineagearr:
        listbylineage = countbydate.columns.str.startswith(lineage).tolist()
        truecount = sum(listbylineage)
        if truecount < 2:
            countbydate = countbydate.loc[:,~countbydate.columns.str.startswith(lineage)]
        else:
            lineagecommon.append(lineage)

    countbydate.to_csv('timeseriesraw.csv', sep=',')

    countbydate = padseries(countbydate)

    plotseries(countbydate, lineagecommon, region_list)

    return countbydate, lineagecommon, region_list

def plotseries(dataframe, lineagelist, regionlist):

    colors = ['r', 'g', 'b', 'm', 'c', 'y']

    ncolor = 0
    for lineage in lineagelist:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
        for region in regionlist:
            dataframe[lineage + '_' + region].plot(ax=ax1, c=colors[ncolor])
            pd.plotting.lag_plot(dataframe[lineage + '_' + region], c=colors[ncolor], ax=ax2)
            ncolor+=1
        ax1.title.set_text('Time series for ' + lineage + ' for ' + str(regionlist))
        ax1.set_ylabel('Number of cases')
        ax2.title.set_text('Lag plot for ' + lineage + ' for ' + str(regionlist))
        plt.tight_layout()
        plt.savefig(lineage + '.png')

def padseries(dataframe):

    dataframe = dataframe.replace(0, np.nan)
    dataframe = dataframe.fillna(method='pad')
    dataframe = dataframe.dropna()

    return dataframe
