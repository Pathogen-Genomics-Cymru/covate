import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .utils import getdate, getenddate, createoutputdir

def buildseries(metadata, regions, adm, lineagetype, timeperiod, enddate, output, validate):
    """ Build the time series for lineages common to specified regions"""

    # load metadata and index by date
    df = pd.read_csv(metadata, usecols=['central_sample_id', adm, 'sample_date', lineagetype], parse_dates=['sample_date'], index_col='sample_date', dayfirst=True)
    df.dropna(inplace=True)
    df[lineagetype] = df[lineagetype].astype(str)
    df[adm] = df[adm].astype(str)

    # select time period
    df, enddate = gettimeperiod(df, timeperiod, enddate, validate)

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
        lineagestr = str(lineage) + '_'
        listbylineage = countbydate.columns.str.startswith(lineagestr).tolist()
        truecount = sum(listbylineage)
        if truecount < 2:
            countbydate = countbydate.loc[:,~countbydate.columns.str.startswith(lineagestr)]
        else:
            lineagecommon.append(lineage)

    # pad time series
    #countbydate = padseries(countbydate)

    # create output directory
    for lineage in lineagecommon:
        createoutputdir(lineage, output, enddate)

    if not validate:
        # save raw time series
        path = os.path.join(output, str(getenddate(enddate)))
        countbydate.to_csv(path + '/timeseriesraw.csv', sep=',')

        # plot time series and lag plot
        plotseries(countbydate, lineagecommon, region_list, output, enddate)

    return countbydate, lineagecommon, region_list, enddate


def gettimeperiod(dataframe, timeperiod, enddate, validate):
    """Extract time period from metadata specified by --time-period"""

    # if enddate is not specified, get the most recent date in metadata and -7 days
    if enddate:
        enddate = datetime.strptime(enddate, '%d/%m/%Y')
    else:
        enddate = dataframe.index.max() - relativedelta(days=7)

    # select previous x weeks from most recent date
    if not validate:
        startdate = enddate - relativedelta(weeks=+int(timeperiod))
    else:
        startdate = enddate - relativedelta(weeks=+int(timeperiod)+2)

    # get range of dates
    dataframe = dataframe.sort_index().loc[str(startdate):str(enddate)]

    return dataframe, enddate


def plotseries(dataframe, lineagelist, regionlist, output, enddate):
    """Plot the time series and lag plots"""

    colors = ['r', 'g', 'b', 'm', 'c', 'y']

    for lineage in lineagelist:
        ncolor = 0
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10,6))
        path = os.path.join(output, str(getenddate(enddate)), lineage, 'additional-plots')
        for region in regionlist:
            dataframe[lineage + '_' + region].plot(ax=ax1, c=colors[ncolor])
            pd.plotting.lag_plot(dataframe[lineage + '_' + region], c=colors[ncolor], ax=ax2)
            ncolor+=1
        ax1.title.set_text('Time series for ' + lineage + ' for ' + str(regionlist))
        ax1.set_ylabel('Number of cases')
        ax1.legend(loc="upper left")
        ax2.title.set_text('Lag plot for ' + lineage + ' for ' + str(regionlist))
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_timeseries.png')
        plt.clf()
        plt.close(fig)


def padseries(dataframe):
    """Pad the time series"""

    dataframe = dataframe.replace(0.0, np.nan)
    dataframe = dataframe.fillna(value=1.0)
    dataframe = dataframe.dropna()

    return dataframe
