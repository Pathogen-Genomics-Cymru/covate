import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .utils import getenddate, createoutputdir


def buildseries(metadata, regions, adm, lineagetype, timeperiod, enddate,
                output, nsteps, validate, crosscorr, primaryregion):
    """ Build the time series for lineages common to specified regions"""

    # load metadata and index by date
    df = pd.read_csv(metadata, usecols=['cog_id', adm, 'sample_date',
                                        lineagetype],
                     parse_dates=['sample_date'], index_col='sample_date',
                     dayfirst=True, dtype={'cog_id': str, adm: str,
                                           lineagetype: str})

    df.replace(['None', 'nan'], np.nan, inplace=True)
    df.dropna(inplace=True)

    # select time period
    df, enddate = gettimeperiod(df, timeperiod, enddate, nsteps, validate)

    # get region list
    region_list = [str(region) for region in regions.split(', ')]

    # only keep regions in regions list
    df = df[df[adm].isin(region_list)]

    # create array of lineages
    lineagearr = df[lineagetype].unique()

    # rename lineage by region (e.g UKX_UK-WLS)
    df['lineage_adm'] = df[[lineagetype, adm]].apply(lambda x: '_'.join(x),
                                                     axis=1)

    # get the lineage count by date in each region
    groupdf = (df.groupby('lineage_adm').resample('D')['cog_id'].count()
                 .reset_index(name="count"))

    countbydateall = groupdf.pivot_table('count', ['sample_date'],
                                         'lineage_adm')

    countbydateall.replace([np.nan], '0', inplace=True)

    # get column names for lineages and convert to numeric
    lineage_adm_cols = [item for item in countbydateall.columns
                        if item not in ["sample_date"]]

    for col in lineage_adm_cols:

        countbydateall[col] = pd.to_numeric(countbydateall[col],
                                            downcast="integer")

    # fill in missing dates
    countbydateall = countbydateall.asfreq('D', fill_value=0)

    # only keep lineages found in all the regions
    lineagecommon = []
    numcountry = len(region_list)

    for lineage in lineagearr:

        lineagestr = str(lineage) + '_'

        listbylineage = (countbydateall.columns.str.startswith(lineagestr)
                         .tolist())

        truecount = sum(listbylineage)

        countbydate = countbydateall.copy()

        if truecount < numcountry:

            countbydate = countbydateall.loc[:,
                                             ~countbydateall.columns
                                             .str.startswith(lineagestr)]
        else:

            lineagecommon.append(lineage)

    # create output directory
    for lineage in lineagecommon:

        createoutputdir(lineage, output, enddate)

    if not validate:
        # save time series
        path = os.path.join(output, str(getenddate(enddate)))

        countbydateall.to_csv(path + '/timeseriesall.csv', sep=',')
        countbydate.to_csv(path + '/timeseriescommon.csv', sep=',')

        # plot time series and lag plot
        plotseries(countbydate, lineagecommon, region_list, output, enddate)

    # if cross-correlation True
    toplineagelist = []
    ascendregionlist = []

    if crosscorr:

        toplineagelist, ascendregionlist = plottopseries(df, lineagecommon,
                                                         region_list, output,
                                                         enddate, adm,
                                                         lineagetype,
                                                         primaryregion, 40)

    return (countbydate, lineagecommon, region_list, enddate, toplineagelist,
            ascendregionlist)


def gettimeperiod(dataframe, timeperiod, enddate, nsteps, validate):
    """Extract time period from metadata specified by --time-period"""

    # if enddate not specified, get most recent date in metadata and -7 days
    if enddate:
        enddate = datetime.strptime(enddate, '%d/%m/%Y')
    else:
        enddate = dataframe.index.max() - relativedelta(days=7)

    # select previous x weeks from most recent date
    if not validate:
        startdate = enddate - relativedelta(weeks=+int(timeperiod))
    else:
        startdate = enddate - relativedelta(weeks=+int(timeperiod))
        startdate = startdate - relativedelta(days=+int(nsteps))

    # get range of dates
    dataframe = dataframe.sort_index().loc[str(startdate):str(enddate)]

    return dataframe, enddate


def plotseries(dataframe, lineagelist, regionlist, output, enddate):
    """Plot the time series and lag plots"""

    colors = ['r', 'g', 'b', 'm', 'c', 'y']

    for lineage in lineagelist:
        ncolor = 0
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
        path = os.path.join(output, str(getenddate(enddate)), lineage,
                            'additional-plots')
        for region in regionlist:
            dataframe[lineage + '_' + region].plot(ax=ax1, c=colors[ncolor])
            pd.plotting.lag_plot(dataframe[lineage + '_' + region],
                                 c=colors[ncolor], ax=ax2)
            ncolor += 1
        ax1.title.set_text('Time series for ' + lineage + ' for '
                           + str(regionlist))
        ax1.set_ylabel('Number of cases')
        ax1.legend(loc="upper left")
        ax2.title.set_text('Lag plot for ' + lineage + ' for '
                           + str(regionlist))
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_timeseries.png')
        plt.clf()
        plt.close(fig)


def plottopseries(dataframe, lineagelist, regionlist, output, enddate, adm,
                  lineage, primaryregion, num):
    """Plot the top $num time series"""

    colors = ['slategrey', 'maroon', 'g', 'm', 'c', 'y']
    ncolor = 0

    path = os.path.join(output, str(getenddate(enddate)), 'cross-correlation')

    # set scale for points in fig
    scalepoint = 0.75

    # initialise figure
    fig, ax = plt.subplots()
    fig.set_size_inches(14, 18)
    ax.set_xlabel("Date collected")
    ax.set_ylabel("Lineage")
    ax.set_axisbelow(True)
    ax.grid()

    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width,
                     box.height * 0.9])

    # reorder list by country with most cases
    counts = dataframe[adm].value_counts()

    ascendregionlist = []
    for elem in range(0, len(regionlist)):
        ascendregionlist.append(str(counts.index[elem]))

    # get dataframe for primary region
    primaryregionframe = dataframe[dataframe[adm].str.match(primaryregion)]
    primaryregionframe = primaryregionframe.reset_index()

    # get lineage counts for primary region
    lineageregioncount = (primaryregionframe.groupby([lineage]).size()
                          .reset_index(name=primaryregion))
    lineageregioncount.sort_values(primaryregion, ascending=False,
                                   inplace=True)
    lineageregioncount[primaryregion] = (lineageregioncount[primaryregion]
                                         .astype(int))
    lineageregioncount.to_csv(path + '/' + primaryregion + '_lineagefreq.csv',
                              sep=',', index=False)

    # get top $num lineages from primary region that are common to all regions
    rownum = int(num)
    lineageregioncount[lineage] = pd.Categorical(lineageregioncount[lineage],
                                                 categories=lineagelist)
    lineageregioncount.dropna(inplace=True)
    toplineagelist = lineageregioncount[lineage].iloc[0:rownum].tolist()

    # get secondary regions and their lineage counts
    seclist = [s for s in regionlist if not str(primaryregion) in s]

    for region in seclist:

        secondregionframe = dataframe[dataframe[adm].str.match(region)]
        secondregionframe = secondregionframe.reset_index()

        lineageregioncount2 = (secondregionframe.groupby([lineage]).size()
                               .reset_index(name=region))
        lineageregioncount2.sort_values(region, ascending=False, inplace=True)

        lineageregioncount2.to_csv(path + '/' + region + '_lineagefreq.csv',
                                   sep=',', index=False)

        # get combined lineage region count dataframe
        lineageregioncount = lineageregioncount.merge(lineageregioncount2,
                                                      on=lineage,
                                                      how='outer')

    lineageregioncount = lineageregioncount.fillna(0)

    for region in ascendregionlist:

        lineageregioncount[region] = (lineageregioncount[region]
                                      .astype(int))

    lineageregioncount.to_csv(path + '/' + 'lineagefreqbyregion.csv',
                              sep=',', index=False)

    for region in ascendregionlist:

        regionframe = dataframe[dataframe[adm].str.match(region)]
        regionframe = regionframe.reset_index()

        regionframe[lineage] = pd.Categorical(regionframe[lineage],
                                              categories=toplineagelist)
        regionframe.dropna(inplace=True)

        regionframe['combine'] = (regionframe['sample_date'].astype(str) + "_"
                                  + regionframe[lineage].astype(str))
        regionframe['frequency'] = (regionframe['combine']
                                    .map(regionframe['combine']
                                    .value_counts()))

        regionframe.drop_duplicates(subset=['combine'], inplace=True)

        cc = regionframe['frequency'].to_numpy()

        def sc(x, scalepoint): return scalepoint*x

        regionframe['sample_date'] = pd.to_datetime(regionframe['sample_date'],
                                                    format='%Y-%m-%d')
        regionframe[lineage] = regionframe[lineage].astype(str)

        # plt scatter reads global lineages as floats, replace dot with dash
        regionframe[lineage] = [x.replace('.', '-') for x in
                                regionframe[lineage].astype(str)]

        plt.scatter(regionframe['sample_date'], regionframe[lineage],
                    s=sc(cc, scalepoint), c=colors[ncolor], marker='o',
                    label=str(region))

        ncolor += 1

    leg = plt.legend(bbox_to_anchor=(0.8, -0.07), frameon=False)
    ax.add_artist(leg)
    ax.title.set_text('Number of cases in ' + str(ascendregionlist)
                      + ' for the ' + str(num)
                      + ' most observed lineages in ' + primaryregion)

    msizes = [1, 10, 50, 100, 500, 1000, 2000]
    markers = []
    for size in msizes:
        markers.append(plt.scatter([], [], s=scalepoint*size, label=size,
                       c='grey'))
    leg = plt.legend(bbox_to_anchor=(0.8, -0.07), frameon=False)
    ax.legend(handles=markers, labelspacing=3, title="Number of cases",
              loc='upper center', bbox_to_anchor=(0.3, -0.05), frameon=False,
              ncol=len(msizes))

    plt.savefig(path + '/' + 'allLineages.png', format="png", dpi=300)
    plt.clf()
    plt.close(fig)

    return toplineagelist, ascendregionlist


def padseries(dataframe):
    """Pad the time series"""

    dataframe = dataframe.replace(0.0, np.nan)
    dataframe = dataframe.fillna(value=1.0)
    dataframe = dataframe.dropna()

    return dataframe
