import sys
import pandas as pd
import numpy as np

countries_list = ['UK-WLS', 'UK-ENG']

def buildseries(metadata):

    # load metadata and index by date
    df = pd.read_csv(metadata, usecols=['central_sample_id', 'adm1', 'sample_date', 'uk_lineage'], parse_dates=['sample_date'], index_col='sample_date', dayfirst=True)
    df.dropna(inplace=True)
    df.uk_lineage = df.uk_lineage.astype(str)
    df.adm1 = df.adm1.astype(str)

    # only keep countries in countries_list
    df = df[df['adm1'].isin(countries_list)]

    # create array of lineages
    uklineagearr = df.uk_lineage.unique()

    # rename uk lineage by country (e.g UKX_UK-WLS)
    df['uk_lineage_adm1'] = df[['uk_lineage', 'adm1']].apply(lambda x: '_'.join(x), axis=1)     

    # get the lineage count by date in each country
    groupdf = df.groupby('uk_lineage_adm1').resample('D')['central_sample_id'].count().reset_index(name="count")
    countbydate = groupdf.pivot_table('count', ['sample_date'], 'uk_lineage_adm1')
    countbydate.replace([np.nan], '0', inplace=True)

    # only keep lineages found in all countries in countries_list
    numcountry = len(countries_list)
    for lineage in uklineagearr:
       listbylineage = countbydate.columns.str.startswith(lineage).tolist()
       truecount = sum(listbylineage)
       if truecount < numcountry:
           countbydate = countbydate.loc[:,~countbydate.columns.str.startswith(lineage)]

    countbydate.to_csv('timeseries.csv', sep=',')

    return countbydate
