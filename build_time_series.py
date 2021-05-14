import sys
import pandas as pd
import numpy as np

countries_list = ['UK-WLS', 'UK-ENG']

def buildseries(metadata):

    df = pd.read_csv(metadata, usecols=['central_sample_id', 'adm1', 'sample_date', 'uk_lineage'], parse_dates=['sample_date'], index_col='sample_date', dayfirst=True)
    df.dropna(inplace=True)
    df.uk_lineage = df.uk_lineage.astype(str)
    df.adm1 = df.adm1.astype(str)

    df = df[df['adm1'].isin(countries_list)]
    df['uk_lineage_adm1'] = df[['uk_lineage', 'adm1']].apply(lambda x: '_'.join(x), axis=1)     

    groupdf = df.groupby('uk_lineage_adm1').resample('D')['central_sample_id'].count().reset_index(name="count")
    countbydate = groupdf.pivot_table('count', ['sample_date'], 'uk_lineage_adm1')
    countbydate.replace([np.nan], '0', inplace=True)
    countbydate.to_csv('timeseries.csv', sep=',')

    return countbydate
