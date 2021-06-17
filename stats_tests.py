import pandas as pd
from statsmodels.tsa.stattools import grangercausalitytests
from utils import pairwise


def runtests(timeseries, lineagelist, regionlist):

    grangercausality(timeseries, lineagelist, regionlist)


def grangercausality(timeseries, lineagelist, regionlist):

    maxlag=14

    for loc1, loc2 in pairwise(regionlist):
        for lineage in lineagelist:
           data = pd.concat([timeseries[lineage + '_' + loc1], timeseries[lineage + '_' +  loc2]], axis=1)
           try:
               test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
               p_values = [test_result[i+1][0][test][1] for i in range(maxlag)]
               min_p_value = np.min(p_values)
               print (lineage)
               print (loc2 + '->' + loc1)
               print (min_p_value)
           except:
               print(lineage + " " + loc2 +  "-> " + loc1 + " failed")
