import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from utils import pairwise, appendline

def runtests(timeseries, lineagelist, regionlist):

    grangercausality(timeseries, lineagelist, regionlist)


def grangercausality(timeseries, lineagelist, regionlist):

    maxlag = 14
    test = "ssr_chi2test"

    for loc1, loc2 in pairwise(regionlist):
        for lineage in lineagelist:
            data = pd.concat([timeseries[lineage + '_' + loc1], timeseries[lineage + '_' +  loc2]], axis=1)
            try:
                test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                p_values = [test_result[lag+1][0][test][1] for lag in range(maxlag)]
                min_p_value = np.min(p_values)
                filename = str(lineage) + '_log.txt'
                appendtext = 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + 'min p-value = ' + str(min_p_value)
                appendline(filename, appendtext)
            except:
                appendtext = 'Granger causality test for ' + str(loc2) +  '-> ' + str(loc1) + ' failed'
                appendline(filename, appendtext)
