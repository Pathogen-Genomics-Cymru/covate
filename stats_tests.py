import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils import pairwise, pairwiseunique, appendline

def runtests(timeseries, lineagelist, regionlist):

    maxlag = 30

    grangercausality(timeseries, lineagelist, regionlist, maxlag)

    #cointegration(timeseries, lineagelist, regionlist, maxlag)


def grangercausality(timeseries, lineagelist, regionlist, maxlag):

    test = "ssr_chi2test"

    for loc1, loc2 in pairwise(regionlist):
        for lineage in lineagelist:
            data = pd.concat([timeseries[lineage + '_' + loc1], timeseries[lineage + '_' +  loc2]], axis=1)
            try:
                test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
                min_p_value = min(p_values, key=lambda x: x[0])
                filename = str(lineage) + '_log.txt'
                appendtext = 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + '(min p-value, lag) = ' + str(min_p_value) + ' =>  ' + str(min_p_value[0] < 0.05)
                appendline(filename, appendtext)
            except:
                appendtext = 'Granger causality test for ' + str(loc2) +  '-> ' + str(loc1) + ' failed'
                appendline(filename, appendtext)


def cointegration(timeseries, lineagelist, regionlist, maxlag):
    """Perform Johanson's Cointegration Test"""

    for loc1, loc2 in pairwiseunique(regionlist):
        for lineage in lineagelist:
            data = pd.concat([timeseries[lineage + '_' + loc1], timeseries[lineage + '_' +  loc2]], axis=1)
            filename = str(lineage) + '_log.txt'
            appendtext = 'Lags with Cointegration'
            appendline(filename, appendtext)
            for lag in range(maxlag):
                out = coint_johansen(data, -1, lag)
                d = {'0.90':0, '0.95':1, '0.99':2}
                traces = out.lr1
                cvts = out.cvt[:, d[str(0.95)]]
                for col, trace, cvt in zip(data.columns, traces, cvts):
                    if trace > cvt:
                        appendtext = str(col) + ' lag= ' + str(lag)
                        appendline(filename, appendtext)
