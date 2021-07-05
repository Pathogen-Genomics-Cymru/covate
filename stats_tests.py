import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils import pairwise, pairwiseunique, appendline

def runtests(timeseries, lineagelist, regionlist):

    maxlag = 30
    alpha = 0.05

    grangercausality(timeseries, lineagelist, regionlist, maxlag, alpha)

    cointegration(timeseries, lineagelist, regionlist, maxlag)

    adfullertest(timeseries, lineagelist, alpha)


def grangercausality(timeseries, lineagelist, regionlist, maxlag, alpha):

    test = "ssr_chi2test"

    for loc1, loc2 in pairwise(regionlist):
        for lineage in lineagelist:
            data = pd.concat([timeseries[lineage + '_' + loc1], timeseries[lineage + '_' +  loc2]], axis=1)
            try:
                test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
                min_p_value = min(p_values, key=lambda x: x[0])
                filename = str(lineage) + '_log.txt'
                appendtext = 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + '(min p-value, lag) = ' + str(min_p_value) + ' =>  ' + str(min_p_value[0] <= alpha)
                appendline(filename, appendtext)
            except:
                appendtext = 'Granger causality test for ' + str(loc2) +  '-> ' + str(loc1) + ' failed'
                appendline(filename, appendtext)


def cointegration(timeseries, lineagelist, regionlist, maxlag):
    """Perform Johanson's Cointegration Test"""

    for lineage in lineagelist:
        data = timeseries.filter(like=lineage)
        data.reset_index(drop=True, inplace=True)
        filename = str(lineage) + '_log.txt'
        appendline(filename, 'Lags with Cointegration')
        for lag in range(maxlag):
            out = coint_johansen(data, -1, lag)
            d = {'0.90':0, '0.95':1, '0.99':2}
            traces = out.lr1
            cvts = out.cvt[:, d[str(0.95)]]
            for col, trace, cvt in zip(data.columns, traces, cvts):
                if trace > cvt:
                    appendline(filename, str(col) + ' lag= ' + str(lag))


def adfullertest(timeseries, lineagelist, alpha):
    """Perform ADFuller to test for Stationarity"""

    for lineage in lineagelist:
        data = timeseries.filter(like=lineage)
        data.reset_index(drop=True, inplace=True)
        filename = str(lineage) + '_log.txt'
        for name, col in data.iteritems():
            stat = adfuller(col, autolag ="AIC")
            out = {'test_statistic':round(stat[0], 4), 'pvalue':round(stat[1], 4), 'n_lags':round(stat[2], 4), 'n_obs':stat[3]}
            p_value = out['pvalue']
            appendline(filename, 'Augmented Dickey-Fuller Test for ' + str(name))
            appendline(filename, 'Test Statistic = ' + str(out["test_statistic"]))
            appendline(filename, 'No. Lags Chosen = ' + str(out["n_lags"]))
            if p_value <= alpha:
                appendline(filename, '=> P-Value = ' +  str(p_value) + ' => Reject Null Hypothesis')
                appendline(filename, '=> Series is Stationary')
            else:
                appendline(filename, '=> P-Value = ' + str(p_value))
                appendline(filename, '=> Series is Non-Stationary')

