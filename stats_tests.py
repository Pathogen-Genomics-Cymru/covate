import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from utils import pairwise, pairwiseunique, appendline
from statsmodels.graphics.tsaplots import plot_acf

def runtests(timeseries, lineagelist, regionlist):

    maxlag = 30
    alpha = 0.05

    lineageVECM = []
    lineageVAR = []
    VARdiff = []

    count=0
    for lineage in lineagelist:

        checkdistribution(timeseries, lineage, alpha)

        plotautocorr(timeseries, lineage, maxlag)

        grangercausality(timeseries, lineage, regionlist, maxlag, alpha)

        runVECM = cointegration(timeseries, lineage, regionlist, maxlag)

        # if lineage has cointegration, add to list of lineages for VECM
        if runVECM:

            lineageVECM.append(lineage)

        # else add to list of lineages for VAR and check for stationarity and difference
        else:

            lineageVAR.append(lineage)

            data = timeseries.filter(like=lineage)

            adf_result = adfullertest(data, lineagelist, alpha)

            # record whether no diff, first diff or second diff
            VARdiff.insert(count, 'none')

            # first diff and recalculate ADF
            if False in adf_result:
                first_diff = data.diff().dropna()
                adf_result = adfullertest(first_diff, lineagelist, alpha)
                VARdiff[count] =  'first'
                # update timeseries
                for loc in regionlist:
                    timeseries[lineage + '_' + loc] = first_diff[lineage + '_' + loc]

            # second diff
            if False in adf_result:
                second_diff = first_diff.diff().dropna()
                adf_result = adfullertest(second_diff, lineagelist, alpha)
                VARdiff[count] = 'second'
                # update timeseries
                for loc in regionlist:
                    timeseries[lineage + '_' + loc] = second_diff[lineage + '_' + loc]

        count+=1

    return timeseries, lineageVECM, lineageVAR, VARdiff


def checkdistribution(timeseries, lineage, alpha):
    """Get information about distribution"""

    data = timeseries.filter(like=lineage)
    data.reset_index(drop=True, inplace=True)
    filename = str(lineage) + '_log.txt'
    for name, col in data.iteritems():
        stat, pvalue = stats.normaltest(col)
        appendline(filename, 'Information on distribution for ' + str(name))
        appendline(filename, 'Stat = ' + str(round(stat, 4)))
        appendline(filename, 'p-value = ' + str(round(pvalue, 4)))
        if pvalue <= alpha:
            appendline(filename, '=> Data looks non-Gaussian')
        else:
            appendline(filename, '=> Data looks Gaussian')
        stat_kur = stats.kurtosis(col)
        stat_skew = stats.skew(col)
        appendline(filename, 'Kurtosis = ' + str(round(stat_kur, 4)))
        appendline(filename, 'Skewness = ' + str(round(stat_skew, 4)))


def plotautocorr(timeseries, lineage, maxlag):
    """Plot autocorrelation"""

    data = timeseries.filter(like=lineage)
    data.reset_index(drop=True, inplace=True)
    for name, col in data.iteritems():
        plot_acf(col, lags=maxlag)
        plt.title('ACF for ' + str(name))
        plt.savefig(name + '_ACF.png')


def grangercausality(timeseries, lineage, regionlist, maxlag, alpha):
    """Check for Granger Causality"""

    test = "ssr_chi2test"

    for loc1, loc2 in pairwise(regionlist):
        data = pd.concat([timeseries[lineage + '_' + loc1], timeseries[lineage + '_' +  loc2]], axis=1)
        try:
            test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
            min_p_value = min(p_values, key=lambda x: x[0])
            filename = str(lineage) + '_log.txt'
            appendline(filename, 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + '(min p-value, lag) = ' + str(min_p_value) + ' =>  ' + str(min_p_value[0] <= alpha))
        except:
            appendline(filename, 'Granger causality test for ' + str(loc2) +  '-> ' + str(loc1) + ' failed')


def cointegration(timeseries, lineage, regionlist, maxlag):
    """Perform Johanson's Cointegration Test"""

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
                runVECM = True

    return runVECM


def adfullertest(timeseries, lineage, alpha):
    """Perform ADFuller to test for Stationarity"""

    adf_result = []
    filename = str(lineage) + '_log.txt'
    for name, col in timeseries.iteritems():
        stat = adfuller(col, autolag ="AIC")
        out = {'test_statistic':round(stat[0], 4), 'pvalue':round(stat[1], 4), 'n_lags':round(stat[2], 4), 'n_obs':stat[3]}
        p_value = out['pvalue']
        appendline(filename, 'Augmented Dickey-Fuller Test for ' + str(name))
        appendline(filename, 'Test Statistic = ' + str(out["test_statistic"]))
        appendline(filename, 'No. Lags Chosen = ' + str(out["n_lags"]))
        if p_value <= alpha:
            appendline(filename, '=> P-Value = ' +  str(p_value) + ' => Reject Null Hypothesis')
            appendline(filename, '=> Series is Stationary')
            adf_result.append(True)
        else:
            appendline(filename, '=> P-Value = ' + str(p_value))
            appendline(filename, '=> Series is Non-Stationary')
            adf_result.append(False)

    return adf_result

