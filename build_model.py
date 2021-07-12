from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from utils import appendline, pairwise

def buildmodel(timeseries, lineagelist, regionlist):

    maxlag=14
    alpha = 0.05
    nsteps=14

    for lineage in lineagelist:

        # filter timeseries by lineage
        data = timeseries.filter(like=lineage)

        # get basic information on timeseries
        checkdistribution(data, lineage, alpha)

        # plot the autocorrelation
        plotautocorr(data, lineage, maxlag)

        # check for granger causality
        grangercausality(data, lineage, regionlist, maxlag, alpha)

        # record deterministic terms for cointegration
        VECMdeterm = ''

        # first check cointegration for constant term and linear trend
        for determ in range(0, 2):

            runVECM = cointegration(timeseries, lineage, regionlist, maxlag, determ)

            VECMdeterm+= str(runVECM)

        # if no constant or linear determ then check cointegration for no determ
        if not VECMdeterm:

            runVECM = cointegration(data, lineage, regionlist, maxlag, -1)

        # if lineage has cointegration, then run VECM
        if VECMdeterm:

            vectorErrorCorr(data, lineage, VECMdeterm, regionlist, nsteps)

        # else check for stationarity and difference then run VAR
        else:

            adf_result = adfullertest(data, lineagelist, alpha)

            # record whether no diff, first diff or second diff
            VARdiff = 'none'

            # first diff and recalculate ADF
            if False in adf_result:
                first_diff = data.diff().dropna()
                adf_result = adfullertest(first_diff, lineagelist, alpha)
                VARdiff = 'first'
                data = first_diff

            # second diff
            if False in adf_result:
                second_diff = first_diff.diff().dropna()
                adf_result = adfullertest(second_diff, lineagelist, alpha)
                VARdiff = 'second'
                data = second_diff


def checkdistribution(data, lineage, alpha):
    """Get information about distribution"""

    #data.reset_index(drop=True, inplace=True)
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


def plotautocorr(data, lineage, maxlag):
    """Plot autocorrelation"""

    #data.reset_index(drop=True, inplace=True)
    for name, col in data.iteritems():
        plot_acf(col, lags=maxlag)
        plt.title('ACF for ' + str(name))
        plt.savefig(name + '_ACF.png')
        plt.clf()

def grangercausality(data, lineage, regionlist, maxlag, alpha):
    """Check for Granger Causality"""

    test = "ssr_chi2test"

    for loc1, loc2 in pairwise(regionlist):
        data = pd.concat([data[lineage + '_' + loc1], data[lineage + '_' +  loc2]], axis=1)
        try:
            test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
            min_p_value = min(p_values, key=lambda x: x[0])
            filename = str(lineage) + '_log.txt'
            appendline(filename, 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + '(min p-value, lag) = ' + str(min_p_value) + ' =>  ' + str(min_p_value[0] <= alpha))
        except:
            appendline(filename, 'Granger causality test for ' + str(loc2) +  '-> ' + str(loc1) + ' failed')


def cointegration(timeseries, lineage, regionlist, maxlag, determ):
    """Perform Johanson's Cointegration Test"""

    runVECM = ''

    data = timeseries.filter(like=lineage)
    data.reset_index(drop=True, inplace=True)
    filename = str(lineage) + '_log.txt'
    appendline(filename, 'Lags with Cointegration for ' + str(determ))
    for lag in range(maxlag):
        d = {'0.90':0, '0.95':1, '0.99':2}

        out = coint_johansen(data, determ, lag)
        out_traces = out.lr1
        out_cvts = out.cvt[:, d[str(0.95)]]

        for col, trace, cvt in zip(data.columns, out_traces, out_cvts):
            if trace > cvt:
                appendline(filename, str(col) + ' lag= ' + str(lag))
                if determ == 0:
                    runVECM = 'co'
                elif determ == 1:
                    runVECM = 'lo'
                elif determ == -1:
                    runVECM = 'nc'

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


def vectorErrorCorr(data, lineage, determ, regionlist, nsteps):
    """ Build VECM model"""

    data = data.asfreq('d')

    nobs = nsteps
    X_train, X_test = data[0:-nobs], data[-nobs:]

    varmodel = VAR(endog=X_train)
    lagorder = varmodel.select_order(14, trend='c')
    lag = int(lagorder.aic) - 1

    vecm_rank = select_coint_rank(X_train, det_order = 1, k_ar_diff = lag, method = 'trace', signif=0.05)

    vecm = VECM(endog = X_train, k_ar_diff = lag, coint_rank = vecm_rank.rank, deterministic = determ)

    vecm_fit = vecm.fit()
    vecm_fit.predict(steps=nsteps)

    forecast, lower, upper = vecm_fit.predict(nsteps, 0.05)

    pred = (pd.DataFrame(forecast.round(0), index=X_test.index, columns=X_test.columns))

    filename = str(lineage) + '_log.txt'
    appendline(filename, lagorder.summary().as_text())
    appendline(filename, vecm_rank.summary().as_text())
    appendline(filename, vecm_fit.summary().as_text())

    for region in regionlist:
        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')
        plt.title('VECM Predicted and Actual Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.savefig(lineage + '_' + region + '_VECM.png')
        plt.clf()
