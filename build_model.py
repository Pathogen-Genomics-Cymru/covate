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
    nsteps=14
    alpha = 0.05

    for lineage in lineagelist:

        # set log file
        filename = str(lineage) + '_log.txt'

        # filter timeseries by lineage
        data = timeseries.filter(like=lineage)

        # set index freq
        data = data.asfreq('d')

        # build training and testing datasets
        X_train, X_test = data[0:-nsteps], data[-nsteps:]

        # get basic information on timeseries
        checkdistribution(X_train, lineage, alpha, filename)

        # plot the autocorrelation
        plotautocorr(X_train, lineage, maxlag)

        # check for granger causality
        grangercausality(X_train, lineage, regionlist, maxlag, alpha, filename)

        # find lag order
        lag = lagorder(X_train, lineage, maxlag, filename)

        # record deterministic terms for cointegration
        VECMdeterm = ''

        # first check cointegration for constant term and linear trend, note if VECMdeterm='colo' then coint_count for 'lo' is selected
        for determ in range(0, 2):

            runVECM, coint_count = cointegration(X_train, lineage, regionlist, lag, determ, filename)

            VECMdeterm+= str(runVECM)

        # if no constant or linear determ then check cointegration for no determ
        if not VECMdeterm:

            runVECM, coint_count = cointegration(X_train, lineage, regionlist, lag, -1, filename)

        # if lineage has cointegration, then run VECM
        if VECMdeterm:

            appendline(filename, 'Lineage has cointegration => Run VECM')

            vectorErrorCorr(X_train, X_test, lineage, VECMdeterm, lag, coint_count, regionlist, nsteps, alpha, filename)

        # else check for stationarity and difference then run VAR
        else:

            adf_result = adfullertest(X_train, lineagelist, alpha, filename)

            # record whether no diff, first diff or second diff
            VARdiff = 'none'

            # first diff and recalculate ADF
            if False in adf_result:

                X_train = X_train.diff().dropna()

                adf_result = adfullertest(X_train, lineagelist, alpha, filename)

                VARdiff = 'first'

            # second diff
            if False in adf_result:

                X_train = X_train.diff().dropna()

                adf_result = adfullertest(X_train, lineagelist, alpha, filename)

                VARdiff = 'second'


def checkdistribution(X_train, lineage, alpha, filename):
    """Get information about distribution"""

    for name, col in X_train.iteritems():
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


def plotautocorr(X_train, lineage, maxlag):
    """Plot autocorrelation"""

    for name, col in X_train.iteritems():
        plot_acf(col, lags=maxlag)
        plt.title('ACF for ' + str(name))
        plt.savefig(name + '_ACF.png')
        plt.clf()


def grangercausality(X_train,  lineage, regionlist, maxlag, alpha, filename):
    """Check for Granger Causality"""

    test = "ssr_chi2test"

    for loc1, loc2 in pairwise(regionlist):
        data = pd.concat([X_train[lineage + '_' + loc1], X_train[lineage + '_' +  loc2]], axis=1)
        try:
            test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
            p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
            min_p_value = min(p_values, key=lambda x: x[0])
            appendline(filename, 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + 'minimum p-value = ' + str(round(min_p_value[0], 4)) + ' =>  ' + str(min_p_value[0] <= alpha))
        except:
            appendline(filename, 'Granger causality test for ' + str(loc2) +  '-> ' + str(loc1) + ' failed')


def lagorder(X_train, lineage, maxlag, filename):
    """Use VAR to select lag order"""

    varmodel = VAR(endog=X_train)
    lagorder = varmodel.select_order(maxlag)
    lag = int(lagorder.aic)

    appendline(filename, lagorder.summary().as_text())

    return lag


def cointegration(X_train, lineage, regionlist, lag, determ, filename):
    """Perform Johanson's Cointegration Test"""

    runVECM = ''

    d = {'0.90':0, '0.95':1, '0.99':2}

    out = coint_johansen(X_train, determ, lag)
    out_traces = out.lr1
    out_cvts = out.cvt[:, d[str(0.95)]]

    count_coint=0
    for col, trace, cvt in zip(X_train.columns, out_traces, out_cvts):
        if trace > cvt:
            count_coint +=1
            if determ == 0:
                runVECM = 'co'
            elif determ == 1:
                runVECM = 'lo'
            elif determ == -1:
                runVECM = 'nc'
    appendline(filename, 'Cointegration rank for ' + str(determ) + ' at lag ' + str(lag) + ' = ' + str(count_coint))

    return runVECM, count_coint


def adfullertest(X_train, lineage, alpha, filename):
    """Perform ADFuller to test for Stationarity"""

    adf_result = []
    for name, col in X_train.iteritems():
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


def vectorErrorCorr(X_train, X_test, lineage, VECMdeterm, lag, coint_count, regionlist, nsteps, alpha, filename):
    """ Build VECM model"""

    # minus 1 from lag for VECM
    if lag != 0:
        lag = int(lag) - 1

    vecm = VECM(endog = X_train, k_ar_diff = lag, coint_rank = coint_count, deterministic = VECMdeterm)

    vecm_fit = vecm.fit()
    vecm_fit.predict(steps=nsteps)

    forecast, lower, upper = vecm_fit.predict(nsteps, alpha)

    pred = (pd.DataFrame(forecast.round(0), index=X_test.index, columns=X_test.columns))

    appendline(filename, vecm_fit.summary().as_text())

    for region in regionlist:
        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')
        plt.title('VECM Predicted and Actual Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(lineage + '_' + region + '_VECM.png')
        plt.clf()
