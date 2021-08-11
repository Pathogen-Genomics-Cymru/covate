import os
import statsmodels.tools.sm_exceptions as statserror
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank, coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from .utils import appendline, pairwise, getdate

def buildmodel(timeseries, lineagelist, regionlist, output):
    """ Run stats tests for each lineage and select model and parameters"""

    maxlag=14
    nsteps=14
    alpha = 0.05

    for lineage in lineagelist:

        # set log file
        path = os.path.join(output, str(getdate()), lineage, 'logs')
        filename = path + '/' + lineage + '_log.txt'
        errorlog = os.path.join(output, str(getdate()), 'error_log.txt')

        # filter timeseries by lineage
        lineagestr = str(lineage) + '_'
        X_train = timeseries.loc[:, timeseries.columns.str.startswith(lineagestr)]

        # set index freq
        X_train = X_train.asfreq('d')

        # get basic information on timeseries
        checkdistribution(X_train, lineage, alpha, filename)

        # plot the autocorrelation
        plotautocorr(X_train, lineage, maxlag, output)

        # check for granger causality
        try:
            for loc1, loc2 in pairwise(regionlist):
                grangercausality(X_train, lineage, loc1, loc2, maxlag, alpha, filename, errorlog)
        except statserror.InfeasibleTestError:
            appendline(filename, 'ERROR: Cannot run Granger causality test for ' + str(loc2) +  '-> ' + str(loc1))
            appendline(errorlog, str(lineage) + ' ERROR: Cannot run Granger causality test for ' + str(loc2) +  '-> ' + str(loc1))
            continue

        # find lag order
        try:
            lag = lagorder(X_train, lineage, maxlag, filename)
        except np.linalg.LinAlgError:
            appendline(filename, 'ERROR: Cannot compute lag order')
            appendline(errorlog, str(lineage) + ' ERROR: Cannot compute lag order')
            continue

        # record deterministic terms for cointegration
        VECMdeterm = ''

        # first check cointegration for constant term and linear trend, note if VECMdeterm='colo' then coint_count for 'lo' is selected
        try:
            # first check cointegration for constant term and linear trend, note if VECMdeterm='colo' then coint_count for 'lo' is selected
            for determ in range(0, 2):

                runVECM, coint_count = cointegration(X_train, lineage, regionlist, lag, determ, filename)

                VECMdeterm+= str(runVECM)

            # if no constant or linear determ then check cointegration for no determ
            if not VECMdeterm:

                runVECM, coint_count = cointegration(X_train, lineage, regionlist, lag, -1, filename)

                VECMdeterm+= str(runVECM)

        except np.linalg.LinAlgError:
            appendline(filename, 'ERROR: Cannot run cointegration test')
            appendline(errorlog, str(lineage) + ' ERROR: Cannot run cointegration test')
            continue

        # build model
        try:
            # if lineage has cointegration, then run VECM
            if VECMdeterm:

                appendline(filename, 'Lineage has cointegration => Run VECM')

                vecerrcorr(X_train, lineage, VECMdeterm, lag, coint_count, regionlist, nsteps, alpha, filename, output, errorlog)

            # else check for stationarity and difference then run VAR
            else:

                appendline(filename, 'Lineage has no cointegration => Run VAR')

                vecautoreg(X_train, lineage, lag, regionlist, nsteps, alpha, filename, output)

        except np.linalg.LinAlgError:
            appendline(filename, 'ERROR: Cannot build model')
            appendline(errorlog, str(lineage) + ' ERROR: Cannot build model')
            continue

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


def plotautocorr(X_train, lineage, maxlag, output):
    """Plot autocorrelation"""

    path = os.path.join(output, str(getdate()), lineage, 'additional-plots')

    for name, col in X_train.iteritems():
        plot_acf(col, lags=maxlag)
        plt.title('ACF for ' + str(name))
        plt.savefig(path + '/' + name + '_ACF.png')
        plt.clf()
        plt.close()


def grangercausality(X_train, lineage, loc1, loc2, maxlag, alpha, filename, errorlog):
    """Check for Granger Causality"""

    test = "ssr_chi2test"

    data = pd.concat([X_train[lineage + '_' + loc1], X_train[lineage + '_' +  loc2]], axis=1)

    test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
    min_p_value = min(p_values, key=lambda x: x[0])

    appendline(filename, 'Granger causality test result for ' + str(loc2) + '->' + str(loc1) + "\n" + 'minimum p-value = ' + str(round(min_p_value[0], 4)) + ' =>  ' + str(min_p_value[0] <= alpha))

    if min_p_value[0] >= alpha:
        appendline(filename, 'WARN: No Granger causality for ' + str(loc2) + '->' + str(loc1))
        appendline(errorlog, str(lineage) + ' WARN: No Granger causality for ' + str(loc2) + '->' + str(loc1))


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


def vecerrcorr(X_train, lineage, VECMdeterm, lag, coint_count, regionlist, nsteps, alpha, filename, output, errorlog):
    """ Build VECM model"""

    # minus 1 from lag for VECM
    if lag != 0:
        lag = int(lag) - 1

    # predict on entire dataset
    vecm = VECM(endog = X_train, k_ar_diff = lag, coint_rank = coint_count, deterministic = VECMdeterm)

    vecm_fit = vecm.fit()

    try:
        appendline(filename, vecm_fit.summary().as_text())
    except IndexError:
        appendline(filename, 'WARN: Failed to create VECM summary')
        appendline(errorlog, str(lineage) + ' WARN: Failed to create VECM summary')

    vecm_fit.predict(steps=nsteps)

    forecast, lower, upper = vecm_fit.predict(nsteps, alpha)

    # get last index from X_train and build index for prediction
    idx = pd.date_range(X_train.index[-1], periods=nsteps+1, freq='d')[1:] 

    pred = (pd.DataFrame(forecast.round(0), columns=X_train.columns, index=idx))

    # cast negative predictions to zero
    pred[pred<0] = 0

    path = os.path.join(output, str(getdate()), lineage, 'prediction')

    for region in regionlist:
        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')
        plt.title('VECM Predicted Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region + '_VECM.png')
        plt.clf()
        plt.close()

    # build testing dataset for validation
    X_train, X_test = X_train[0:-nsteps], X_train[-nsteps:]

    vecm = VECM(endog = X_train, k_ar_diff = lag, coint_rank = coint_count, deterministic = VECMdeterm)

    vecm_fit = vecm.fit()
    vecm_fit.predict(steps=nsteps)

    forecast, lower, upper = vecm_fit.predict(nsteps, alpha)

    pred = (pd.DataFrame(forecast.round(0), index=X_test.index, columns=X_test.columns))

    # cast negative predictions to 0
    pred[pred<0] = 0

    path = os.path.join(output, str(getdate()), lineage, 'validation')

    for region in regionlist:
        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')
        plt.title('Validation: VECM Predicted and Actual Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region + '_VECM_validation.png')
        plt.clf()
        plt.close()


def vecautoreg(X_train, lineage, lag, regionlist, nsteps, alpha, filename, output):
    """ Build VAR model"""

    # predict on entire dataset
    Xtrain = X_train.copy()

    # check for stationarity and difference

    adf_result = adfullertest(Xtrain, lineage, alpha, filename)

    # record whether no diff, first diff or second diff

    VARdiff = 'none'

    # first diff and recalculate ADF
    if False in adf_result:

        Xtrain = Xtrain.diff().dropna()

        adf_result = adfullertest(Xtrain, lineage, alpha, filename)

        VARdiff = 'first'

        appendline(filename, 'Series has been first differenced')

    # second diff
    if False in adf_result:

         Xtrain = Xtrain.diff().dropna()

         adf_result = adfullertest(Xtrain, lineage, alpha, filename)

         VARdiff = 'second'

         appendline(filename, 'Series has been second differenced')

    # build var
    varm = VAR(endog = Xtrain)

    varm_fit = varm.fit(maxlags=lag)

    lag_order = varm_fit.k_ar

    forecast_input = Xtrain.values[-lag_order:]

    forecast = varm_fit.forecast(y=forecast_input, steps=nsteps)

    # get last index from X_train and build index for prediction
    idx = pd.date_range(Xtrain.index[-1], periods=nsteps+1, freq='d')[1:]

    pred = pd.DataFrame(forecast.round(0), index=idx, columns=Xtrain.columns + '_diff')

    # undo difference
    fc = pred.copy()
    columns = Xtrain.columns
    for col in columns:
        if VARdiff == 'first':
            fc[str(col)] = Xtrain[col].iloc[-1] + fc[str(col)+'_diff'].cumsum()
        elif VARdiff == 'second':
            fc[str(col)+'_1d'] = (Xtrain[col].iloc[-1] - Xtrain[col].iloc[-2]) + fc[str(col)+'_diff'].cumsum()
            fc[str(col)] = Xtrain[col].iloc[-1] + fc[str(col)+'_1d'].cumsum()

    # cast negative predictions to 0
    fc[fc<0] = 0

    path = os.path.join(output, str(getdate()), lineage, 'prediction')

    for region in regionlist:
        plt.plot(fc[lineage + '_' + region], color='r', label='prediction')
        plt.title('VAR Predicted Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region + '_VAR.png')
        plt.clf()
        plt.close()

    # build testing dataset for validation

    X_train, X_test = X_train[0:-nsteps], X_train[-nsteps:]

    # check for stationarity and difference

    adf_result = adfullertest(X_train, lineage, alpha, filename)

    # record whether no diff, first diff or second diff

    VARdiff = 'none'

    # first diff and recalculate ADF
    if False in adf_result:

        X_train = X_train.diff().dropna()

        adf_result = adfullertest(X_train, lineage, alpha, filename)

        VARdiff = 'first'

        appendline(filename, 'Series has been first differenced')

    # second diff
    if False in adf_result:

         X_train = X_train.diff().dropna()

         adf_result = adfullertest(X_train, lineage, alpha, filename)

         VARdiff = 'second'

         appendline(filename, 'Series has been second differenced')

    # build var
    varm = VAR(endog = X_train)

    varm_fit = varm.fit(maxlags=lag)

    lag_order = varm_fit.k_ar

    forecast_input = X_train.values[-lag_order:]

    forecast = varm_fit.forecast(y=forecast_input, steps=nsteps)
    pred = pd.DataFrame(forecast.round(0), index=X_test.index, columns=X_test.columns + '_diff')

    # undo difference
    fc = pred.copy()
    columns = X_train.columns
    for col in columns:
        if VARdiff == 'first':
            fc[str(col)] = X_train[col].iloc[-1] + fc[str(col)+'_diff'].cumsum()
        elif VARdiff == 'second':
            fc[str(col)+'_1d'] = (X_train[col].iloc[-1] - X_train[col].iloc[-2]) + fc[str(col)+'_diff'].cumsum()
            fc[str(col)] = X_train[col].iloc[-1] + fc[str(col)+'_1d'].cumsum()

    # cast negative predictions to 0
    fc[fc<0] = 0

    path = os.path.join(output, str(getdate()), lineage, 'validation')

    for region in regionlist:
        plt.plot(fc[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')
        plt.title('Validation: VAR Predicted and Actual Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region + '_VAR_validation.png')
        plt.clf()
        plt.close()

