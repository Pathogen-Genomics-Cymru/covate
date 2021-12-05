import os
import statsmodels.tools.sm_exceptions as statserror
from statsmodels.tsa.vector_ar.vecm import VECM, coint_johansen
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.graphics.tsaplots import plot_acf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from .utils import appendline, pairwise, getenddate
import warnings


def buildmodel(timeseries, lineagelist, regionlist, enddate, output, maxlag,
               nsteps, validate):
    """ Run stats tests for each lineage and select model and parameters"""

    alpha = 0.05

    for lineage in lineagelist:

        # set plt
        plt.rc('axes', titlesize=10)

        # set log files
        if validate:

            path = os.path.join(output, str(getenddate(enddate)), lineage,
                                'logs/validation')

            filename = path + '/' + lineage

            errorlog = os.path.join(output, str(getenddate(enddate)),
                                    'error_log_validation.txt')

        else:

            path = os.path.join(output, str(getenddate(enddate)), lineage,
                                'logs/prediction')

            filename = path + '/' + lineage

            errorlog = os.path.join(output, str(getenddate(enddate)),
                                    'error_log_prediction.txt')

        # filter timeseries by lineage
        lineagestr = str(lineage) + '_'

        # build Xtrain and set index freq
        X_train = timeseries.loc[:, timeseries.columns.str
                                 .startswith(lineagestr)]

        X_train = X_train.asfreq('d')

        # if running validation, build testing dataset
        if validate:

            X_train, X_test = X_train[0:-nsteps], X_train[-nsteps:]

        # get basic information on timeseries
        checkdistribution(X_train, lineage, filename, errorlog)

        # plot the autocorrelation, ignore warnings
        with warnings.catch_warnings():

            warnings.simplefilter("ignore", UserWarning)

            warnings.simplefilter("ignore", RuntimeWarning)

            if validate:

                plotautocorr(X_train, lineage, maxlag, output, enddate,
                             'additional-plots/validation')

            else:

                plotautocorr(X_train, lineage, maxlag, output, enddate,
                             'additional-plots/prediction')

        # check for granger causality
        try:

            for loc1, loc2 in pairwise(regionlist):

                grangercausality(X_train, lineage, loc1, loc2, maxlag, alpha,
                                 filename, errorlog)

        except statserror.InfeasibleTestError:

            appendline(filename + '_error.txt',
                       'ERROR: Cannot run Granger causality test for '
                       + str(loc2) + '-> ' + str(loc1))

            appendline(errorlog, str(lineage)
                       + ' ERROR: Cannot run Granger causality test for '
                       + str(loc2) + '-> ' + str(loc1))

            continue

        # find lag order
        try:

            lag = lagorder(X_train, lineage, maxlag, filename)

        except np.linalg.LinAlgError:

            appendline(filename + '_error.txt',
                       'ERROR: Cannot compute lag order')

            appendline(errorlog, str(lineage)
                       + ' ERROR: Cannot compute lag order')

            continue

        # record deterministic terms for cointegration
        VECMdeterm = ''
        coint = []

        try:
            # first check cointegration for constant term and linear trend
            for determ in reversed(range(0, 2)):

                runVECM, coint_count = cointegration(X_train, lineage,
                                                     regionlist, lag,
                                                     determ, filename)

                VECMdeterm += str(runVECM)

                coint.append(coint_count)

            # if no constant or linear determ check coint for no determ
            if not VECMdeterm:

                runVECM, coint_count = cointegration(X_train, lineage,
                                                     regionlist, lag,
                                                     -1, filename)

                VECMdeterm += str(runVECM)

                coint.append(coint_count)

            # find minimum count over 0, else 0
            try:

                coint_count = min(item for item in coint if item > 0)

            except ValueError:

                coint_count = 0

        except np.linalg.LinAlgError:

            appendline(filename + '_error.txt',
                       'ERROR: Cannot run cointegration test')

            appendline(errorlog, str(lineage)
                       + ' ERROR: Cannot run cointegration test')

            continue

        # build model
        try:
            # if lineage has cointegration, then run VECM
            if VECMdeterm:

                appendline(filename + '_model.txt',
                           'Lineage has cointegration => Run VECM')

                if not validate:

                    vecerrcorr(X_train, lineage, VECMdeterm, lag, coint_count,
                               regionlist, nsteps, alpha, filename, output,
                               errorlog, enddate)

                else:

                    vecerrcorrvalid(X_train, X_test, lineage, VECMdeterm, lag,
                                    coint_count, regionlist, nsteps, alpha,
                                    filename, output, errorlog, enddate)

            # else check for stationarity and difference then run VAR
            else:

                appendline(filename + '_model.txt',
                           'Lineage has no cointegration => Run VAR')

                if not validate:

                    vecautoreg(X_train, lineage, maxlag, regionlist, nsteps,
                               alpha, filename, output, errorlog, enddate)

                else:

                    vecautoregvalid(X_train, X_test, lineage, maxlag,
                                    regionlist, nsteps, alpha, filename,
                                    output, errorlog, enddate)

        except np.linalg.LinAlgError:

            appendline(filename + '_error.txt', 'ERROR: Cannot build model')

            appendline(errorlog, str(lineage) + ' ERROR: Cannot build model')

            continue


def checkdistribution(X_train, lineage, filename, errorlog):
    """check distribution for null values"""

    for name, col in X_train.iteritems():

        nonzero = np.count_nonzero(col)
        percent_missing = 100 - (nonzero*100 / len(col))

        if percent_missing >= 75:

            appendline(filename + '_error.txt', 'WARN: ' + str(name)
                       + ' has over 75% null values')

            appendline(errorlog, str(lineage) + ' WARN: ' + str(name)
                       + ' has over 75% null values')


def plotautocorr(X_train, lineage, maxlag, output, enddate, folder):
    """Plot autocorrelation"""

    path = os.path.join(output, str(getenddate(enddate)), lineage, folder)

    for name, col in X_train.iteritems():

        plot_acf(col, lags=maxlag)

        plt.title('ACF for ' + str(name))
        plt.ylabel('Autocorrelation')
        plt.xlabel('Lag (days)')
        plt.savefig(path + '/' + name + '_ACF.png')
        plt.clf()
        plt.close()


def grangercausality(X_train, lineage, loc1, loc2, maxlag, alpha, filename,
                     errorlog):
    """Check for Granger Causality"""

    test = "ssr_chi2test"

    data = pd.concat([X_train[lineage + '_' + loc1],
                      X_train[lineage + '_' + loc2]], axis=1)

    test_result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
    p_values = [test_result[lag+1][0][test][1:3] for lag in range(maxlag)]
    min_p_value = min(p_values, key=lambda x: x[0])

    appendline(filename + '_model.txt', 'Granger causality test result for '
               + str(loc2) + '->' + str(loc1) + "\n" + 'minimum p-value = '
               + str(round(min_p_value[0], 4)) + ' =>  '
               + str(min_p_value[0] <= alpha))

    if min_p_value[0] >= alpha:

        appendline(filename + '_error.txt', 'WARN: No Granger causality for '
                   + str(loc2) + '->' + str(loc1))

        appendline(errorlog, str(lineage) + ' WARN: No Granger causality for '
                   + str(loc2) + '->' + str(loc1))


def lagorder(X_train, lineage, maxlag, filename):
    """Use VAR to select lag order"""

    varmodel = VAR(endog=X_train)
    lagorder = varmodel.select_order(maxlag)
    lag = int(lagorder.aic)

    appendline(filename + '_model.txt', lagorder.summary().as_text())

    return lag


def cointegration(X_train, lineage, regionlist, lag, determ, filename):
    """Perform Johanson's Cointegration Test"""

    runVECM = ''

    d = {'0.90': 0, '0.95': 1, '0.99': 2}

    if lag != 0:
        lag = int(lag) - 1

    out = coint_johansen(X_train, determ, lag)
    out_traces = out.lr1
    out_cvts = out.cvt[:, d[str(0.95)]]

    count_coint = 0
    for col, trace, cvt in zip(X_train.columns, out_traces, out_cvts):

        if trace > cvt:
            count_coint += 1
            if determ == 0:
                runVECM = 'co'
            elif determ == 1:
                runVECM = 'lo'
            elif determ == -1:
                runVECM = 'n'

    appendline(filename + '_model.txt', 'Cointegration rank for ' + str(determ)
               + ' at lag ' + str(lag) + ' = ' + str(count_coint))

    return runVECM, count_coint


def adfullertest(X_train, lineage, alpha, filename):
    """Perform ADFuller to test for Stationarity"""

    adf_result = []
    for name, col in X_train.iteritems():

        stat = adfuller(col, autolag="AIC")
        out = {'test_statistic': round(stat[0], 4),
               'pvalue': round(stat[1], 4),
               'n_lags': round(stat[2], 4),
               'n_obs': stat[3]}
        p_value = out['pvalue']

        appendline(filename + '_model.txt',
                   'Augmented Dickey-Fuller Test for '
                   + str(name))
        appendline(filename + '_model.txt', 'Test Statistic = '
                   + str(out["test_statistic"]))
        appendline(filename + '_model.txt', 'No. Lags Chosen = '
                   + str(out["n_lags"]))

        if p_value <= alpha:

            appendline(filename + '_model.txt', '=> P-Value = ' + str(p_value)
                       + ' => Reject Null Hypothesis')
            appendline(filename + '_model.txt', '=> Series is Stationary')

            adf_result.append(True)

        else:

            appendline(filename + '_model.txt', '=> P-Value = '
                       + str(p_value))
            appendline(filename + '_model.txt', '=> Series is Non-Stationary')

            adf_result.append(False)

    return adf_result


def vecerrcorr(X_train, lineage, VECMdeterm, lag, coint_count, regionlist,
               nsteps, alpha, filename, output, errorlog, enddate):
    """ Build VECM model"""

    # minus 1 from lag for VECM
    if lag != 0:
        lag = int(lag) - 1

    # predict on entire dataset
    vecm = VECM(endog=X_train, k_ar_diff=lag, coint_rank=coint_count,
                deterministic=VECMdeterm)

    vecm_fit = vecm.fit()

    try:

        appendline(filename + '_model.txt', vecm_fit.summary().as_text())

    except IndexError:

        appendline(filename + '_error.txt',
                   'WARN: Failed to create VECM summary')

        appendline(errorlog, str(lineage)
                   + ' WARN: Failed to create VECM summary')

    vecm_fit.predict(steps=nsteps)

    forecast, lower, upper = vecm_fit.predict(nsteps, alpha)

    # get last index from X_train and build index for prediction
    idx = pd.date_range(X_train.index[-1], periods=nsteps+1, freq='d')[1:]

    pred = (pd.DataFrame(forecast.round(0), columns=X_train.columns,
            index=idx))

    # cast negative predictions to zero
    pred[pred < 0] = 0

    path = os.path.join(output, str(getenddate(enddate)), lineage,
                        'prediction')

    for region in regionlist:

        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')

        plt.title('VECM Predicted Time series for ' + lineage + ' for '
                  + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.ylabel('Number of sequenced cases')
        plt.xlabel('Sample date')
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region + '_VECM.png')
        plt.clf()
        plt.close()

    # create prediction dataframe and save to csv
    for region in regionlist:

        oldname = str(lineage) + '_' + str(region)
        predname = str(lineage) + '_' + str(region) + '_prediction'

        pred.rename(columns={oldname: predname}, inplace=True)
        pred[predname] = pred[predname].astype(int)

    pred.index.name = 'sample_date'
    pred.to_csv(filename + '_prediction.csv')


def vecerrcorrvalid(X_train, X_test, lineage, VECMdeterm, lag, coint_count,
                    regionlist, nsteps, alpha, filename, output, errorlog,
                    enddate):
    """Build VECM model for validation"""

    # minus 1 from lag for VECM
    if lag != 0:
        lag = int(lag) - 1

    # build predict for validation dataset
    vecm = VECM(endog=X_train, k_ar_diff=lag, coint_rank=coint_count,
                deterministic=VECMdeterm)

    vecm_fit = vecm.fit()

    try:
        appendline(filename + '_model.txt', vecm_fit.summary().as_text())

    except IndexError:

        appendline(filename + '_error.txt',
                   'WARN: Failed to create VECM summary')

        appendline(errorlog, str(lineage)
                   + ' WARN: Failed to create VECM summary')

    vecm_fit.predict(steps=nsteps)

    forecast, lower, upper = vecm_fit.predict(nsteps, alpha)

    pred = (pd.DataFrame(forecast.round(0), index=X_test.index,
            columns=X_test.columns))

    # cast negative predictions to 0
    pred[pred < 0] = 0

    path = os.path.join(output, str(getenddate(enddate)), lineage,
                        'validation')

    for region in regionlist:

        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')

        plt.title('Validation: VECM Predicted and Actual Time series for '
                  + lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.ylabel('Number of sequenced cases')
        plt.xlabel('Sample date')
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region
                    + '_VECM_validation.png')
        plt.clf()
        plt.close()

    # create prediction/actual dataframe and save to csv
    for region in regionlist:

        oldname = str(lineage) + '_' + str(region)
        predname = str(lineage) + '_' + str(region) + '_prediction'
        actname = str(lineage) + '_' + str(region) + '_actual'

        pred.rename(columns={oldname: predname}, inplace=True)
        X_test.rename(columns={oldname: actname}, inplace=True)
        pred[predname] = pred[predname].astype(int)

    predact = pred.join(X_test)
    predact.to_csv(filename + '_validation.csv')


def vecautoreg(X_train, lineage, maxlag, regionlist, nsteps, alpha,
               filename, output, errorlog, enddate):
    """ Build VAR model"""

    # check for stationarity and difference

    adf_result = adfullertest(X_train, lineage, alpha, filename)

    # record whether no diff, or first diff

    VARdiff = 'none'

    # first diff and recalculate ADF
    if False in adf_result:

        X_train = X_train.diff().dropna()

        adf_result = adfullertest(X_train, lineage, alpha, filename)

        VARdiff = 'first'

        appendline(filename + '_model.txt',
                   'Series has been first differenced')

    # add warn message if series is still not stationary
    if False in adf_result:

        appendline(filename + '_error.txt', 'WARN: Series is not stationary')

        appendline(errorlog, str(lineage) + ' WARN: Series is not stationary')

    # plot autocorrelation again, ignore warnings
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)

        plotautocorr(X_train, lineage, maxlag, output, enddate,
                     'additional-plots/prediction/VAR')

    # plot series to check it's stationary
    path = os.path.join(output, str(getenddate(enddate)), lineage,
                        'additional-plots/prediction/VAR')

    X_train.plot()

    plt.tight_layout()
    plt.savefig(path + '/' + lineage + '_stationary_check.png')
    plt.clf()
    plt.close()

    # build var
    varm = VAR(endog=X_train)

    # recalculate lag order for transformed series
    lagorder = varm.select_order(maxlag)
    lag = int(lagorder.aic)

    varm_fit = varm.fit(lag)

    forecast_input = X_train.values[-lag:]

    forecast = varm_fit.forecast(y=forecast_input, steps=nsteps)

    # get last index from X_train and build index for prediction
    idx = pd.date_range(X_train.index[-1], periods=nsteps+1,
                        freq='d')[1:]

    pred = pd.DataFrame(forecast.round(0), index=idx,
                        columns=X_train.columns + '_diff')

    # undo difference
    fc = pred.copy()
    columns = X_train.columns
    for col in columns:

        if VARdiff == 'first':

            fc[str(col)] = (X_train[col].iloc[-1]
                            + fc[str(col)+'_diff'].cumsum())

            # shift series by minimum negative value
            minval = np.amin(fc[str(col)])
            fc[str(col)] += abs(minval)

        elif VARdiff == 'none':

            fc[str(col)] = fc[str(col)+'_diff']

            # shift series by minimum negative value
            minval = np.amin(fc[str(col)])
            fc[str(col)] += abs(minval)

    path = os.path.join(output, str(getenddate(enddate)), lineage,
                        'prediction')

    for region in regionlist:

        plt.plot(fc[lineage + '_' + region], color='r', label='prediction')

        plt.title('VAR Predicted Time series for ' + lineage + ' for '
                  + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.ylabel('Number of sequenced cases')
        plt.xlabel('Sample date')
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region + '_VAR.png')
        plt.clf()
        plt.close()

    # create prediction dataframe and save to csv
    for region in regionlist:

        oldname = str(lineage) + '_' + str(region)
        predname = str(lineage) + '_' + str(region) + '_prediction'

        fc.rename(columns={oldname: predname}, inplace=True)
        fc.drop(columns=[oldname + '_diff'], inplace=True)
        fc[predname] = fc[predname].astype(int)

    fc.index.name = 'sample_date'
    fc.to_csv(filename + '_prediction.csv')


def vecautoregvalid(X_train, X_test, lineage, maxlag, regionlist, nsteps,
                    alpha, filename, output, errorlog, enddate):
    """Build VAR model for validation"""

    # check for stationarity and difference
    adf_result = adfullertest(X_train, lineage, alpha, filename)

    # record whether no diff, first diff or second diff

    VARdiff = 'none'

    # first diff and recalculate ADF
    if False in adf_result:

        X_train = X_train.diff().dropna()

        adf_result = adfullertest(X_train, lineage, alpha, filename)

        VARdiff = 'first'

        appendline(filename + '_model.txt',
                   'Series has been first differenced')

    # add warn message if series is still not stationary
    if False in adf_result:

        appendline(filename + '_error.txt',
                   'WARN: Series is not stationary')

        appendline(errorlog, str(lineage)
                   + ' WARN: Series is not stationary')

    # plot autocorrelation again, ignore warnings
    with warnings.catch_warnings():

        warnings.simplefilter("ignore", UserWarning)
        warnings.simplefilter("ignore", RuntimeWarning)

        plotautocorr(X_train, lineage, maxlag, output, enddate,
                     'additional-plots/validation/VAR')

    # plot series to check it's stationary
    path = os.path.join(output, str(getenddate(enddate)), lineage,
                        'additional-plots/validation/VAR')
    X_train.plot()
    plt.tight_layout()
    plt.savefig(path + '/' + lineage + '_stationary_check.png')
    plt.clf()
    plt.close()

    # build var
    varm = VAR(endog=X_train)

    # recalculate lag order
    lagorder = varm.select_order(maxlag)
    lag = int(lagorder.aic)

    varm_fit = varm.fit(lag)

    forecast_input = X_train.values[-lag:]

    forecast = varm_fit.forecast(y=forecast_input, steps=nsteps)
    pred = pd.DataFrame(forecast.round(0), index=X_test.index,
                        columns=X_test.columns + '_diff')

    # undo difference
    fc = pred.copy()
    columns = X_train.columns
    for col in columns:

        if VARdiff == 'first':

            fc[str(col)] = (X_train[col].iloc[-1]
                            + fc[str(col)+'_diff'].cumsum())

            minval = np.amin(fc[str(col)])
            fc[str(col)] += abs(minval)

        elif VARdiff == 'none':

            fc[str(col)] = fc[str(col)+'_diff']

            # shift series by minimum negative value
            minval = np.amin(fc[str(col)])
            fc[str(col)] += abs(minval)

    path = os.path.join(output, str(getenddate(enddate)), lineage,
                        'validation')

    # plot prediction and actual time series
    for region in regionlist:

        plt.plot(fc[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')

        plt.title('Validation: VAR Predicted and Actual Time series for '
                  + lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.locator_params(axis="y", integer=True, tight=True)
        plt.xticks(rotation=45)
        plt.ylabel('Number of sequenced cases')
        plt.xlabel('Sample date')
        plt.tight_layout()
        plt.savefig(path + '/' + lineage + '_' + region
                    + '_VAR_validation.png')
        plt.clf()
        plt.close()

    # create prediction/actual dataframe and save to csv
    for region in regionlist:

        oldname = str(lineage) + '_' + str(region)
        predname = str(lineage) + '_' + str(region) + '_prediction'
        actname = str(lineage) + '_' + str(region) + '_actual'

        fc.rename(columns={oldname: predname}, inplace=True)
        X_test.rename(columns={oldname: actname}, inplace=True)
        fc.drop(columns=[oldname + '_diff'], inplace=True)
        fc[predname] = fc[predname].astype(int)

    predact = fc.join(X_test)
    predact.to_csv(filename + '_validation.csv')
