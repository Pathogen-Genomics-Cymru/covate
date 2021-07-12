from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
from statsmodels.tsa.api import VAR
import pandas as pd
import matplotlib.pyplot as plt

def buildmodel(timeseries, lineageVECM, VECMdeterm, lineageVAR, VARdiff, regionlist):

    nsteps=14

    for lineage, determ in zip(lineageVECM, VECMdeterm):

        vectorErrorCorr(timeseries, lineage, determ, regionlist, nsteps)


def vectorErrorCorr(timeseries, lineage, determ, regionlist, nsteps):

    data = timeseries.filter(like=lineage)
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

    for region in regionlist:
        plt.plot(pred[lineage + '_' + region], color='r', label='prediction')
        plt.plot(X_test[lineage + '_' + region], color='b',  label='actual')
        plt.title('VECM Predicted and Actual Time series for ' +  lineage + ' for ' + region)
        plt.legend(loc="upper left")
        plt.savefig(lineage + '_' + region + '_VECM.png')
        plt.clf()
