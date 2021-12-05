#!/usr/bin/env python3

import subprocess
import pandas as pd


def go():

    runcovate()

    checktimeseries()

    checkcrosscorr()

    checkpredictvalidate()

    comparepredictvalidate()


def comparepredictvalidate():
    """check 16/8/2021 prediction curve is the same as the 30/8/2021 validation curve"""

    lineagelist = ['UKX', 'UKY']

    for lineage in lineagelist:

        dfpredict = pd.read_csv('output/Aug-16-2021/' + str(lineage) + '/logs/prediction/' + str(lineage) + '_prediction.csv')
        dfvalidate = pd.read_csv('output/Aug-30-2021/' + str(lineage) + '/logs/validation/' + str(lineage) + '_validation.csv', usecols=[0, 1, 2])

        falsearr = dfvalidate[dfvalidate.eq(dfpredict).all(axis=1) == False]

        if falsearr.empty:
            print(str(lineage) + ' passed integration test comparing 16/8 prediction with the 30/8 validation')
        else:
            raise ValueError(str(lineage) + ' failed integration test comparing 16/8 prediction with the 30/8 validation')


def checktimeseries():
    """compare the time series csvs for 30/8/2021 against truth set"""

    filelist = ['timeseriesall.csv', 'timeseriescommon.csv']

    for filename in filelist:

        dftrue = pd.read_csv('test-output/' + str(filename))
        dftest = pd.read_csv('output/Aug-30-2021/' + str(filename))

        falsearr = dftest[dftest.eq(dftrue).all(axis=1) == False]

        if falsearr.empty:
            print(str(filename) + ' passed integration test')
        else:
            raise ValueError(str(filename) + ' failed integration test')


def checkcrosscorr():
    """compare the cross-correlation csvs for 30/8/2021 against truth set"""

    filelist = ['lineagefreqbyregion.csv', 'maxcorrlag.csv', 'maxcorrlag05.csv', 'Wales_lineagefreq.csv']

    for filename in filelist:

        dftrue = pd.read_csv('test-output/' + str(filename))
        dftest = pd.read_csv('output/Aug-30-2021/cross-correlation/' + str(filename))

        falsearr = dftest[dftest.eq(dftrue).all(axis=1) == False]

        if falsearr.empty:
            print(str(filename) + ' passed integration test')
        else:
            raise ValueError(str(filename) + ' failed integration test')


def checkpredictvalidate():
    """compare the prediction and validation csvs for 30/8/2021 against truth set"""

    filelist = ['UKX_prediction.csv', 'UKX_validation.csv', 'UKY_prediction.csv', 'UKY_validation.csv']

    for filename in filelist:

        topdir = filename.split('_')[0]
        logdir = filename.split('_')[1].split('.')[0]

        dftrue = pd.read_csv('test-output/' + str(filename))
        dftest = pd.read_csv('output/Aug-30-2021/' + str(topdir) + '/logs/' + str(logdir) + '/' + str(filename))

        falsearr = dftest[dftest.eq(dftrue).all(axis=1) == False]

        if falsearr.empty:
            print(str(filename) + ' passed integration test')
        else:
            raise ValueError(str(filename) + ' failed integration test')


def runcovate():
    """Run covate using test metadata for 30/8/2021 and 16/8/2021"""

    toRun = 'covate -i test-metadata.csv -o output -p -c -v -e 30/8/2021'
    result = subprocess.run([toRun], shell=True)

    toRun2 = 'covate -i test-metadata.csv -o output -p -c -v -e 16/8/2021'
    result2 = subprocess.run([toRun2], shell=True)


def main():
    go()


if __name__ == '__main__':
    main()
