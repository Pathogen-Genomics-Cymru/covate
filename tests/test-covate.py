#!/usr/bin/env python3

import subprocess
import os
import pandas as pd

def go():

     toRun = 'covate -i test-metadata.csv -o output -c -v -e 30/8/2021' 
     result = subprocess.run([toRun], shell=True)

     filelist = ['UKX_prediction.csv', 'UKX_validation.csv', 'UKY_prediction.csv', 'UKY_validation.csv']

     for filename in filelist:

         topdir = filename.split('_')[0]
         logdir = filename.split('_')[1].split('.')[0]

         dftrue = pd.read_csv('test-output/' + str(filename))
         dftest = pd.read_csv('output/Aug-30-2021/' + str(topdir) + '/logs/' + str(logdir) + '/' + str(filename))

         #print(dftest.eq(dftrue).to_string(index=True))

         falsearr = dftest[dftest.eq(dftrue).all(axis=1) == False]

         if falsearr.empty:
            print (str(filename) + ' passed integration test')
         else:
            raise ValueError(str(filename) + ' failed integration run')


def main():
    go()

if __name__ == '__main__':
    main()
