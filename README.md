![Build Status](https://github.com/Pathogen-Genomics-Cymru/covate/workflows/Covate-CI/badge.svg)
# covate #
Covate uses the COG-UK metadata to forecast the time series for lineages of SARS-CoV-2 that are common to a specified list of regions. It can also be used to investigate the likelihood of lineages being imported between regions.

Covate consists of three analyses:
1) **PREDICT, --predict ,-p** <br />
Covate can forecast the time series of sequenced cases for lineages that are common to all the regions. The selection of either a VAR or VECM model is automated on a per lineage basis from the results of a cointegration test. The selection of parameters for the chosen model is also automated.

2) **VALIDATE, --validate, -v** <br />
Covate can also build validation forecasts for existing metadata. For example, the validation forecast from 30/8/2021 would be a replicate of the prediction forecast from 16/8/2021 (when running with default parameters). The validation forecast is plotted against the actual time series.

3) **CROSS-CORRELATION, --cross-correlation, -c** <br />
Covate can run a cross-correlation analysis that investigates the likelihood of lineages of SARS-CoV-2 being imported between the regions.

## Install ##
The recommended Python versions for running covate are 3.7.x - 3.9.x (other versions may work but are untested). 

For stability, it is recommended you download the latest [release](https://github.com/Pathogen-Genomics-Cymru/covate/releases) and install using `python setup.py install`

To install the very latest updates (as on main branch) you can use pip with git+:
```
 pip install git+https://github.com/Pathogen-Genomics-Cymru/covate.git
```

## Usage ##

To run all three analyses with default arguments:
```
covate -i metadata.csv -o output_dir -p -v -c
```
A full description of the available arguments and their default values can be found below.


Help message:
```
usage: covate [-h] -i METADATA -o OUTPUT [-r REGIONS] [-a ADM]
              [-l LINEAGETYPE] [-t TIMEPERIOD] [-e ENDDATE] [-p] [-v] [-c]
              [-f PRIMARYREGION] [-m MAXLAGS] [-n NSTEPS]

optional arguments:
  -h, --help            show this help message and exit
  -i METADATA, --input-csv METADATA
                        Input metadata csv, expects columns: cog_id,
                        adm1/adm2, sample_date, lineage/uk_lineage
  -o OUTPUT, --output-dir OUTPUT
                        Output directory for the results
  -r REGIONS, --region-list REGIONS
                        Input list of regions to compare, e.g. Wales, England
  -a ADM, --adm ADM     Select either adm1 or adm2
  -l LINEAGETYPE, --lineage-type LINEAGETYPE
                        Select either lineage or uk_lineage
  -t TIMEPERIOD, --time-period TIMEPERIOD
                        Select time period in weeks to take from metadata
  -e ENDDATE, --end-date ENDDATE
                        Select end date to take from metadata. Format: d/m/Y
  -p, --predict         Run prediction forecast
  -v, --validate        Run validation forecast
  -c, --cross-correlation
                        Run cross-correlation analysis
  -f PRIMARYREGION, --primary-region PRIMARYREGION
                        Region of primary interest for cross-correlation
  -m MAXLAGS, --max-lags MAXLAGS
                        Maximum number of lags to investigate
  -n NSTEPS, --n-steps NSTEPS
                        Number of days to predict
```

### Arguments ###
* **--input-csv** <br /> Input metadata csv. The following columns are required: **cog_id, adm1/adm2, sample_date, lineage/uk_lineage**
* **--output-dir** <br /> Output directory for results
* **--region-list** <br /> Input list of regions to compare. Default **Wales, England**
* **--adm** <br /> Select adm the regions belong to (adm1 or adm2). Default **adm1**
* **--lineage-type** <br /> Select whether to compare global or uk lineages (lineage or uk_lineage). Default **uk_lineage**
* **--time-period** <br /> Select time period in weeks to take from the input metadata csv. Default **12**
* **--end-date** <br /> The end date of the time period to take from the input metadata csv. Expected format is d/m/Y, e.g. 31/7/2021. Default **latest date in the metadata -7 days** (to account for lag in data)
* **--predict** <br /> If specified, prediction forecasts will be created
* **--validate** <br /> If specified, validation forecasts will be created
* **--cross-correlation** <br /> If specifed, cross-correlation analysis will be run
* **--primary-region** <br /> Primary region for cross-correlation analysis. Default **Wales**
* **--max-lags** <br /> Select maximum number of lags to investigate. Default **14**
* **--n-steps** <br /> Number of days to predict. Default **14**

## Workflow ##
<img height="600" src="https://github.com/Pathogen-Genomics-Cymru/covate/blob/main/covate-workflow.png" />

## Output ##
A date-stamped output directory is created with sub-directories for each common lineage and a cross-correlation sub-directory. At the top level you will find a csv of the timeseries and summary error log file(s) for prediction and validation (provided --predict and --validate). The cross-correlation sub-directory contains multiple plots and csvs from the cross-correlation analysis (provided --cross-correlation). In a lineage sub-directory you should find the following directories and plots:
* **prediction** The forecasted time series for each region. This directory will be empty if --predict is not specified. If --predict has been specified and directory is empty then the forecast has failed to run (check logs/prediction for the error log).
* **validation** A validation forecast for each region (plots the time series for the last nsteps prior to the set end date with a forecast). This directory will be empty if --validate is not specified. If --validate has been specified and the directory is empty then the forecast has failed to run (check logs/validation for the error log).
* **logs** There are separate log files for prediction and validation. Log files $lineage_model.txt contain information on the built models. If there are any errors raised for the lineage then an error log $lineage_error.txt will also be generated. There are also csvs of the forecasted time series values.
* **additional-plots** Time series for the lineage and ACF plots for each region. There may be additional VAR plots if relevant.

### Error Log ###
There are separate error log files for prediction and validation. The error logs will likely contain ERROR and WARN messages for some lineages. ERROR messages indicate a fatal error where the code was unable to build a model for a lineage due to poor quality data. WARN messages indicate a non-fatal error, in this case the model should build for a lineage, but the message may indicate that the model might not be accurate (e.g. A WARN message is recorded if causality is not found). 
