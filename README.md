![Build Status](https://github.com/Pathogen-Genomics-Cymru/covate/workflows/Covate-CI/badge.svg)
# covate #
Covate uses the COG-UK metadata to forecast time series for lineages of SARS-CoV-2 that are common to a specified list of regions. The selection of either a VAR or VECM model is automated on a per lineage basis, based on the results of a cointegration test. The selection of parameters for the chosen model is also automated.

In addition, covate can also build validation forecasts for existing metadata, and run a cross-correlation analysis that investigates the likelihood of lineages of SARS-CoV-2 being imported between two regions.

## Install ##
The recommended Python versions for running covate are 3.7.x - 3.9.x (other versions may work but are untested). To install using pip with git+:
```
 pip install git+https://github.com/Pathogen-Genomics-Cymru/covate.git
```
Alternatively, clone the repository and run `python setup.py install`

## Usage ##

To run with default arguments:
```
covate -i metadata.csv -o output_dir
```
A validation forecast will also be run if you use the `--validate` flag, and a cross-correlation analysis will be run if you specify `--cross-correlation`. Note the cross-correlation analysis currently only works for two regions. E.g.:
```
covate -i metadata.csv -o output_dir --validate --cross-correlation --region-list "Wales, England"
```
A full description of the available arguments and their default values can be found below.


Help message:
```
usage: covate [-h] -i METADATA -o OUTPUT [-r REGIONS] [-a ADM]
              [-l LINEAGETYPE] [-t TIMEPERIOD] [-e ENDDATE] [-v] [-c]
              [-p PRIMARYREGION] [-m MAXLAGS] [-n NSTEPS]

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
  -v, --validate        Run validation forecast
  -c, --cross-correlation
                        Run cross-correlation analysis
  -p PRIMARYREGION, --primary-region PRIMARYREGION
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
* **--validate** <br /> If specified, validation forecasts will be created
* **--cross-correlation** <br /> If specifed, cross-correlation analysis will be run
* **--primary-region** <br /> Primary region for cross-correlation analysis. Default **Wales**
* **--max-lags** <br /> Select maximum number of lags to investigate. Default **14**
* **--n-steps** <br /> Number of days to predict. Default **14**

## Workflow ##
<img height="600" src="https://github.com/Pathogen-Genomics-Cymru/covate/blob/main/covate-workflow.png" />

## Output ##
A date-stamped output directory is created with sub-directories for each common lineage and a cross-correlation sub-directory. The cross-correlation directory contains two plots and a csv for the primary region (if --cross-correlation). At the top level you will find a csv of the timeseries and summary error log file(s) for prediction and validation (if --validate). In a lineage sub-directory you should find the following directories and plots:
* **prediction** The forecasted time series for each region. If the directory is empty then the forecast has failed to run (check logs/prediction for the error log).
* **validation** A validation forecast for each region (plots the time series for the last nsteps prior to the set end date with a forecast). This directory will be empty if --validate is not specified. If --validate has been specified and the directory is empty then the forecast has failed to run (check logs/validation for the error log).
* **logs** There are separate log files for prediction and validation. Log files $lineage_model.txt contain information on the built models. If there are any errors raised for the lineage then an error log $lineage_error.txt will also be generated. There are also csvs of the forecasted time series values.
* **additional-plots** Time series for the lineage and ACF plots for each region. There may be additional VAR plots if relevant.

### Error Log ###
There are separate error log files for prediction and validation. The error logs will likely contain ERROR and WARN messages for some lineages. ERROR messages indicate a fatal error where the code was unable to build a model for a lineage due to poor quality data. WARN messages indicate a non-fatal error, in this case the model should build for a lineage, but the message may indicate that the model might not be accurate (e.g. A WARN message is recorded if causality is not found). 
