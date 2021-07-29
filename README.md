# covate #
Covate forecasts time series for lineages that are common to a specified list of regions. The selection of either a VAR or VECM model is automated on a per lineage basis, based on the results of a cointegration test. The selection of parameters for the chosen model is also automated based on the results of stats tests.

## Usage ##
```
usage: covate.py [-h] -i METADATA [-r REGIONS] [-a ADM] [-l LINEAGETYPE]
                 [-t TIMEPERIOD]

optional arguments:
  -h, --help            show this help message and exit
  -i METADATA, --input-csv METADATA
                        Input metadata csv, expects columns:
                        central_sample_id, adm1/adm2, sample_date,
                        lineage/uk_lineage
  -r REGIONS, --region-list REGIONS
                        Input list of regions to compare, e.g. UK-WLS, UK-ENG
  -a ADM, --adm ADM     Select either adm1 or adm2
  -l LINEAGETYPE, --lineage-type LINEAGETYPE
                        Select either lineage or uk_lineage
  -t TIMEPERIOD, --time-period TIMEPERIOD
                        Select time period in months to take from metadata
```

### Arguments ###
* **--input-csv** <br /> Input metadata csv. The following columns are required: **central_sample_id, adm1/adm2, sample_date, lineage/uk_lineage**
* **--region-list** <br /> Input list of regions to compare. Default **UK-WLS, UK-ENG**
* **--adm** <br /> Select adm the regions belong to (adm1 or adm2). Default **adm1**
* **--lineage-type** <br /> Select whether to compare global or uk lineages (lineage or uk_lineage). Default **uk_lineage**
* **--time-period** <br /> Select time period in months to take from the input metadata csv. Default **6**

## Output ##
A date-stamped output directory is created with sub-directories for each common lineage. In a lineage sub-directory you should find the following directories and plots:
* **prediction** The forecasted time series for each region
* **validation** A validation forecast for each region (plots the time series for the last two weeks of the metadata with a forecast)
* **logs** A log file containing results of the stats tests
* **additional-plots** Time series for the lineage and ACF plots for each region
