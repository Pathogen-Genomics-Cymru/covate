# covate #
Covate forecasts time series for lineages that are common to a specified list of regions. The selection of either a VAR or VECM model is automated on a per lineage basis, based on the results of a cointegration test. The selection of parameters for the chosen model is also automated based on the results of stats tests.

## Install ##
```
 pip install git+https://github.com/Pathogen-Genomics-Cymru/covate.git
```

## Usage ##

To run with default parameters:
```
covate -i metadata.csv -o output_dir
```

Help message:
```
usage: covate [-h] -i METADATA -o OUTPUT [-r REGIONS] [-a ADM]
              [-l LINEAGETYPE] [-t TIMEPERIOD] [-e ENDDATE]

optional arguments:
  -h, --help            show this help message and exit
  -i METADATA, --input-csv METADATA
                        Input metadata csv, expects columns:
                        central_sample_id, adm1/adm2, sample_date,
                        lineage/uk_lineage
  -o OUTPUT, --output-dir OUTPUT
                        Output directory for the results
  -r REGIONS, --region-list REGIONS
                        Input list of regions to compare, e.g. UK-WLS, UK-ENG
  -a ADM, --adm ADM     Select either adm1 or adm2
  -l LINEAGETYPE, --lineage-type LINEAGETYPE
                        Select either lineage or uk_lineage
  -t TIMEPERIOD, --time-period TIMEPERIOD
                        Select time period in months to take from metadata
  -e ENDDATE, --end-date ENDDATE
                        Select end date to take from metadata. Format: d/m/Y
```

### Arguments ###
* **--input-csv** <br /> Input metadata csv. The following columns are required: **central_sample_id, adm1/adm2, sample_date, lineage/uk_lineage**
* **--output-dir** <br /> Output directory for results
* **--region-list** <br /> Input list of regions to compare. Default **UK-WLS, UK-ENG**
* **--adm** <br /> Select adm the regions belong to (adm1 or adm2). Default **adm1**
* **--lineage-type** <br /> Select whether to compare global or uk lineages (lineage or uk_lineage). Default **uk_lineage**
* **--time-period** <br /> Select time period in months to take from the input metadata csv. Default **3**
* **--end-date** <br /> The end date of the time period to take from the input metadata csv. Default **latest date in the metadata -7 days** (to account for lag in data)

## Workflow ##
<img height="600" src="https://github.com/Pathogen-Genomics-Cymru/covate/blob/main/covate-workflow.png" />

## Output ##
A date-stamped output directory is created with sub-directories for each common lineage. At the top level you will find a csv of the timeseries and an error log file. In a lineage sub-directory you should find the following directories and plots:
* **prediction** The forecasted time series for each region
* **validation** A validation forecast for each region (plots the time series for the last two weeks of the metadata with a forecast)
* **logs** A log file containing results of the stats tests
* **additional-plots** Time series for the lineage and ACF plots for each region

## Error Log ##
The error log will likely contain ERROR and WARN messages for some lineages. ERROR messages indicate a fatal error where the code was unable to build a model for a lineage due to poor quality data. WARN messages indicate a non-fatal error, in this case the model should build for a lineage, but the message may indicate that the model might not be accurate (e.g. A WARN message is recorded if causality is not found). These ERROR and WARN messages are also duplicated in the individual log file for each lineage.
