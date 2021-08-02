#!/usr/bin/env python3

import argparse
from .build_time_series import buildseries
from .build_model import buildmodel

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-csv", dest="metadata", required=True,
                        help="Input metadata csv, expects columns: central_sample_id, adm1/adm2, sample_date, lineage/uk_lineage")
    parser.add_argument("-o", "--output-dir", dest="output", required=True,
                        help="Output directory for the results")
    parser.add_argument("-r", "--region-list", dest="regions", required=False, default="UK-WLS, UK-ENG",
                        help="Input list of regions to compare, e.g. UK-WLS, UK-ENG")
    parser.add_argument("-a", "--adm", dest="adm", required=False, default="adm1",
                        help="Select either adm1 or adm2")
    parser.add_argument("-l", "--lineage-type", dest="lineagetype", required=False, default="uk_lineage",
                        help="Select either lineage or uk_lineage")
    parser.add_argument("-t", "--time-period", dest="timeperiod", required=False, default="6",
                        help="Select time period in months to take from metadata")
    args = parser.parse_args()

    # build the time series
    countbydate, lineagecommon, region_list = buildseries(args.metadata, args.regions, args.adm, args.lineagetype, args.timeperiod, args.output)

    # build the model
    buildmodel(countbydate, lineagecommon, region_list, args.output)

if __name__ == '__main__':
    main()