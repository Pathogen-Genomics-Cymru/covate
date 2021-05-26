#!/usr/bin/env python3

import argparse
from build_time_series import buildseries

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-csv", dest="metadata", required=True,
                        help="Input metadata csv, expects columns: central_sample_id, adm1/adm2, sample_date, lineage/uk_lineage")
    parser.add_argument("-r", "--region-list", dest="regions", required=True,
                        help="Input list of regions to compare, e.g. UK-WLS, UK-ENG")
    parser.add_argument("-a", "--adm", dest="adm", required=True,
                        help="Select either adm1 or adm2")
    parser.add_argument("-l", "--lineage-type", dest="lineagetype", required=True,
                        help="Select either lineage or uk_lineage")
    
    args = parser.parse_args()
    metadata = args.metadata
    regions = args.regions
    adm = args.adm
    lineagetype = args.lineagetype

    buildseries(metadata, regions, adm, lineagetype)

if __name__ == '__main__':
    main()
