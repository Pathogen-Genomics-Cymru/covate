#!/usr/bin/env python3


import argparse
from .build_time_series import buildseries
from .build_model import buildmodel
from .cross_correlation import crosscorrelation


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input-csv", dest="metadata", required=True,
                        help="""Input metadata csv, expects columns: cog_id,
                        adm1/adm2, sample_date, lineage/uk_lineage""")

    parser.add_argument("-o", "--output-dir", dest="output", required=True,
                        help="Output directory for the results")

    parser.add_argument("-r", "--region-list", dest="regions", required=False,
                        default="Wales, England",
                        help="""Input list of regions to compare,
                        e.g. Wales, England""")

    parser.add_argument("-a", "--adm", dest="adm", required=False,
                        default="adm1",
                        help="Select either adm1 or adm2")

    parser.add_argument("-l", "--lineage-type", dest="lineagetype",
                        required=False, default="uk_lineage",
                        help="Select either lineage or uk_lineage")

    parser.add_argument("-t", "--time-period", dest="timeperiod",
                        required=False, default="12", type=int,
                        help="""Select time period in weeks to take
                        from metadata""")

    parser.add_argument("-e", "--end-date", dest="enddate", required=False,
                        default="",
                        help="""Select end date to take from metadata.
                        Format: d/m/Y""")

    parser.add_argument("-p", "--predict", dest="predict", required=False,
                        action="store_true",
                        help="Run prediction forecast")

    parser.add_argument("-v", "--validate", dest="validate", required=False,
                        action="store_true",
                        help="Run validation forecast")

    parser.add_argument("-c", "--cross-correlation", dest="crosscorr",
                        required=False, action="store_true",
                        help="Run cross-correlation analysis")

    parser.add_argument("-f", "--primary-region", dest="primaryregion",
                        required=False, default="Wales",
                        help="""Region of primary interest for
                        cross-correlation""")

    parser.add_argument("-m", "--max-lags", dest="maxlags", required=False,
                        default="14", type=int,
                        help="Maximum number of lags to investigate")

    parser.add_argument("-n", "--n-steps", dest="nsteps", required=False,
                        default="14", type=int,
                        help="Number of days to predict")

    args = parser.parse_args()

    # check args
    # raise error if more than two regions for cross-correlation
    if args.crosscorr:

        regionlist = [str(region) for region in args.regions.split(', ')]

        if len(regionlist) > 2:

            raise ValueError('''The cross-correlation analysis is currently only
                             supported for two regions.''')

    # build the time series
    countbydate, lineagecommon, region_list, enddate, toplineagelist = (
        buildseries(args.metadata, args.regions, args.adm, args.lineagetype,
                    args.timeperiod, args.enddate, args.output, args.nsteps,
                    False, args.crosscorr, args.primaryregion))

    # run cross-correlation analysis
    if args.crosscorr:

        crosscorrelation(countbydate, lineagecommon, region_list, enddate,
                         args.output, args.primaryregion, toplineagelist)

    # build the model
    if args.predict:

        buildmodel(countbydate, lineagecommon, region_list, enddate,
                   args.output, args.maxlags, args.nsteps, False)

    # if validation forecast selected, run again
    if args.validate:

        countbydate, lineagecommon, region_list, enddate, toplineagelist = (
            buildseries(args.metadata, args.regions, args.adm,
                        args.lineagetype, args.timeperiod, args.enddate,
                        args.output, args.nsteps, args.validate,
                        False, args.primaryregion))

        buildmodel(countbydate, lineagecommon, region_list, enddate,
                   args.output, args.maxlags, args.nsteps, args.validate)


if __name__ == '__main__':
    main()
