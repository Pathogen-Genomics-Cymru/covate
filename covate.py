import argparse
from build_time_series import buildseries

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-csv", dest="metadata", required=True,
                        help="Input metadata csv, expects columns central_sample_id, adm1, sample_data, collection_date")

    
    args = parser.parse_args()
    metadata = args.metadata

    buildseries(metadata)

if __name__ == '__main__':
    main()
