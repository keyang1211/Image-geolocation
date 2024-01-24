from argparse import ArgumentParser
import logging
from pathlib import Path
import json
import pandas as pd

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="resources/im2gps3k_places365.csv")
    parser.add_argument("-p", "--path", type=Path, default="resources/im2gps3k_places365_mapping_h3.json")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    datapath = args.config
    df = pd.read_csv(datapath)
        
    result_dict = dict()
    for _, row in df.iterrows():
        img_id = row["IMG_ID"]
        lat_lon = [row["LAT"], row["LON"]]
        result_dict[img_id] = lat_lon
    with open(args.path, "w") as json_file:
        json.dump(result_dict, json_file)

if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=logging.INFO,
    )

    main()
