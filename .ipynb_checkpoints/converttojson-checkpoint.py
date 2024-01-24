from argparse import ArgumentParser
import logging
import os
import re
import json
from io import BytesIO
from pathlib import Path
from typing import Union

import yaml
import msgpack
import pandas as pd

#CSV转成json


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="config/newbaseM.yml")
    args = parser.parse_args()
    return args

def main():
    
    for dataset_type in ["train", "val"]:
        datapath = config[f"{dataset_type}_meta_path"]
        df = pd.read_csv(datapath)
        
        result_dict = dict()
        for _, row in df.iterrows():
            img_id = row["IMG_ID"]
            lat_lon = [row["LAT"], row["LON"]]
            result_dict[img_id] = lat_lon
        with open(config[f"{dataset_type}_label_mapping"], "w") as json_file:
            json.dump(result_dict, json_file)
    return

if __name__ == "__main__":

    logging.basicConfig(
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%d-%m-%Y %H:%M:%S",
        level=logging.INFO,
    )

    args = parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    config = config["model_params"]

    main()
