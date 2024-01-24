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




def main():

    for dataset_type in ["test"]:
        with open(config[f"{dataset_type}_label_mapping"]) as f:
            mapping = json.load(f)

        logging.info(f"Expected dataset size: {len(mapping)}")
        image_path = config[f"{dataset_type}_meta_path"]
        files = os.listdir(image_path)
        filtered_mapping = {}
        # 筛选照片文件
        photo_extensions = ['.jpg', '.jpeg', '.png']  # 可能的照片文件扩展名

        photo_files = [file for file in files if os.path.splitext(file)[1].lower() in photo_extensions]

        # 输出照片文件的文件名
        for photo_file in photo_files:
            if photo_file in mapping:
                filtered_mapping[photo_file] = mapping[photo_file]
            
        
        
       
    
        logging.info(f"True dataset size: {len(filtered_mapping)}")

        with open(config[f"after_{dataset_type}_label_mapping"], "w") as fw:
            json.dump(filtered_mapping, fw)
    return


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="config/newbaseM.yml")
    args = parser.parse_args()
    return args


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
