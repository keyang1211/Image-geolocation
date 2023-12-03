from classification import train_resnet_nonlinear
from argparse import Namespace, ArgumentParser
from datetime import datetime
import json
import logging
from pathlib import Path

import yaml
import torch
import torchvision
import pytorch_lightning as pl
import pandas as pd

from classification import utils_global

def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("config/newbasenonlinear.yml"))
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, filename="/work3/s212495/trainresnonlinear.log")
    logger = pl.loggers.CSVLogger(save_dir="/work3/s212495/resnet_nonlinear", name="resnetlog")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    utils_global.check_is_valid_torchvision_architecture(model_params["arch"])

    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init 
    model = train_resnet_nonlinear.resnetregressor(modelparams=Namespace(**model_params))

    checkpoint_dir = out_dir / "ckpts" 
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                filename='{epoch}-{the_val_loss:.2f}',
                                                save_top_k = 10,
                                                save_last = True,
                                                monitor = 'the_val_loss', 
                                                mode = 'min')

    progress_bar_refresh_rate = False
    if args.progbar:
        progress_bar_refresh_rate = True

    trainer = pl.Trainer(
        **trainer_params,
        logger=logger,
        accelerator="gpu",
        devices=-1,
        val_check_interval=model_params["val_check_interval"], 
        callbacks=[checkpointer],
        enable_progress_bar=progress_bar_refresh_rate,
    )

    trainer.fit(model,ckpt_path="last")


if __name__ == "__main__":
    main()
