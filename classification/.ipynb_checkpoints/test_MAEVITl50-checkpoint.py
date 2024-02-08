from argparse import ArgumentParser
from math import ceil
from pathlib import Path
import json
import logging
import yaml
import torch
import torchvision
import pytorch_lightning as pl
import pandas as pd

from classification import utils_global,trainwMAEprel50
from classification.dataset import MsgPackIterableDatasetMultiTargetWithDynLabels,TestsetIterableDataset



def parse_args():
    args = ArgumentParser()
    args.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("/work3/s212495/data/models/trwMAEpre/240203-2237/last50ckpts/epoch=18-val_loss_epoch=4974.74.ckpt"),
        help="Checkpoint to already trained model (*.ckpt)",
    )
    args.add_argument("-c", "--config", type=Path, default=Path("config/ViTconfig.yml"))
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]

    # init 
    print("Load model from checkpoint", args.checkpoint)
    checkpoint_dir = args.checkpoint
    model = trainwMAEprel50.trainwMAEpretrain.load_from_checkpoint(checkpoint_dir)
    with open(model_params["after_test_label_mapping"], "r") as f:
        target_mapping = json.load(f)

    tfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
        
    dataset = TestsetIterableDataset(
        path=model_params["test_meta_path"],
        target_mapping=target_mapping,
        shuffle=False,
        transformation=tfm,
    )

    dataloader1 = torch.utils.data.DataLoader(
        dataset,
        batch_size=model_params["batch_size"],
        num_workers=model_params["num_workers_per_loader"],
        pin_memory=True,
    )
    print("-------------testdataloader lenth---------------")
    print(len(dataloader1))
    print("-------------------------------------------------")
        
        
        
    with open(model_params["after_test1_label_mapping"], "r") as f:
        target_mapping = json.load(f)

    tfm = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
            ),
        ]
    )
    dataset = TestsetIterableDataset(
        path=model_params["test1_meta_path"],
        target_mapping=target_mapping,
        shuffle=False,
        transformation=tfm,
    )

    dataloader2 = torch.utils.data.DataLoader(
        dataset,
        batch_size=model_params["batch_size"],
        num_workers=model_params["num_workers_per_loader"],
        pin_memory=True,
    )
    print("-------------testdataloader lenth---------------")
    print(len(dataloader2))
    print("-------------------------------------------------")

    trainer = pl.Trainer()
    trainer.test(model, dataloaders=dataloader1)

if __name__ == "__main__":
    main()


