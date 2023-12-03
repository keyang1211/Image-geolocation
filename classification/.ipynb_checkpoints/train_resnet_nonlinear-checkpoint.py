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
from classification.dataset import MsgPackIterableDatasetMultiTargetWithDynLabels,TestsetIterableDataset


class resnetregressor(pl.LightningModule):
    def __init__(self, modelparams: Namespace):
        super().__init__()
        self.save_hyperparameters()
        self.model, self.regressor = self.__build_model()
        self.validation_step_outputs = []
        self.test_outputs = []
        self.training_step_outputs = []

 

    def __build_model(self):
        logging.info("Build model")
        model, nfeatures = utils_global.build_base_model(self.hparams.modelparams.arch)

        regressor = torch.nn.Sequential(
            torch.nn.Linear(nfeatures, 128),  
            torch.nn.ReLU(),  
            torch.nn.Linear(128, 2),
            torch.nn.Tanh()# 输出两个数字（-1 - 1）
        )

        return model, regressor

    def forward(self, x):
        fv = self.model(x)
        yhats = self.regressor(fv)
        return yhats
    
 
        

    def training_step(self, batch, batch_idx):
        images, target = batch
        
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output = self(images)  #形状为 (batch_size, 2) 
        
        
       # 缩放和映射
        output_scaled = torch.stack([
            output[:, 0] * 90.0,   # 映射到 -90 到 +90 范围
            output[:, 1] * 180.0   # 映射到 -180 到 +180 范围
        ], dim=1)

        # 检测 output 中是否存在 NaN 值
        has_nan_output = torch.isnan(output).any().item()
        if has_nan_output:
            print("There is nan in trainoutput")
        # 检测 output_scaled 中是否存在 NaN 值
        has_nan_output_scaled = torch.isnan(output_scaled).any().item()
        if has_nan_output_scaled:
            print("There is nan in trainoutput_scaled")
        
        losses = [
            utils_global.vectorized_gc_distance(output_scaled[i][0],output_scaled[i][1], target[0][i],target[1][i])
            for i in range(output_scaled.shape[0])
        ]
        
        # 检查nan值
        has_nan = any([torch.isnan(loss1).any() for loss1 in losses])
        if has_nan:
            print("There is NaN in list losses")
       
        loss = sum(losses)
        
        has_nan = torch.isnan(loss)
        if has_nan:
            print("There is NaN in total loss")
            
            
        errors = [oneloss.item() for oneloss in losses]
        thissize = output_scaled.shape[0]
        output = {
            "loss" : loss,
            "size" : thissize,
            "losses" : errors}
        self.log("train_loss_batch", loss)
        self.training_step_outputs.append(output)
        return output

#     def on_train_batch_end(self,outputs, batch, batch_idx):
#         if batch_idx % 3999 == 0:
#             print("----------------train_batch_end_loss_every4000---------------")
#             print(outputs["losses"])
#             print("---------------------------------------------------")
    

    def on_train_epoch_end(self):
        
        total_loss = sum([x["loss"].item() for x in self.training_step_outputs])
        print(f"Epoch {self.current_epoch}: Total Loss: {total_loss}")
        total_sample = sum([x["size"] for x in self.training_step_outputs])
        print(f"Epoch {self.current_epoch}: Total Sample: {total_sample}")
        epoch_mean = total_loss / total_sample
        print(f"Epoch {self.current_epoch}: Epoch Mean: {epoch_mean}")
        self.log("training_epoch_mean", epoch_mean)
        # free up the memory
        self.training_step_outputs.clear()

    
    def validation_step(self, batch, batch_idx):
        images, target = batch #iamge是（batch size，2），target是两个张量的列表，一个是lat，一个是lon
        
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward
        output = self(images)
        # 缩放和映射
        output_scaled = torch.stack([
            output[:, 0] * 90.0,   # 映射到 -90 到 +90 范围
            output[:, 1] * 180.0   # 映射到 -180 到 +180 范围
        ], dim=1)
        
        
        
        
        
        has_nan_output = torch.isnan(output).any().item()
        if has_nan_output:
            print("There is nan in valoutput")
        # 检测 output_scaled 中是否存在 NaN 值
        has_nan_output_scaled = torch.isnan(output_scaled).any().item()
        if has_nan_output_scaled:
            print("There is nan in valoutput_scaled")

        # loss calculation
        losses = [
            utils_global.vectorized_gc_distance(output_scaled[i][0],output_scaled[i][1], target[0][i],target[1][i])
            for i in range(output.shape[0])
        ]
       
        loss = sum(losses)

        thissize = output.shape[0]
        # 计算误差统计
        errors = [loss.item() for loss in losses]
    # 统计不同误差范围内的样本数量
        num_samples = len(errors)
        error_100 = sum([1 for error in errors if error <= 100])
        error_500 = sum([1 for error in errors if  error <= 500])
        error_1000 = sum([1 for error in errors if  error <= 1000])
        error_2000 = sum([1 for error in errors if  error <= 2000])

    # 输出统计信息
#         logging.info("NumSamples", num_samples)
#         logging.info("Error100", error_100)
#         logging.info("Error500", error_500)
#         logging.info("Error1000", error_1000)
#         logging.info("Error2000", error_2000)
       
     
       

        output = {
            "val_loss": loss,
            "errors" : errors,
            "avg_1loss":loss/thissize, 
            "size" : thissize,
            "ACC100" : error_100,
            "ACC500" : error_500,
            "ACC1000" : error_1000,
            "ACC2000" : error_2000
        }
        self.log("val_loss", loss)
        # print("-------------valoutput----------")
        # print(output)
        # print("------------------------------------------------------")
        self.validation_step_outputs.append(output)
        return output
    
    
    
    
    
#     def on_validation_batch_end(self,outputs, batch, batch_idx): 
#         if batch_idx % 100 == 0:
#             print("----------------val_batch_end_loss---------------")
#             print(outputs["errors"])
#             print("---------------------------------------------------")
    
        
            
           
        
        
        
        
        
    
    def on_validation_epoch_end(self):
        epoch_num = self.current_epoch
        logging.info(f"Starting epoch {epoch_num}")
        avg_loss = torch.tensor([x["avg_1loss"].item() for x in self.validation_step_outputs]).mean()

    
        total_samples = sum([x["size"] for x in self.validation_step_outputs])
        total_error_100 = sum([x["ACC100"] for x in self.validation_step_outputs])
        total_error_500 = sum([x["ACC500"] for x in self.validation_step_outputs])
        total_error_1000 = sum([x["ACC1000"] for x in self.validation_step_outputs])
        total_error_2000 = sum([x["ACC2000"] for x in self.validation_step_outputs])
    
        error_100_ratio = total_error_100 / total_samples
        error_500_ratio = total_error_500 / total_samples
        error_1000_ratio = total_error_1000 / total_samples
        error_2000_ratio = total_error_2000 / total_samples
    
        logging.info("the_val_loss: %s", avg_loss.item())
        logging.info("100_accratio: %s", error_100_ratio)
        logging.info("500_accratio: %s", error_500_ratio)
        logging.info("1000_accratio: %s", error_1000_ratio)
        logging.info("2000_accratio: %s", error_2000_ratio)
        self.log("the_val_loss", avg_loss)
        self.log("100_accratio", error_100_ratio)
        self.log("500_accratio", error_500_ratio)
        self.log("1000_accratio", error_1000_ratio)
        self.log("2000_accratio", error_2000_ratio)
        self.validation_step_outputs.clear()




    def test_step(self, batch, batch_idx, dataloader_idx=None):

        images, target = batch #iamge是（batch size，2），target是两个张量的列表，一个是lat，一个是lon
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        output = self(images)
        output_scaled = torch.stack([
            output[:, 0] * 90.0,   # 映射到 -90 到 +90 范围
            output[:, 1] * 180.0   # 映射到 -180 到 +180 范围
        ], dim=1)
        # 检测 output 中是否存在 NaN 值
        has_nan_output = torch.isnan(output).any().item()
        if has_nan_output:
            print("There is nan in output")
        # 检测 output_scaled 中是否存在 NaN 值
        has_nan_output_scaled = torch.isnan(output_scaled).any().item()
        if has_nan_output_scaled:
            print("There is nan in output_scaled")
        
        losses = [
            utils_global.vectorized_gc_distance(output_scaled[i][0],output_scaled[i][1], target[0][i],target[1][i])
            for i in range(output.shape[0])
        ]
        loss = sum(losses)
        errors = [loss.item() for loss in losses]
        
        num_samples = len(errors)
        error_100 = sum([1 for error in errors if error <= 100])
        error_500 = sum([1 for error in errors if  error <= 500])
        error_1000 = sum([1 for error in errors if  error <= 1000])
        error_2000 = sum([1 for error in errors if  error <= 2000])

        output = {
            "batch_loss" : loss,
            "losses" : errors,
            "number" : num_samples,
            "num in 100km" : error_100,
            "num in 500km" : error_500,
            "num in 1000km" : error_1000,
            "num in 2000km" : error_2000
        }
        self.test_outputs.append(output)
        

    def on_test_end(self):
        total_loss = sum([x["batch_loss"].item() for x in self.test_outputs])
        total_num = sum([x["number"] for x in self.test_outputs])
        avg_loss = total_loss/total_num
        num_100 = sum([x["num in 100km"] for x in self.test_outputs])
        num_500 = sum([x["num in 500km"] for x in self.test_outputs])
        num_1000 = sum([x["num in 1000km"] for x in self.test_outputs])
        num_2000 = sum([x["num in 2000km"] for x in self.test_outputs])
        acc_100 = num_100/total_num
        acc_500 = num_500/total_num
        acc_1000 = num_1000/total_num
        acc_2000 = num_2000/total_num
        
        self.test_outputs.clear()
        
        with open('/work3/s212495/test_results.txt', 'w') as file:
            file.write(f"Avg Loss: {avg_loss}\n")
            file.write(f"Accuracy in 100km: {acc_100}\n")
            file.write(f"Accuracy in 500km: {acc_500}\n")
            file.write(f"Accuracy in 1000km: {acc_1000}\n")
            file.write(f"Accuracy in 2000km: {acc_2000}\n")
        
        
        
        
    def configure_optimizers(self):

        optim_feature_extrator = torch.optim.SGD(
            self.parameters(), **self.hparams.modelparams.optim["params"]
        )
        Ascheduler = torch.optim.lr_scheduler.MultiStepLR(
            optim_feature_extrator, **self.hparams.modelparams.scheduler["params"]
        )
        
        return [optim_feature_extrator],[Ascheduler]
        # return {
        #     "optimizer": optim_feature_extrator,
        #     "lr_scheduler": {
        #         "scheduler": torch.optim.lr_scheduler.MultiStepLR(
        #             optim_feature_extrator, **self.hparams.modelparams.scheduler["params"]
        #         ),
        #         "interval": "epoch",
        #         "name": "lr"
        #     },
        # }

    def train_dataloader(self):

        with open(self.hparams.modelparams.after_train_label_mapping, "r") as f:
            target_mapping = json.load(f)

        tfm = torchvision.transforms.Compose(
            [
                
                torchvision.transforms.Resize(256),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.AutoAugment(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.modelparams.msgpack_train_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.modelparams.key_img_id,
            key_img_encoded=self.hparams.modelparams.key_img_encoded,
            shuffle=True,
            transformation=tfm,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.modelparams.batch_size,
            num_workers=self.hparams.modelparams.num_workers_per_loader,
            pin_memory=True,
        )
        print("-------------traindataloader lenth---------------")
        print(len(dataloader))
        print("-------------------------------------------------")
        return dataloader

    def val_dataloader(self):

        with open(self.hparams.modelparams.after_val_label_mapping, "r") as f:
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
        dataset = MsgPackIterableDatasetMultiTargetWithDynLabels(
            path=self.hparams.modelparams.msgpack_val_dir,
            target_mapping=target_mapping,
            key_img_id=self.hparams.modelparams.key_img_id,
            key_img_encoded=self.hparams.modelparams.key_img_encoded,
            shuffle=False,
            transformation=tfm,
            cache_size=1024,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.modelparams.batch_size,
            num_workers=self.hparams.modelparams.num_workers_per_loader,
            pin_memory=True,
        )
        print("-------------valdataloader lenth---------------")
        print(len(dataloader))
        print("-------------------------------------------------")

        return dataloader
    
    
    
    def test_dataloader(self):
        with open(self.hparams.modelparams.after_test_label_mapping, "r") as f:
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
            path=self.hparams.modelparams.msgpack_test_dir,
            target_mapping=target_mapping,
            shuffle=False,
            transformation=tfm,
        )

        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.modelparams.batch_size,
            num_workers=self.hparams.modelparams.num_workers_per_loader,
            pin_memory=True,
        )
        print("-------------testdataloader lenth---------------")
        print(len(dataloader))
        print("-------------------------------------------------")

        return dataloader
    
    
    
        


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
    model = resnetregressor(modelparams=Namespace(**model_params))

    checkpoint_dir = out_dir / "ckpts" 
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                filename='{epoch}-{the_val_loss:.2f}',
                                                save_top_k = 5,
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

    trainer.fit(model)


if __name__ == "__main__":
    main()
