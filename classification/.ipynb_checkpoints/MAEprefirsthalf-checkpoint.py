from argparse import Namespace, ArgumentParser
from datetime import datetime
import json
import logging
from pathlib import Path

import yaml
import torch
import torch.nn as nn

import torchvision
import pytorch_lightning as pl
import pandas as pd
from classification import utils_global,ViTencoder
from classification.dataset import MsgPackIterableDatasetfirst50,TestsetIterableDataset,MsgPackIterableDatasetMultiTargetWithDynLabels



# encoder（随机排序，遮蔽之后的数据输入），_process_input之后的数据，在encoder之前，数据应该整理成(n, (n_h * n_w), hidden_dim) (batchsize,块数，投影的嵌入维度)
# decoder（恢复排序之前的数据输出）输入也是(n, (n_h * n_w), hidden_dim) (batchsize,块数，投影的嵌入维度)的维度，然后恢复排序





class MAEpretrainencoder(nn.Module):
        
    def __init__(self,image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.1,
            mask_ratio=0.75):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.mask_ratio = mask_ratio
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size)
        seq_length_encoder =  ((image_size // patch_size) ** 2) + 1
        self.pos_emb_encoder = nn.Parameter(torch.empty(1, seq_length_encoder, hidden_dim).normal_(std=0.02))
        self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.inencoder = ViTencoder.MAEencoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout)
        
        
    
        
    def patchify(self,img):
        
        
        # img: (N, 3, H, W)
        # x: (N, L, embed块)
        # 先用卷积映射加分块，再加pos_emb
        
        x = self.conv_proj(img) #(N,emb,H块数,W块数)
        n = x.shape[0]
        n_h = x.shape[2]
        n_w = x.shape[3]
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)
        
        x = x.permute(0, 2, 1)
        
        return x
    def random_mask(self,mask_ratio,x):
        #  x: (N, L, embed块)
        # 输出x:(N,newL,embed块)
        if mask_ratio == 0.0:
            return x,0,0
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore
        



    def forward(self, x: torch.Tensor):
        #图片tensor==>切块，pos_emb，打乱排序，删除。==> (n, (n_h * n_w), hidden_dim) (batchsize,块数，投影的嵌入维度)
        origin_x = self.patchify(x)
        #加pos_emb
        x = origin_x + self.pos_emb_encoder[:, 1:, :]
        
        #random mask
        x,mask,ids_restore=self.random_mask(self.mask_ratio,x)
        
        #加clstoken
        n = x.shape[0]
        # Expand the class token to the full batch
        class_token = self.class_token + self.pos_emb_encoder[:,:1,:]
        batch_class_token = class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        #再encode
        x=self.inencoder(x)
        
        return x,mask,ids_restore
    


class MAEpretrain(pl.LightningModule):
    def __init__(self,modelparams: Namespace,image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout=0.1,
            deco_emb_dim=384,
            mask_ratio=0.75):
        super().__init__()
        self.save_hyperparameters()
        self.validation_step_outputs = []
        self.test_outputs = []
        self.training_step_outputs = []
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout
        self.deco_emb_dim = deco_emb_dim
        self.mask_ratio = mask_ratio
        self.decoder_emb = nn.Linear(hidden_dim, deco_emb_dim, bias=True)
        self.mask_num_block = int(((image_size // patch_size) ** 2) * mask_ratio)
        self.mask_token = nn.Parameter(torch.zeros(1, self.mask_num_block, deco_emb_dim))  #decoder之前加的空白块
        seq_length_encoder =  ((image_size // patch_size) ** 2) + 1
        self.pos_emb_decoder = nn.Parameter(torch.empty(1, seq_length_encoder, deco_emb_dim).normal_(std=0.02))
        self.deconv_layer = nn.ConvTranspose2d(deco_emb_dim, 3, kernel_size=patch_size, stride=patch_size, bias=False)
        self.encoder, self.decoder = self.__build_model()
        
        
        
        
    def __build_model(self):
        encoder = self.__build_encoder()
        decoder = self.__build_decoder()
        return encoder, decoder
    
    def __build_encoder(self):
        logging.info("Build model")
        new_model = MAEpretrainencoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout
        )
        return new_model
    
    def __build_decoder(self):
        new_model = ViTencoder.MAEdecoder(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_layers=self.num_layers//3,
            num_heads=self.num_heads,
            hidden_dim=self.deco_emb_dim,
            mlp_dim=self.mlp_dim//2,
            dropout=self.dropout
        )
        return new_model
    
    
    
        
    def process_encoder_output(self,x,ids_restore):
        #x(batch,enco块数，encembdim)
        x = self.decoder_emb(x)
        n = x.shape[0]
        mask_token1 = self.mask_token.expand(n,-1,-1)
        x = torch.cat((x, mask_token1), dim=1)
        cls_token1 = x[:,:1,:]
        x_noclstoken = x[:,1:,:]
        x_noclstoken = torch.gather(x_noclstoken, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat((cls_token1,x_noclstoken),dim=1)
        return x
        
        
        
    def unpatchify(self,x):
        # x:decoder的输出, x: (N, L, embed) 先变成N，EMBed，n_h，n_w
        # img:imgs: (N, 3, H, W)
        # 反卷积还原img
        n_h = self.image_size // self.patch_size
        n_w = self.image_size // self.patch_size
        N, L, embed_dim = x.size()

        # 将 x 变形为 (N, embed_dim, n_h, n_w)
        x_reshaped = x.reshape(N, embed_dim, n_h, n_w)
        img = self.deconv_layer(x_reshaped)

    
    
        return img
    
    def patchify_img(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    
    
    def forward(self, x: torch.Tensor):
        x,mask,ids_restore = self.encoder(x) #ids_restore不包含clstoken
    
        #encoder的输出再进行emb，然后添加空白的块，重排序
        x = self.process_encoder_output(x,ids_restore)
        
        #再加pos_emb
        x = x + self.pos_emb_decoder
        
        #decode
        x = self.decoder(x)
        x = x[:,1:,:]
        #将输出变回图片
        img = self.unpatchify(x) 
        
        #到底要不要反卷积还原成img？
        return img, mask
    
    def training_step(self,batch,batch_idx):
        
        images, target = batch
        
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output,mask = self(images)  #output形状为 (batch_size,3,H,W ) mask（batch_size, L）
        ori_img = self.patchify_img(images)
        pred_img = self.patchify_img(output)  #全变成(N, L, patch_size**2 *3)
        mask = mask.unsqueeze(2).expand(-1, -1, self.patch_size**2 * 3)
        # 只剩下被遮蔽部分
        ori_img = ori_img * mask
        pred_img = pred_img * mask
        
        loss = (pred_img - ori_img) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = torch.sum(loss)
        self.log("train_loss_batch", loss)
        self.training_step_outputs.append(loss)
        return loss
    
    
    def on_train_epoch_end(self):
        epoch_loss = sum(self.training_step_outputs)
        self.log("train_loss_epoch",epoch_loss)
        self.training_step_outputs.clear()
        
        
        
        
    def validation_step(self, batch, batch_idx):
        images, target = batch #iamge是（batch size，2），target是两个张量的列表，一个是lat，一个是lon
        
        if not isinstance(target, list) and len(target.shape) == 1:
            target = [target]

        # forward pass
        output,mask = self(images)  #output形状为 (batch_size,3,H,W ) mask（batch_size, L）
        ori_img = self.patchify_img(images)
        pred_img = self.patchify_img(output)  #全变成(N, L, patch_size**2 *3)
        mask = mask.unsqueeze(2).expand(-1, -1, self.patch_size**2 * 3)
        # 只剩下被遮蔽部分
        ori_img = ori_img * mask
        pred_img = pred_img * mask
        
        loss = (pred_img - ori_img) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = torch.sum(loss)
        self.log("val_loss_batch", loss)
        self.validation_step_outputs.append(loss)
        return loss
    
    
    def on_validation_epoch_end(self):
        epoch_num = self.current_epoch
        logging.info(f"Starting epoch {epoch_num}")
        epoch_loss = sum(self.validation_step_outputs)
        self.log("val_loss_epoch",epoch_loss)
        self.validation_step_outputs.clear()

            
        
    def configure_optimizers(self):

        optim_feature_extrator =torch.optim.AdamW(
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
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.RandomResizedCrop(224, scale=(0.66, 1.0)),
                # torchvision.transforms.RandAugment(),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ),
            ]
        )

        dataset = MsgPackIterableDatasetfirst50(
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
    
    
    
    
    
    
    
def parse_args():
    args = ArgumentParser()
    args.add_argument("-c", "--config", type=Path, default=Path("config/MAEpreconfig.yml"))
    args.add_argument("--progbar", action="store_true")
    return args.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, filename="/work3/s212495/trainMAE.log")
    logger = pl.loggers.CSVLogger(save_dir="/work3/s212495/MAElog", name="MAElogf50")
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_params = config["model_params"]
    trainer_params = config["trainer_params"]


    out_dir = Path(config["out_dir"]) / datetime.now().strftime("%y%m%d-%H%M")
    out_dir.mkdir(exist_ok=True, parents=True)
    logging.info(f"Output directory: {out_dir}")

    # init 
    model = MAEpretrain(modelparams=Namespace(**model_params))

    checkpoint_dir = out_dir / "f50ckpts" 
    checkpointer = pl.callbacks.ModelCheckpoint(dirpath=checkpoint_dir,
                                                filename='{epoch}-{val_loss_epoch:.2f}',
                                                save_top_k = 3,
                                                save_last = True,
                                                monitor = 'val_loss_epoch', 
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

    