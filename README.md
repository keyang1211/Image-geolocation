
# Geolocation Estimation of Photos   




## Reproduce Results

### Test on Already Trained Model
The (list of) image files for testing can be found on the following links:
* Im2GPS: http://graphics.cs.cmu.edu/projects/im2gps/ (can be downloaded automatically)
* Im2GPS3k: https://github.com/lugiavn/revisiting-im2gps/

Download and extract the two testsets (Im2GPS, Im2GPS3k) in `resources/images/<dataset_name>` and run the evaluation script with the provided meta data, i.e., the ground-truth coordinate for each image.
When using the default paramters, make sure that the pre-trained model is available. 
```bash
# download im2gps testset
mkdir resources/images/im2gps
wget http://graphics.cs.cmu.edu/projects/im2gps/gps_query_imgs.zip -O resources/images/im2gps.zip
unzip resources/images/im2gps.zip -d resources/images/im2gps/

wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps_places365.csv -O resources/images/im2gps_places365.csv
wget https://raw.githubusercontent.com/TIBHannover/GeoEstimation/original_tf/meta/im2gps3k_places365.csv -O resources/images/im2gps3k_places365.csv
python -m classification.test
```

Available argparse paramters:
```
--checkpoint CHECKPOINT
    Checkpoint to already trained model (*.ckpt)
--hparams HPARAMS     
    Path to hparams file (*.yaml) generated during training
--image_dirs IMAGE_DIRS [IMAGE_DIRS ...]
    Whitespace separated list of image folders to evaluate
--meta_files META_FILES [META_FILES ...]
    Whitespace separated list of respective meta data (ground-truth GPS positions). Required columns: IMG_ID,LAT,LON
--gpu
    Use GPU for inference if CUDA is available, default to True
--precision PRECISION
    Full precision (32), half precision (16)
--batch_size BATCH_SIZE
--num_workers NUM_WORKERS
    Number of workers for image loading and pre-processing

```


### Training from Scratch
We provide a complete training script which is written in [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) and report all hyper-paramters used for the provided model. Furthermore, a script is given to download and pre-process the images that are used for training and validiation.

1) Download training and validation images
    - We provide a script to download the images given a list of URLs
    - Due to no longer publicly available images, the size of the dataset might be smaller than the original.
    - We also store the images in chunks using [MessagePack](https://msgpack.org/) to speed-up the training process (similar to multiple TFRecord files)
2) Training and hyper-paramters: All hyper-paramters can be configured in `configs/*.yml` as well as paramters from PyTorch Lightning [`Trainer`](https://pytorch-lightning.readthedocs.io/en/latest/trainer.html#trainer-class-api) class.


Necessary steps:
```bash
# download and preprocess images
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_urls.csv -O resources/mp16_urls.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_urls.csv -O resources/yfcc25600_urls.csv 
python download_images.py --output resources/images/mp16 --url_csv resources/mp16_urls.csv --shuffle
python download_images.py --output resources/images/yfcc25600 --url_csv resources/yfcc25600_urls.csv --shuffle --size_suffix ""

# assign cell(s) for each image using the original meta information
wget https://github.com/TIBHannover/GeoEstimation/releases/download/v1.0/mp16_places365.csv -O resources/mp16_places365.csv
wget https://github.com/TIBHannover/GeoEstimation/releases/download/pytorch/yfcc25600_places365.csv -O resources/yfcc25600_places365.csv
# remove images that were not downloaded 
python filter_by_downloaded_images.py

# train geo model from scratch
python -m classification.train_resnet_nonlinear 
```


## Requirements
All requirements are listed in the `environment.yml`. We recomment to use [*conda*](https://docs.conda.io/en/latest/) to install all required packages in an individual environment.
```bash

# install dependencies
conda env create -f environment.yml 
conda activate geoestimation-github-pytorch


```



