# Identifying Deforestation Drivers

## Description

The goal of this competition is to identify, segment, and classify deforestation drives 
from sattelite images. For the competition the following four classes of deforestation are 
as followed; grassland/shrubland, logging, mining, and plantation.

The dataset consists of sattelite images from the Sentinel-2 dataset, 
where each image contains 12 bands.

Link to competition: [here](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=)

Our report can be found [here](./report/report.pdf).

Made by: [Simon Vedaa](https://github.com/simsam8), [Khalil Ibrahim](https://github.com/KhalilIbrahimm), and [Safiya Mahamood](https://github.com/SafiyaMah)

## Setup environment

Make sure [uv](https://github.com/astral-sh/uv) is installed.

Run: `uv sync`, to install all dependencies.

## Running training and validation

### Training

To train a model run the following command:

`python train.py --epochs 3 --config unet`

available configs:
- unet
- deeplab
- segformer
- transunet
- vit_seg

Model checkpoints are stored in `./data/training_result/config_name` by default. 

Both `transunet` and `vit_seg` configs can be run with additional parameters to 
freeze the encoders during training.

`python train --epochs 3 --config transunet --frozen_start --f_epochs 1`

Use `frozen_start` to freeze the encoder, and `f_epochs` determines for 
how many epochs to freeze.

### Model Selection/Validation

To run model selection on validation data and create a submission:

`python eval.py` 

The submission is stored in `data/submission.json`


## File structure

- [`Models.py`](./Models.py): Pytorch Lightning module containing training, validation and testing loop/logic.
- [`train.py`](./train.py): Script for training models.
- [`eval.py`](./eval.py): Script for model selection and creating competition submission.
- [`utils.py`](./utils.py): Contains various utility functions.
- [`ensemble.py`](./ensemble.py): Module for creating ensemble models.
- [`datasets.py`](./datasets.py): Module for defining dataset and Datamodule used in `Models.py`
- [`configs.py`](./configs.py): Contains configurations for the models.
- [`generate_masks.py`](./generate_masks.py): Script for generating segmentation masks/labels from competition data.
- [`visualization.ipynb`](./visualization.ipynb): Notebook for creating visualization of predictions.
- [`graphs.ipynb`](./graphs.ipynb): Notebook for plotting training and validation metrics.
- [`report/`](./report/README.md): Contains the final report and all files and scripts to generate it.
- [`trans_unet/`](./trans_unet/README.md): Contains all code and documentation for our implementation of TransUNet.
- [`vision_transformer/`](./vision_transformer/README.md): Contains all code and documentation for our implementation of Vision Transformer.
- [`segformer/`](./segformer/README.md): Documentation of our usage of Segformer.

# Credits

- We used motokimuras baseline as a starting point for our solution: [here](https://github.com/motokimura/solafune_deforestation_baseline)
