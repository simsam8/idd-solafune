# Identifying Deforestation Drivers

Link to competition: https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=

## Running training and validation

To train a model run the following command:

`python train.py --epochs 3 --config unet`

available configs:
- unet
- deeplab
- segformer

Model checkpoints are stored in `./lightning_logs/version_n` by default. 


To run validation and create a submission:

`python eval.py ./lightning_logs/version_0/checkpoints/epoch=1-step=48.ckpt`

Change the path to the model you want to use.


## Setup environment

With cuda:

`uv sync --extra cu124`

With cpu:

`uv sync --extra cpu`


# Credits

- Code used for visualization: [here](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=discussion&tab=&topicId=d689d4a8-a939-4f0e-87bb-273707e8263f&page=1)
- Code to generate segmentation masks from train_annotations.json: [here](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=discussion&tab=&topicId=efe1aec6-0050-4214-ae77-9e17f56cddfd&page=1)
- Notebook for basline training pipeline: [here](https://github.com/motokimura/solafune_deforestation_baseline)
