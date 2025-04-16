import os
from argparse import ArgumentParser
from pathlib import Path

import albumentations as albu
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

import configs
from datasets import IDDDataModule
from Models import Model
import torch

seed_everything(367)

augmentations = albu.Compose(
    [
        # shift, scale, and rotate
        albu.ShiftScaleRotate(
            p=0.5,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,  # constant border
            fill=0,
            fill_mask=0,
            interpolation=2,  # bicubic
        ),
        # flip, transpose, and rotate90
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.RandomRotate90(p=0.5),
    ]
)

def pl_trainer(config, data_dir, epochs=10):
    model = Model(config)
    train_ouput_dir = data_dir / "training_result"
    trainer = pl.Trainer(
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(
                dirpath=train_ouput_dir,
                filename="best_f1",
                save_top_k=1,
                monitor="val/f1",
                mode="max",
                save_last=False,
            ),
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=[TensorBoardLogger(train_ouput_dir, name=None)],
        precision="16-mixed",
        deterministic=True,
        benchmark=False,
        sync_batchnorm=False,
        check_val_every_n_epoch=1, # To evaluate after every epoch to find the best epoch 
        default_root_dir=os.getcwd(),
        accelerator="auto",
        devices="auto",
        strategy="auto",
        log_every_n_steps=5,
        enable_progress_bar=True,
    )
    data_module = IDDDataModule(
        data_dir, augmentations, config["batch_size"], config["num_workers"]
    )
    trainer.fit(model, datamodule=data_module)

def main(args):
    torch.cuda.empty_cache()
    data_path = Path("data/").absolute()
    pl_trainer(getattr(configs, args.config), data_path, epochs=int(args.epochs))

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
