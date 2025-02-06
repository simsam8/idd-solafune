import math
import os
from pathlib import Path

import albumentations as albu
import lightning as pl
import segmentation_models_pytorch as smp
import sklearn
import torch

# import pytorch_lightning as pl
from filelock import FileLock

# from pytorch_lightning.loggers import TensorBoardLogger
from lightning.pytorch.loggers import TensorBoardLogger
from ray import train, tune
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from ray.tune.schedulers import ASHAScheduler
from timm.optim import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2
from torch.utils.data import DataLoader

from datasets import TrainValDataset

# from timm.scheduler import create_scheduler_v2


epochs = 1
class_names = ["grassland_shrubland", "logging", "mining", "plantation"]

augmentations = albu.Compose(
    [
        # shift, scale, and rotate
        albu.ShiftScaleRotate(
            p=0.5,
            shift_limit=0.0625,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,  # constant border
            value=0,
            mask_value=0,
            interpolation=2,  # bicubic
        ),
        # random crop
        albu.RandomCrop(
            p=1,
            width=512,
            height=512,
        ),
        # flip, transpose, and rotate90
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.Transpose(p=0.5),
        albu.RandomRotate90(p=0.5),
    ]
)


class Model(pl.LightningModule):
    def __init__(self, config, data_dir=None):
        super().__init__()

        self.data_dir = data_dir or os.getcwd()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

        # prepare segmentation model
        self.model = smp.create_model(
            arch="unet",
            encoder_name="tu-tf_efficientnetv2_s",  # use `tf_efficientnetv2_s` from timm
            encoder_weights="imagenet",  # always starts from imagenet pre-trained weight
            in_channels=12,
            classes=4,
        )

        # prepare loss functions
        self.dice_loss_fn = smp.losses.DiceLoss(
            mode=smp.losses.MULTILABEL_MODE, from_logits=True
        )
        self.bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        # assuming image is already normalized
        return self.model(image)  # logits

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        logits_mask = self.forward(image)

        loss = self.dice_loss_fn(logits_mask, mask) + self.bce_loss_fn(
            logits_mask, mask
        )

        # count tp, fp, fn, tn for each class to compute validation metrics at the end of epoch
        thresh = 0.5
        prob_mask = logits_mask.sigmoid()
        tp, fp, fn, tn = smp.metrics.get_stats(
            (prob_mask > thresh).long(),
            mask.long(),
            mode=smp.losses.MULTILABEL_MODE,
        )  # each of tp, fp, fn, tn is a tensor of shape (batch_size, num_classes) and of type long

        output = {
            "loss": loss.detach().cpu(),
            "tp": tp.detach().cpu(),
            "fp": fp.detach().cpu(),
            "fn": fn.detach().cpu(),
            "tn": tn.detach().cpu(),
        }
        if stage == "train":
            self.training_step_outputs.append(output)
        else:
            self.validation_step_outputs.append(output)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def shared_epoch_end(self, outputs, stage):
        def log(name, tensor, prog_bar=False):
            self.log(
                f"{stage}/{name}",
                tensor.to(self.device),
                sync_dist=True,
                prog_bar=prog_bar,
            )

        # aggregate loss
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        log("loss", loss, prog_bar=True)

        # aggregate tp, fp, fn, tn to compose F1 score for each class
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        f1_scores = {}
        for i, class_name in enumerate(class_names):
            f1_scores[class_name] = smp.metrics.f1_score(
                tp[:, i], fp[:, i], fn[:, i], tn[:, i], reduction="macro-imagewise"
            )
            log(f"f1/{class_name}", f1_scores[class_name], prog_bar=False)

        f1_avg = torch.stack([v for v in f1_scores.values()]).mean()
        log("f1", f1_avg, prog_bar=True)

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def prepare_data(self) -> None:
        sample_indices = list(range(176))  # train_0.tif to train_175.tif
        train_indices, val_indices = sklearn.model_selection.train_test_split(
            sample_indices, test_size=0.2, random_state=42
        )
        with FileLock("./.data.lock"):
            self.idd_train = TrainValDataset(
                self.data_dir,
                train_indices,
                augmentations=augmentations,
            )
            self.idd_val = TrainValDataset(
                self.data_dir, val_indices, None
            )

    def train_dataloader(self):
        return DataLoader(
            self.idd_train,
            batch_size=int(self.batch_size),
            num_workers=int(self.num_workers),
        )

    def val_dataloader(self):
        return DataLoader(
            self.idd_val,
            batch_size=int(self.batch_size),
            num_workers=int(self.num_workers),
        )

    def configure_optimizers(self):
        # optimizer
        # optimizer = create_optimizer_v2(
        #     self.parameters(),
        #     opt="adamw",
        #     lr=self.lr,
        #     weight_decay=self.weight_decay,
        #     filter_bias_and_bn=True,  # filter out bias and batchnorm from weight decay
        # )
        #
        # # lr scheduler
        # scheduler, _ = create_scheduler_v2(
        #     optimizer,
        #     sched="cosine",
        #     num_epochs=epochs,
        #     min_lr=0.0,
        #     warmup_lr=1e-5,
        #     warmup_epochs=0,
        #     warmup_prefix=False,
        #     step_on_epochs=True,
        # )
        #
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "interval": "epoch",
        #     },
        # }
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def lr_scheduler_step(self, scheduler, metric):
    #     # workaround for timm's scheduler:
    #     # https://github.com/Lightning-AI/lightning/issues/5555#issuecomment-1065894281
    #     scheduler.step(
    #         epoch=self.current_epoch
    #     )  # timm's scheduler need the epoch value


def train_idd(config):
    model = Model(config, Path("./data/"))
    trainer = pl.Trainer(max_epochs=2, enable_progress_bar=True)
    trainer.fit(model)


def train_idd_no_tune():
    config = {"batch_size": 4, "lr": 1e-4, "weight_decay": 1e-5, "num_workers": 10}
    train_idd(config)


def train_idd_tune(config, num_epochs, num_gpus=0, data_dir="./data"):
    data_dir = Path(data_dir)
    model = Model(config, data_dir)
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # gpu=math.ceil(num_gpus),
        devices="auto",
        accelerator="auto",
        logger=TensorBoardLogger(save_dir=os.getcwd(), name="", version="."),
        enable_progress_bar=False,
        callbacks=[TuneReportCallback({"loss": "loss", "f1": "f1"})],
    )

    trainer.fit(model)


def tune_idd_asha(num_samples=10, num_epochs=10, gpus_per_trial=0, data_dir="./data/"):
    config = {
        "lr": tune.uniform(1e-2, 1e-5),
        "batch_size": tune.choice([4, 6, 8]),
        "weight_decay": tune.uniform(1e-2, 1e-5),
        "num_workers": tune.choice([10]),
    }
    num_epochs = 5
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)
    reporter = tune.CLIReporter(
        parameter_columns=["batch_size", "lr", "weight_decay"],
        metric_columns=["loss", "f1"],
    )
    gpus_per_trial = 0
    data_dir = "./data/"

    train_fn_with_params = tune.with_parameters(
        train_idd_tune, num_epochs=num_epochs, data_dir=data_dir
    )

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_idd_asha",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    result = tuner.fit()
    print("Best hypeparameters found were: ", result.get_best_result().config)


if __name__ == "__main__":
    # tune_idd_asha(2, 2, 1)
    train_idd_no_tune()
