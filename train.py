from pathlib import Path

import albumentations as albu
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import TensorBoardLogger
from ray import train, tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler

from datasets import IDDDataModule
from Models import Model

# seed_everything(42)


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


def train_idd(config):
    model = Model(config)
    trainer = pl.Trainer(max_epochs=2, enable_progress_bar=True)
    trainer.fit(model)


def train_idd_no_tune():
    config = {"batch_size": 4, "lr": 1e-4, "weight_decay": 1e-5, "num_workers": 10}
    train_idd(config)


def train_idd_tune(config, num_epochs, data_dir: Path, log_dir: Path):
    model = Model(config)
    data_module = IDDDataModule(data_dir, augmentations, config["batch_size"])
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        logger=TensorBoardLogger(save_dir="logs/", name=""),
        enable_progress_bar=False,
        default_root_dir=log_dir / "ray_logs",
        log_every_n_steps=20,
        callbacks=[
            TuneReportCheckpointCallback(
                {
                    "train/loss": "train/loss",
                    "train/f1": "train/f1",
                    "val/loss": "val/loss",
                    "val/f1": "val/f1",
                },
                filename="checkpoint.ckpt",
                on="validation_end",
            )
        ],
    )

    trainer.fit(model, datamodule=data_module)


def tune_idd_asha(
    data_dir: Path, log_dir: Path, num_samples=10, num_epochs=10, gpus_per_trial=0.0
):
    config = {
        "lr": tune.uniform(1e-2, 1e-5),
        "batch_size": tune.choice([2]),
        "weight_decay": tune.uniform(1e-2, 1e-5),
        "num_workers": tune.choice([10]),
    }
    # TODO: Tweak grace_period and reduction_factor when tuning
    scheduler = ASHAScheduler(max_t=num_epochs, grace_period=2, reduction_factor=2)
    reporter = tune.CLIReporter(
        parameter_columns=["batch_size", "lr", "weight_decay"],
        metric_columns=["train/loss", "val/loss", "train/f1", "val/f1"],
    )

    train_fn_with_params = tune.with_parameters(
        train_idd_tune, num_epochs=num_epochs, data_dir=data_dir, log_dir=log_dir
    )

    resources_per_trial = {"cpu": 1, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="val/f1",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_idd_asha",
            progress_reporter=reporter,
            storage_path=str(log_dir),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="val/f1",
                checkpoint_score_order="max",
            ),
        ),
        param_space=config,
    )
    result = tuner.fit()
    if num_epochs >= 10:
        best_result = result.get_best_result(scope="last-10-avg")
    else:
        best_result = result.get_best_result(scope="last")
    print(
        "Best hypeparameters found were: ",
        best_result.config,
    )
    return best_result.get_best_checkpoint(metric="val/f1", mode="max")


if __name__ == "__main__":
    data_path = Path("data/").absolute()
    log_dir = Path("logs/").absolute()
    best_checkpoint = tune_idd_asha(data_path, log_dir, 4, 5, 0.5)

    cp = (
        (Path(best_checkpoint.path) / "checkpoint.ckpt")
        .resolve()
        .relative_to(Path("./").resolve())
    )
    print(cp)
    with open("models/best_trials.txt", "a") as f:
        f.writelines(str(cp))
