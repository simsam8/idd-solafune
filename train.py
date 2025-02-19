from argparse import ArgumentParser
from pathlib import Path

import albumentations as albu
import lightning as pl
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from ray import train, tune
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

import configs
from datasets import IDDDataModule
from Models import Model
from utils import save_checkpoint

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


def pl_trainer(config, data_dir, epochs=10):
    model = Model(config)
    trainer = pl.Trainer(max_epochs=epochs, enable_progress_bar=True)
    data_module = IDDDataModule(data_dir, augmentations, config["batch_size"])
    trainer.fit(model, datamodule=data_module)


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
            ),
            EarlyStopping(monitor="val/loss", mode="min", patience=7),
        ],
    )

    trainer.fit(model, datamodule=data_module)
    return model, config


def tune_idd_asha(
    data_dir: Path,
    log_dir: Path,
    config: dict,
    num_samples=10,
    num_epochs=10,
    gpus_per_trial=0.0,
    cpu_per_trial=1,
    grace_period=10,
    reduction_factor=2,
):
    # TODO: Tweak grace_period and reduction_factor when tuning
    scheduler = ASHAScheduler(
        max_t=num_epochs, grace_period=grace_period, reduction_factor=reduction_factor
    )
    # optuna_search = OptunaSearch(metric="val/loss", mode="min")
    reporter = tune.CLIReporter(
        parameter_columns=["batch_size", "lr", "weight_decay"],
        metric_columns=["train/loss", "val/loss", "train/f1", "val/f1"],
    )

    train_fn_with_params = tune.with_parameters(
        train_idd_tune, num_epochs=num_epochs, data_dir=data_dir, log_dir=log_dir
    )

    resources_per_trial = {"cpu": cpu_per_trial, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
        tune.with_resources(train_fn_with_params, resources=resources_per_trial),
        tune_config=tune.TuneConfig(
            metric="val/loss",
            mode="min",
            scheduler=scheduler,
            # search_alg=optuna_search,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_idd_asha",
            progress_reporter=reporter,
            storage_path=str(log_dir),
            checkpoint_config=train.CheckpointConfig(
                num_to_keep=3,
                checkpoint_score_attribute="val/loss",
                checkpoint_score_order="min",
            ),
        ),
        param_space=config,
    )
    result = tuner.fit()
    best_result = result.get_best_result(scope="last")
    print(
        "Best hypeparameters found were: ",
        best_result.config,
    )
    return best_result.checkpoint


def main(args):
    data_path = Path("data/").absolute()
    pl_trainer(getattr(configs, args.config), data_path, epochs=int(args.epochs))


if __name__ == "__main__":
    # Absolute datapaths to work with parallell processes
    # data_path = Path("data/").absolute()
    # log_dir = Path("logs/").absolute()
    # models_dir = Path("models/")
    # best_checkpoint = tune_idd_asha(
    #     data_path,
    #     log_dir,
    #     base_config,
    #     num_samples=10,
    #     num_epochs=10,
    #     gpus_per_trial=0.5,
    #     cpu_per_trial=4,
    #     grace_period=1,
    #     reduction_factor=2,
    # )
    # save_checkpoint(best_checkpoint, models_dir)
    parser = ArgumentParser()
    parser.add_argument("--epochs", required=True)
    parser.add_argument("--config", required=True)
    main(parser.parse_args())
