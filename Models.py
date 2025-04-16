import lightning as pl
import segmentation_models_pytorch as smp
import torch
from timm.optim._optim_factory import create_optimizer_v2
from timm.scheduler.scheduler_factory import create_scheduler_v2

from trans_unet.model import TransUNet
from vision_transformer.model import ViTSegmentation


# class ComboLoss(nn.Module):
#     def __init__(self, smooth=1e-5):
#         super().__init__()
#         self.bce = nn.BCEWithLogitsLoss()
#         self.smooth = smooth
#
#     def forward(self, logits, targets):
#         bce_loss = self.bce(logits, targets)
#         probs = torch.sigmoid(logits)
#         num = 2 * (probs * targets).sum(dim=(2, 3))
#         den = (probs + targets).sum(dim=(2, 3))
#         dice = (num + self.smooth) / (den + self.smooth)
#         dice_loss = 1 - dice.mean()
#         return 0.5 * bce_loss + 0.5 * dice_loss


class Model(pl.LightningModule):
    def __init__(self, config, epochs):
        super().__init__()
        self.save_hyperparameters()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.train_batch_size = config["train_batch_size"]
        self.val_batch_size = config["val_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.num_workers = config["num_workers"]
        self.class_names = ["grassland_shrubland", "logging", "mining", "plantation"]
        self.epochs = epochs

        if config["model_type"] == "pt_seg":
            self.model = smp.create_model(**config["model_params"])
        elif config["model_type"] == "transunet":
            self.model = TransUNet(config["model_params"]["in_channels"], 4)
        elif config["model_type"] == "vit_seg":
            self.model = ViTSegmentation()

        self.dice_loss_fn = smp.losses.DiceLoss(
            mode=smp.losses.MULTILABEL_MODE, from_logits=True
        )
        self.bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)

        # # Loss
        # self.loss_fn = ComboLoss()
        # # Metric
        # self.val_f1 = MultilabelF1Score(num_labels=4, average='macro')

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

        self.log(
            f"{stage}/loss",
            loss.detach().cpu(),
            sync_dist=True,
            prog_bar=False,
            # on_epoch=True,
            on_step=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def shared_epoch_end(self, outputs, stage):
        def log(name, tensor, prog_bar=False):
            self.log(
                f"{stage}/{name}",
                # tensor.to(self.device),
                tensor.detach().cpu(),
                sync_dist=True,
                prog_bar=prog_bar,
                # on_epoch=True,
            )

        # aggregate tp, fp, fn, tn to compose F1 score for each class
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        f1_scores = {}
        for i, class_name in enumerate(self.class_names):
            f1_scores[class_name] = smp.metrics.f1_score(
                tp[:, i], fp[:, i], fn[:, i], tn[:, i], reduction="macro-imagewise"
            )
            log(f"f1/{class_name}", f1_scores[class_name], prog_bar=False)

        f1_avg = torch.stack([v for v in f1_scores.values()]).mean()
        log("f1", f1_avg, prog_bar=True)
        # return outputs

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer_v2(
            self.parameters(),
            opt="adamw",
            lr=self.lr,
            weight_decay=self.weight_decay,
            filter_bias_and_bn=True,  # filter out bias and batchnorm from weight decay
        )

        # lr scheduler
        scheduler, _ = create_scheduler_v2(
            optimizer,
            sched="cosine",
            num_epochs=self.epochs,
            min_lr=0.0,
            warmup_lr=1e-5,
            warmup_epochs=0,
            warmup_prefix=False,
            step_on_epochs=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        # workaround for timm's scheduler:
        # https://github.com/Lightning-AI/lightning/issues/5555#issuecomment-1065894281
        scheduler.step(
            epoch=self.current_epoch
        )  # timm's scheduler need the epoch value       )
