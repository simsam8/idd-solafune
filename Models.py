import lightning as pl
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation


def seg_former():
    model = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b0-finetuned-ade-512-512",
    )
    model.config.num_channels = 12  

    # manually replace the first convolutional layer to accept 12 channels
    new_conv = nn.Conv2d(
        in_channels=12,
        out_channels=model.config.hidden_sizes[0],
        kernel_size=7,
        stride=4,
        padding=3,
        bias=False,
    )

    model.segformer.encoder.patch_embeddings[0].proj = new_conv

    # reinitialize the weights of the new conv layer
    nn.init.kaiming_normal_(model.segformer.encoder.patch_embeddings[0].proj.weight, mode="fan_out", nonlinearity="relu")

    # modify classifier to output 4 segmentation classes
    model.decode_head.classifier = nn.Conv2d(
        in_channels=model.config.decoder_hidden_size,
        out_channels=4,  
        kernel_size=1,
    )
    return model

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()

        self.lr = config["lr"]
        self.weight_decay = config["weight_decay"]
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.class_names = ["grassland_shrubland", "logging", "mining", "plantation"]

        if config["model_type"] == "pt_seg":
            self.model = smp.create_model(**config["model_params"])
        elif config["model_type"] == "seg_former":
            self.model = seg_former()

        self.dice_loss_fn = smp.losses.DiceLoss(
            mode=smp.losses.MULTILABEL_MODE, from_logits=True
        )
        self.bce_loss_fn = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.0)

        self.training_step_outputs = []
        self.validation_step_outputs = []

    def forward(self, image):
        # assuming image is already normalized
        return self.model(image).logits  # logits

    def shared_step(self, batch, stage):
        image = batch["image"]
        mask = batch["mask"]

        logits_mask = self.forward(image)

        logits_mask = torch.nn.functional.interpolate(
            logits_mask, size=mask.shape[2:], mode="bilinear", align_corners=False
        )

        print(f"logits_mask shape: {logits_mask.shape}")
        print(f"mask shape: {mask.shape}")

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
            loss,
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
                tensor,
                sync_dist=False,
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
        return outputs

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        self.training_step_outputs.clear()

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "val")
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )