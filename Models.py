import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights
import lightning as pl
from torchmetrics.classification import MultilabelF1Score
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Vision Transformer for 12-channel input and 4-class output
class ViTSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # Loading pre-trained ViT-B/16 model (ImageNet-1k weights)
        vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)  # default 3-channel, 224x224 patches
        
        # Adapting input conv to 12 channels
        # Original patch embedding conv (3->768) weights
        old_conv = vit.conv_proj  ## nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        old_weight = old_conv.weight.data  # shape (hidden_dim, 3, 16, 16)

        # Createing new conv with 12 input channels
        new_conv = nn.Conv2d(
            in_channels=12, 
            out_channels=old_conv.out_channels, 
            kernel_size=old_conv.kernel_size, 
            stride=old_conv.stride, 
            padding=old_conv.padding, 
            bias=(old_conv.bias is not None)
        )
        # Initialize new conv weights using the mean of pretrained RGB weights (replicated for all 12 channels)
        new_weight = old_weight.mean(dim=1, keepdim=True).repeat(1, 12, 1, 1)  ## shape (768,12,16,16)
        new_conv.weight.data.copy_(new_weight)
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
        vit.conv_proj = new_conv  # replace conv_proj in the model

        # Resize position embeddings to 1024x1024 input
        vit.image_size = 1024  # update image size
        num_patches_old = (224 // vit.patch_size) ** 2  # original patch count (14*14=196)
        num_patches_new = (1024 // vit.patch_size) ** 2  # new patch count (64*64=4096)
        # Separate class token embedding and patch embeddings from the old pos_embedding
        pos_embed = vit.encoder.pos_embedding.data  # shape (1, 197, 768) for old (including class token)
        class_pos_emb = pos_embed[:, :1, :] # shape (1, 1, 768)
        spatial_pos_emb = pos_embed[:, 1:, :] # shape (1, 196, 768)
        # Reshape and interpolate the spatial position embeddings to 64x64
        old_grid = int(num_patches_old ** 0.5) # 14
        new_grid = int(num_patches_new ** 0.5) # 64
        spatial_pos_emb = spatial_pos_emb.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)  # (1,768,14,14)
        new_spatial_pos_emb = F.interpolate(spatial_pos_emb, size=(new_grid, new_grid), mode='bilinear', align_corners=False)
        new_spatial_pos_emb = new_spatial_pos_emb.permute(0, 2, 3, 1).reshape(1, new_grid*new_grid, -1)  # (1,4096,768)
        # Combine with class token embedding
        new_pos_embed = torch.cat([class_pos_emb, new_spatial_pos_emb], dim=1)  # shape (1, 4097, 768)
        vit.encoder.pos_embedding = nn.Parameter(new_pos_embed)  # set new position embedding parameter

        # Remove classification head and add 4-class segmentation head 
        vit.heads = nn.Sequential()  ## discard the classification head (wnot use vit.heads forward)
        self.vit = vit  # backbone with transformer encoder
        # Enables gradient checkpointing
        #self.vit.encoder.gradient_checkpointing = True

        # Dropout
        self.dropout = nn.Dropout(0.3)
        # Segmentation head: 1x1 conv to 4 classes (will operate on patch embeddings)
        self.seg_head = nn.Conv2d(vit.hidden_dim, 4, kernel_size=1)  # vit.hidden_dim is 768 for ViT-B/16

        # Freeze ViT backbone
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze segmentation head
        for param in self.seg_head.parameters():
            param.requires_grad = True

        # Also unfreeze dropout if needed (not usually trainable, but for completeness)
        for param in self.dropout.parameters():
            param.requires_grad = True

    def forward(self, x):
        ## Forward pass: returns logits of shape (batch, 4, 1024, 1024)
        B, C, H, W = x.shape  # expecting C=12, H=W=1024
        # Patchify input and get patch embeddings (B, num_patches, hidden_dim)
        patches = self.vit._process_input(x) # uses conv_proj internally to get (B, 4096, 768) patch tokens
        # Prepend class token and encode through transformer
        cls_token_expanded = self.vit.class_token.expand(B, -1, -1)  # (B,1,768)
        tokens = torch.cat([cls_token_expanded, patches], dim=1) # (B, 4096+1, 768)
        encoded = self.vit.encoder(tokens) # (B, 4097, 768) after positional embedding + Transformer encoder + LN
        patch_embeddings = encoded[:, 1:, :] # drop the [CLS] token output, keep patch outputs (B, 4096, 768)
        # Reshape to spatial feature map (B, 768, 64, 64)
        patch_embeddings = patch_embeddings.permute(0, 2, 1).reshape(B, self.vit.hidden_dim, int(H/16), int(W/16))
        patch_embeddings = self.dropout(patch_embeddings)
        # Apply 1x1 conv to get class logits for each patch, then upsample to full resolution
        patch_logits = self.seg_head(patch_embeddings) # (B, 4, 64, 64) logits for each patch
        full_res_logits = F.interpolate(patch_logits, scale_factor=16, mode='bilinear', align_corners=False) # (B,4,1024,1024)
        return full_res_logits

class ComboLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2, 3))
        den = (probs + targets).sum(dim=(2, 3))
        dice = (num + self.smooth) / (den + self.smooth)
        dice_loss = 1 - dice.mean()
        return 0.5 * bce_loss + 0.5 * dice_loss

class Model(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.model = ViTSegmentation()
        # Hyperparams
        self.lr = config.get("lr", 1e-4)
        self.weight_decay = config.get("weight_decay", 1e-5)
        # Loss
        self.loss_fn = ComboLoss()
        # Metric
        self.class_names = ["grassland_shrubland", "logging", "mining", "plantation"]
        self.val_f1 = MultilabelF1Score(num_labels=4, average='macro')
        self.validation_step_outputs = []

    def forward(self, x):
        return self.model(x) # (B, 4, 1024, 1024)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            self.log("debug/max_prob", probs.max(), prog_bar=True)
            self.log("debug/min_prob", probs.min(), prog_bar=False)
            self.log("debug/mean_prob", probs.mean(), prog_bar=False)
            self.log("debug/low_conf_fraction", (probs < 0.3).float().mean(), prog_bar=False)

        self.log("train/loss", loss, on_epoch=True, batch_size=x.size(0), prog_bar=True)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"].float()
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)

        # Removing hard thresholding — let torchmetrics handle it
        probs = torch.sigmoid(logits)
        f1 = self.val_f1(probs, y.int())

        self.log("val/loss", loss, on_epoch=True, prog_bar=False, batch_size=x.size(0))
        self.log("val/f1", f1, on_epoch=True, prog_bar=True, batch_size=x.size(0))

        self.validation_step_outputs.append({"f1": f1.detach()})
        return loss

    def on_validation_epoch_end(self):
        f1_scores = [x["f1"] for x in self.validation_step_outputs if "f1" in x]
        if f1_scores:
            f1_avg = torch.stack(f1_scores).mean()
            self.log("val/f1", f1_avg, prog_bar=True)
            print(f"[VAL END] val/f1 avg: {f1_avg.item():.4f}")
        self.validation_step_outputs.clear()

    def on_epoch_start(self):
        if self.current_epoch == 5:
            print("Unfreezing ViT backbone")
            for param in self.model.vit.parameters():
                param.requires_grad = True

        # Log what's frozen
        frozen = sum(p.numel() for p in self.model.vit.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Frozen ViT params: {frozen:,} | Trainable params: {trainable:,}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = {
            "scheduler": ReduceLROnPlateau(
                optimizer,
                mode="max", # to maximize val/f1
                factor=0.5, # reduce LR by half
                patience=5, # wait 5 epochs before reducing
                verbose=True,
                min_lr=1e-6 # minimum lr
            ),
            "monitor": "val/f1",
            "interval": "epoch",
            "frequency": 1
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
