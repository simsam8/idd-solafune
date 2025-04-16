import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ViT_B_16_Weights, vit_b_16


# Vision Transformer for 12-channel input and 4-class output
class ViTSegmentation(nn.Module):
    def __init__(self):
        super().__init__()
        # Loading pre-trained ViT-B/16 model (ImageNet-1k weights)
        vit = vit_b_16(
            weights=ViT_B_16_Weights.IMAGENET1K_V1
        )  # default 3-channel, 224x224 patches

        # Adapting input conv to 12 channels
        # Original patch embedding conv (3->768) weights
        old_conv = vit.conv_proj  # nn.Conv2d(3, hidden_dim, kernel_size=16, stride=16)
        old_weight = old_conv.weight.data  # shape (hidden_dim, 3, 16, 16)

        # Createing new conv with 12 input channels
        new_conv = nn.Conv2d(
            in_channels=12,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        # Initialize new conv weights using the mean of pretrained RGB weights (replicated for all 12 channels)
        new_weight = old_weight.mean(dim=1, keepdim=True).repeat(
            1, 12, 1, 1
        )  # shape (768,12,16,16)
        new_conv.weight.data.copy_(new_weight)
        if old_conv.bias is not None:
            new_conv.bias.data.copy_(old_conv.bias.data)
        vit.conv_proj = new_conv  # replace conv_proj in the model

        # Resize position embeddings to 1024x1024 input
        vit.image_size = 1024  # update image size
        num_patches_old = (
            224 // vit.patch_size
        ) ** 2  # original patch count (14*14=196)
        num_patches_new = (1024 // vit.patch_size) ** 2  # new patch count (64*64=4096)
        # Separate class token embedding and patch embeddings from the old pos_embedding
        pos_embed = (
            vit.encoder.pos_embedding.data
        )  # shape (1, 197, 768) for old (including class token)
        class_pos_emb = pos_embed[:, :1, :]  # shape (1, 1, 768)
        spatial_pos_emb = pos_embed[:, 1:, :]  # shape (1, 196, 768)
        # Reshape and interpolate the spatial position embeddings to 64x64
        old_grid = int(num_patches_old**0.5)  # 14
        new_grid = int(num_patches_new**0.5)  # 64
        spatial_pos_emb = spatial_pos_emb.reshape(1, old_grid, old_grid, -1).permute(
            0, 3, 1, 2
        )  # (1,768,14,14)
        new_spatial_pos_emb = F.interpolate(
            spatial_pos_emb,
            size=(new_grid, new_grid),
            mode="bilinear",
            align_corners=False,
        )
        new_spatial_pos_emb = new_spatial_pos_emb.permute(0, 2, 3, 1).reshape(
            1, new_grid * new_grid, -1
        )  # (1,4096,768)
        # Combine with class token embedding
        new_pos_embed = torch.cat(
            [class_pos_emb, new_spatial_pos_emb], dim=1
        )  # shape (1, 4097, 768)
        vit.encoder.pos_embedding = nn.Parameter(
            new_pos_embed
        )  # set new position embedding parameter

        # Remove classification head and add 4-class segmentation head
        vit.heads = (
            nn.Sequential()
        )  # discard the classification head (wnot use vit.heads forward)
        self.vit = vit  # backbone with transformer encoder
        # Enables gradient checkpointing
        # self.vit.encoder.gradient_checkpointing = True

        # Dropout
        self.dropout = nn.Dropout(0.3)
        # Segmentation head: 1x1 conv to 4 classes (will operate on patch embeddings)
        self.seg_head = nn.Conv2d(
            vit.hidden_dim, 4, kernel_size=1
        )  # vit.hidden_dim is 768 for ViT-B/16

        # Freeze ViT backbone
        for param in self.vit.parameters():
            param.requires_grad = False

        # Unfreeze segmentation head
        for param in self.seg_head.parameters():
            param.requires_grad = True

        # Also unfreeze dropout if needed (not usually trainable, but for completeness)
        for param in self.dropout.parameters():
            param.requires_grad = True

    def unlock_encoder(self, frozen=True):
        for param in self.vit.parameters():
            param.requires_grad = frozen

    def forward(self, x):
        # Forward pass: returns logits of shape (batch, 4, 1024, 1024)
        B, C, H, W = x.shape  # expecting C=12, H=W=1024
        # Patchify input and get patch embeddings (B, num_patches, hidden_dim)
        patches = self.vit._process_input(
            x
        )  # uses conv_proj internally to get (B, 4096, 768) patch tokens
        # Prepend class token and encode through transformer
        cls_token_expanded = self.vit.class_token.expand(B, -1, -1)  # (B,1,768)
        tokens = torch.cat([cls_token_expanded, patches], dim=1)  # (B, 4096+1, 768)
        encoded = self.vit.encoder(
            tokens
        )  # (B, 4097, 768) after positional embedding + Transformer encoder + LN
        patch_embeddings = encoded[
            :, 1:, :
        ]  # drop the [CLS] token output, keep patch outputs (B, 4096, 768)
        # Reshape to spatial feature map (B, 768, 64, 64)
        patch_embeddings = patch_embeddings.permute(0, 2, 1).reshape(
            B, self.vit.hidden_dim, int(H / 16), int(W / 16)
        )
        patch_embeddings = self.dropout(patch_embeddings)
        # Apply 1x1 conv to get class logits for each patch, then upsample to full resolution
        patch_logits = self.seg_head(
            patch_embeddings
        )  # (B, 4, 64, 64) logits for each patch
        full_res_logits = F.interpolate(
            patch_logits, scale_factor=16, mode="bilinear", align_corners=False
        )  # (B,4,1024,1024)
        return full_res_logits
