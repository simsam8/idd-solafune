import torch
import torch.nn as nn
from timm.layers.norm_act import GroupNormAct
from timm.layers.std_conv import StdConv2dSame
from timm.models._factory import create_model


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        padding=1,
        kernel_size=3,
        stride=1,
        with_nonlinearity=True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            padding=padding,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_channels=0,
    ) -> None:
        super().__init__()
        self.conv1 = ConvBlock(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.conv2 = ConvBlock(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCUP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = ConvBlock(768, 512, padding=1, kernel_size=3, stride=1)
        self.decoder_block1 = DecoderBlock(512, 256, skip_channels=512)
        self.decoder_block2 = DecoderBlock(256, 128, skip_channels=256)
        self.decoder_block3 = DecoderBlock(128, 64, skip_channels=64)

    def forward(self, x, skip_features):
        x = self.bridge(x)
        s1, s2, s3 = reversed(skip_features)
        x = self.decoder_block1(x, s1)
        x = self.decoder_block2(x, s2)
        x = self.decoder_block3(x, s3)
        return x


class SegmentationHead(nn.Module):
    def __init__(self, input_channels, segmentation_channels) -> None:
        super().__init__()
        self.conv1 = DecoderBlock(input_channels, input_channels)
        self.segmentation = ConvBlock(input_channels, segmentation_channels, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.segmentation(x)
        return x


class TransUNet(nn.Module):
    """
    Implementation of TransUNet using
    ViT-base ResNet50 pretrained on ImageNet-21k.

    12 Transformer layers.
    Patch size of 16.
    ResNet50 pre-trained on 224x224 images.
    """

    def __init__(self, input_channels=12, segmentation_channels=4) -> None:
        super().__init__()
        self.resnet_vit = create_model(
            "vit_base_r50_s16_224.orig_in21k", pretrained=True, num_classes=0
        )
        self.resnet_stem = self.resnet_vit.patch_embed.backbone.stem
        self.stem = nn.Sequential(
            StdConv2dSame(input_channels, 64, kernel_size=7, stride=2, bias=False),
            GroupNormAct(64, 32),
        )

        self.resnet_stages = self.resnet_vit.patch_embed.backbone.stages
        self.resnet_post = nn.Sequential(
            self.resnet_vit.patch_embed.backbone.norm,
            self.resnet_vit.patch_embed.backbone.head,
        )
        self.projection = self.resnet_vit.patch_embed.proj

        self.vit = nn.Sequential(*list(self.resnet_vit.children())[1:])

        self.decoder = DecoderCUP()

        self.head = SegmentationHead(64, segmentation_channels)
        self.unlock_encoder(False)

    def unlock_encoder(self, frozen=True):
        for param in self.resnet_stages.parameters():
            param.requires_grad = frozen
        for param in self.resnet_post.parameters():
            param.requires_grad = frozen
        for param in self.projection.parameters():
            param.requires_grad = frozen
        for param in self.vit.parameters():
            param.requires_grad = frozen

    def forward(self, x):
        # -- ResNet --
        x = self.stem(x)
        skip_features = [x]  # Add stem as first skip connection

        # Max pool after adding to skip connections
        x = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)(x)

        for i, stage in enumerate(self.resnet_stages.children()):
            x = stage(x)
            # Last stage is not used as skip connection
            if i != 2:
                skip_features.append(x)

        x = self.resnet_post(x)
        x = self.projection(x)  # (b, hidden, n_patches^(1/2), n_patches^(1/2))
        patch_size = x.shape[-1]
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # -- Forward pass through ViT --
        x = self.vit(x)

        # Reshape for CUP
        x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], x.shape[1], patch_size, patch_size)

        # -- Applying CUP --
        x = self.decoder(x, skip_features)

        # Segmentation Head
        x = self.head(x)

        return x


if __name__ == "__main__":
    # For testing purposes
    vit = create_model(
        "vit_base_r50_s16_224.orig_in21k", pretrained=True, num_classes=0
    )
    test_input = torch.ones((1, 12, 1024, 1024))
    print(vit)
    model = TransUNet()
    model(test_input)
