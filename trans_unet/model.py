import timm
import torch
import torch.nn as nn


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
        last_skip=False,
    ) -> None:
        super().__init__()
        self.last_skip = last_skip
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
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.up(x)
        return x


class DecoderCUP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bridge = ConvBlock(768, 512, padding=1, kernel_size=3, stride=1)
        self.skip_channels = [256, 512, 1024]
        self.decoder_block1 = DecoderBlock(512, 256, skip_channels=1024)
        self.decoder_block2 = DecoderBlock(256, 128, skip_channels=512)
        self.decoder_block3 = DecoderBlock(128, 64, skip_channels=256, last_skip=False)

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
        self.segmentation = ConvBlock(
            input_channels, segmentation_channels, padding=1
        )  # Adjust to match input resolution

    def forward(self, x):
        x = self.conv1(x)
        x = self.segmentation(x)
        return x


class TransUNet(nn.Module):
    def __init__(self, input_channels=12, segmentation_channels=4) -> None:
        super().__init__()
        self.resnet_vit = timm.models.vit_base_r50_s16_224()
        self.resnet_stem = self.resnet_vit.patch_embed.backbone.stem

        # Adjust channels to match the expected input for Resnet50 encoder
        self.adjust_channels = nn.Conv2d(
            input_channels, 3, kernel_size=1, stride=1, padding=0
        )
        self.resnet_stages = self.resnet_vit.patch_embed.backbone.stages
        self.resnet_norm = self.resnet_vit.patch_embed.backbone.norm
        self.resnet_head = self.resnet_vit.patch_embed.backbone.head
        self.projection = self.resnet_vit.patch_embed.proj

        self.decoder = DecoderCUP()

        self.head = SegmentationHead(64, segmentation_channels)

    def forward(self, x):

        # -- ResNet --
        x = self.adjust_channels(x)
        x = self.resnet_stem(x)
        skip_features = []
        for stage in self.resnet_stages.children():
            x = stage(x)
            skip_features.append(x)

        x = self.resnet_norm(x)
        x = self.resnet_head(x)
        x = self.projection(x)  # (b, hidden, n_patches^(1/2), n_patches^(1/2))
        patch_size = x.shape[-1]
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        # -- Forward pass through ViT --
        for module in list(self.resnet_vit.children())[1:]:
            x = module(x)

        # Reshape for CUP
        x = x.transpose(-1, -2)
        x = x.reshape(x.shape[0], x.shape[1], patch_size, patch_size)

        # -- Applying CUP --
        x = self.decoder(x, skip_features)

        # Segmentation Head
        x = self.head(x)

        return x


if __name__ == "__main__":
    vit = timm.models.vit_base_r50_s16_224()
    test_input = torch.ones((1, 12, 1024, 1024))
    print(vit.patch_embed.backbone.children())
    for child in vit.patch_embed.backbone.children():
        print(child)

    # model = TransUNet()
    # model(test_input)
