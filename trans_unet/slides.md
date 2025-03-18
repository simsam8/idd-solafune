
![](./img/title.png)

# Introduction

## Background

- U-Net, de-facto choice in medical image segmentation
- Limitations in long-range relation due to locality 
of convolution operations.
- Other studies apply self-attention to CNNs or 
use Transformers instead to capture global contexts.

__INSERT IMAGE OF UNET__


## In this paper

- Study the potential of transformers in medical image segmentation.
- Using only a transformer for encoding is not enough.
- Transformers only on global context, while CNNs extracts low-level details.
- TransUNet: a hybrid CNN-Transformer architecture
- Compare with other architectures on medical images segmentation.


# Method and Architecture

## Transformer as Encoder

- Input image $\mathbf{x} \in \mathbb{R}^{H\times W \times C}$
- Image Sequentialization
- Patch Embeddings
- Naive Upsampling
- Output a $H \times W \times S$ pixel-wise segmentation. 


## Image Sequentialization

- Reshape input $\bf x$ into a sequence of 2D patches.
- {$\mathbf{x}_{p}^{i} \in \mathbb {R}^{P^{2}} \cdot C | i = 1,\dots , N$}
- Each patch is $P \times P$.
- $N = \frac{NW}{P^2}$ is the number of patches, i.e. the sequence length.

## Patch Embedding


## TransUNet


# Experiments and Results

## Dataset and Evaluation

## Implementation Details

## Comparison with State-of-the-arts

![Comparison of the Synapse multi-organ CT dataset (average dice score % and average hausdorf distance in mm, and dice score % for each organ).](./img/comparison.png)

# Ablation Studies

## Number of Skip-connections

## Influence of Input Resolution

## Influence of Patch Size/Sequence Length



