---
documentclass: nldl
classoption: review
title: Applied Segmentation Models for Identifying Deforestation Drivers
bibliography: ./references.bib
header-includes: |
    \paperID{1}
    \vol{1}
    \usepackage{mathtools}
    \usepackage{graphicx}
    \usepackage{booktabs}
    \usepackage{enumitem}
    \usepackage{algorithm}
    \usepackage{algorithmic}
    \newcommand{\theHalgorithm}{\arabic{algorithm}}
    \usepackage{listings}
    \lstset{
      basicstyle=\small\ttfamily,
      breaklines,
    }
    \addbibresource{./references.bib}
    \usepackage{hyperref}
    \usepackage{url}
    \hypersetup{
      pdfusetitle,
      colorlinks,
      linkcolor = BrickRed,
      citecolor = NavyBlue,
      urlcolor  = Magenta!80!black,
    }
    \author[1]{Simon Vedaa}
    \author[1]{Khalil Ibrahim}
    \author[1]{Safiya Mahamood}
    \affil[1]{University of Bergen}
---

# Abstract
<!--This will be added later ignore-->
Add abstract when all other sections are complete.

# Introduction

In this paper we are working on the task of identifying deforestation drivers.
This is part of a Solafune ML competition[^1] where the goal is to classify and 
segment drivers of deforestation in satellite imagery.

Several architectures have been developed for image segmentation in different fields,
such as UNet [@ronneberger2015unetconvolutionalnetworksbiomedical] and TransUNet [@chen2021transunet]
for medical image segmentation, or the Segformer [@xie2021segformer] for general purpose 
segmentation.

We have several goals we want to achieve. First, achieve the best possible performance
on the competition dataset. 
Second, compare the different implementations in terms of performance vs computational cost.
Lastly, to reproduce results reported in the original architecture papers, which we 
have implemented.

Our contributions consist of applying different segmentation model architectures 
on a deforestation segmentation task, and comparing their performance.

[^1]: [Competition website](https://solafune.com/competitions/68ad4759-4686-4bb3-94b8-7063f755b43d?menu=about&tab=&modal=%22%22)



# Methods

Our entire pipeline is based upon a GitHub repository made by motokimura.[^2]
We have made a couple of modifications to the training and evaluation pipeline,
but the pre- and post-processing steps remain mainly unchanged.

[^2]: [Baseline pipeline by motokimura](https://github.com/motokimura/solafune_deforestation_baseline)

## Pre-processing

The competition data comes in the form of annotated polygons in a json file.
We convert those into tensors of 4 channels.
We apply standard image augmentations such as flipping, scaling and rotation.
Additionally we apply random cropping reducing the input image by half.
The cropping is not applied for the Vision Transformer (ViT) model.
Images are also normalized using mean and standard deviation calculated from
training images. Normalization values depend on the number of input channels, i.e. 
using only RGB or all channels.

## Post-processing

We applied a score threshold of 0.5 to binarize the predicted masks.
Additionally, we discard masks smaller than 10 000 pixels during inference.
The idea is that removing small segmentations reduces the 
number of false positive predictions.

## Model architectures

We have applied the following model architectures; UNet, DeepLabV3+, Vision Transformer (ViT), 
Segformer, and TransUNet.
We used the Pytorch Segmentation Models library [@Iakubovskii:2019] to implement UNet,
DeepLabV3+, and Segformer, while ViT and TransUNet are implemented following 
their respective papers and source code.


### Vision Transformer (ViT)

The Vision Transformer (ViT) model treats an image as a sequence of
fixed-sized patches and processes them with a standard Transformer encoder,
rather than convolutional inductive biases to learn long-range
dependencies in the data it relies on self-attention. 

#### Model Architecture

The Vision Transformer breaks an image into fixed size patches,
linearly embeds each patch (plus a learnable class token and positional embeddings) into vectors.
The result of this is then fed as sequence through standard Transformer
encoder layers that contains multi-head self-attention followed by
feed-forward networks, using the final class token as representation for prediction.

#### Implementation 

In our ViT implementation we started from the torchvision ViT-B/16
model pretrained on ImageNet [@dosovitskiy2020vit] swapped its first layer
to accept all 12 Sentinel-2 bands, resized its built-in positional embeddings
to our 1024×1024 image grid, and replaced the classification head with
a lightweight segmentation head for four land-use classes.
The transformer backbone remained frozen,
and only the new segmentation head was trained on our data.

#### Why Vision Transformer

Based on [@dosovitskiy2020vit] the ViT applies a pure Transformer to image patches and,
with large-scale pre-training, matches or exceeds CNNs on vision benchmarks.
Since our task requires capturing long-range,
multispectral context in high-resolution satellite imagery,
we wanted to see if ViT could similarly improve segmentation performance.


### Segformer

SegFormer [@xie2021segformer] is a transformer-based architecture designed for efficient semantic segmentation. It combines the strengths of hierarchical representations from convolutional networks with the global context modeling of transformers. In our project, we included SegFormer as one of the core models to evaluate its ability to identify deforestation drivers in satellite imagery.

#### Model architecture

SegFormer consists of two main components: the Mix Transformer (MiT) encoder and a lightweight MLP-based decoder. The encoder is optimized for visual tasks, using overlapping patch embeddings and a hierarchical structure to effectively capture both local and global image features. Unlike traditional Vision Transformers, SegFormer replaces explicit positional encodings with Mix-FFN modules, improving robustness to varying input resolutions.

The decoder is composed entirely of Multi-Layer Perceptrons (MLPs), which aggregate multi-scale features from the encoder. This design keeps the decoder computationally lightweight while maintaining strong segmentation performance.

#### Implementation

We used the implementation of SegFormer provided by the *segmentation_models* library [@Iakubovskii:2019], which integrates smoothly with PyTorch and supports modular experimentation. This allowed us to quickly prototype and evaluate different model variants under our unified training and evaluation pipeline.

#### Why SegFormer?

We selected the SegFormer-B5 variant, which is the most powerful configuration of the architecture. Its deep encoder is particularly effective at capturing both fine-grained and large-scale features, making it well suited for complex segmentation tasks like deforestation mapping. In addition, SegFormer has demonstrated strong benchmark results on datasets such as Cityscapes and ADE20K, indicating reliable generalization to a variety of segmentation domains.

### TransUNet

#### Architecture 

TransUNet is very similar to its predecessor UNet.
It consists of an encoder and decoder architecture,
where the main difference is the introduction of a
transformer in the encoder as seen in Figure \ref{transunet_arch}. 
The decoder block called CUP, short for Cascaded Upsampler,
consists of multiple upsampling blocks,
which are made up of a 2x bilinear upsampler followed by two 
convolutional blocks.
The decoder also uses skip connections from the CNN encoder,
and passes them into the first convolutional block in the 
corresponding upsampling stage.

#### Implementation

In our implementation we use ResNet50-VisionTransformer for the hybrid encoder,
using pre-trained weights loaded from the `timm` library.
We implement the base version of TransUNet as they do in [@chen2021transunet].
Because our inputs have three or more channels, we replaced the ResNet encoder's 
first convolutional layer; the rest of the hybrid encoder remained unchanged.


#### Motivation

According to [@chen2021transunet], TransUNet is an improvement to UNet for the task of medical image 
segmentation. Since we use UNet as one of our baseline models, we were interested 
to see if we could get similar results for our task.


![TransUNet architecture [@chen2021transunet]\label{transunet_arch}](../trans_unet/img/transunet.png)

## Ensemble models

We create two ensemble models, one with all models called `ensemble1` and one without TransUNet called `ensemble2`.
Ensemble models average the output logits of all its models. 
As shown in Results, TransUNet without post-processing performed 
significantly worse than the other models.

## Training and evaluation

### Hyperparameters

Across all models, we use a learning rate of 1e-4 and a weight decay of 1e-2,
which were found to balance convergence speed and regularization.
Training is parallelized using 12 workers to optimize data loading efficiency.

Batch sizes and accumulation steps are tuned based on the computational cost
and memory footprint of each model.
Lightweight models like UNet and DeepLabV3+ use batch sizes of 8 with accumulation of 2.
For more computationally demanding models such as ViT, TransUNet,
and SegFormer, we reduce the batch size (1–3) and increase the accumulation (5–8)
to maintain stable gradient estimates while fitting within GPU memory limits.

### Loss and Metric

<!--Maybe add the formulas as well?-->
The loss function is the sum of Dice loss and Soft Binary Cross entropy with a smoothing factor of 0.

We use the pixel-based F1 score as the evaluation metric,
in line with the competition rules. It balances precision
and recall based on the overlap between predicted and ground truth masks,
computed per class and averaged across the dataset.


### Batch gradient accumulation

As some of the model are quite large, and we have limited resources,
we decided to use batch gradient accumulation.
Instead of using larger batches, we use smaller `k` batches 
and accumulate the gradients of `N` batches before the backward pass. 
The effective batch size then becomes `kxN`. All models are trained 
on an effective batch size of either 15 or 16.
We used pytorch lightning's built in batch gradient accumulation.

### Learning rate scheduler

We use an AdamW optimizer with a cosine‐decay schedule,
the learning rate starts at our base value and smoothly decays
to zero over the full training run, with no explicit warmup period.
The cosine curve ensures that the LR decreases gently
at first and then more rapidly toward the end of training.

### Channel input

Each model is trained in two variants, one ingesting all 12
Sentinel-2 spectral bands and one using only the standard RGB channels,
so we can measure the benefit of the extra multispectral information. 

### Frozen start

We freeze the Transformer encoder for the first five epochs,
while TransUnet uses 15 epochs updating only the new segmentation
head before unfreezing the backbone for joint fine-tuning.

### Training process

We train each model for 200 epochs, evaluating on the validation set 
every 5 epochs. The final version of the model we keep, 
is the one that achieves the highest f1 score throughout training.

### Model selection

Once every model is finished training, we run them 
through our post-processing step, and calculate their validation score.
The model with the highest score is then chosen and used to generate 
the final predictions for the test set, i.e. the competition submission.


# Results

## Effect of adding minimum area

Adding a minimum area for segmentation predictions seem to improve 
model performance quite a lot, as seen in Table \ref{min_area_f1}.
Remarkably, this more than doubled TransUNet's F1 score.

\begin{table}[!ht]
\resizebox{6cm}{!}{
    \centering
    \begin{tabular}{lll}
    \hline
        Model & Min area = 0 & Min area = 10k \\ \hline
        unet\_rgb & 0.5961 & 0.6917 \\ 
        deeplab\_rgb & 0.6289 & 0.7159 \\ 
        segformer\_rgb & 0.6174 & 0.7029 \\ 
        vit\_seg\_rgb & 0.6652 & 0.7200 \\ 
        transunet\_rgb & 0.2089 & 0.6514 \\ 
        ensemble1\_rgb & \textbf{0.6727} & 0.7182 \\ 
        ensemble2\_rgb & 0.6725 & 0.7180 \\ 
        unet\_full & 0.6303 & 0.6906 \\ 
        deeplab\_full & 0.6520 & \textbf{0.7367} \\ 
        segformer\_full & 0.6302 & 0.7048 \\ 
        vit\_seg\_full & 0.6098 & 0.7072 \\ 
        transunet\_full & 0.2456 & 0.5915 \\ 
        ensemble1\_full & 0.6706 & 0.7335 \\ 
        ensemble2\_full & 0.6698 & 0.7327 \\ \hline
    \end{tabular}
    }
    \caption{Validation f1 scores with and without Minimum Area of 10k(pixels)}
    \label{min_area_f1}
\end{table}

## Effect of channels

When comparing the models trained on only RGB channels and those trained on all channels,
overall performance improves marginally.

When looking at Figure \ref{full} and Figure \ref{rgb}, both seem to produce
similar segmentations. However, the models trained on only the RGB channels seem to
predict more false positives as seen especially on the final row of predictions.

![Segmentations using all channels\label{full}](./imgs/val_preds_full.png){width=90%}

![Segmentations using rgb channels\label{rgb}](./imgs/val_preds_rgb.png){width=90%}

## Training and validation performance

### Overall performance

Most of the models seem to converge around an F1-score of 0.8 during training
and 0.6 on validation, as seen in Figure \ref{f1_train} and Figure \ref{f1_val}[^3].
TransUNet's substantially lower performance is unexpected,
only achieving an F1 score of around 0.2 in both datasets.

![Overall training f1 score\label{f1_train}](./imgs/train_f1.png){width=60%}

![Overall validation f1 score\label{f1_val}](./imgs/val_f1.png){width=60%}


[^3]: The figures only show the results from models trained on all channels, but results 
are similar for RGB as well. No post-processing is applied here.


### Class-wise performance

Looking at the F1-score of each class in Figure \ref{f1_train_classes} and Figure \ref{f1_val_classes},
we see that most models, except TransUNet, attain a similar performance in the different classes.
They perform slightly better on the training data, which is to be expected. 
All models seem to struggle with the classes `logging` and `grassland_shrubland`, more than `plantation`
and `mining`.

The logging class consists of many small lines as seen in the last two images in Figure \ref{full}, 
and the models either completely ignores those areas, or predicts logging on similar, but unrelated lines.
For the grassland/shrubland class, the models tend to overpredict. Looking at the ground truth, 
it is hard to actually see what the grassland/shrubland area is, as it blends into surrounding vegetation,
making it difficult to distinguish.


![Training f1 score for all classes\label{f1_train_classes}](./imgs/train_f1_classes.png){width=80%}

![Validation f1 score for all classes\label{f1_val_classes}](./imgs/val_f1_classes.png){width=80%}

## Training time

All the models we tried had varying sizes, and took different amount of time to train.
Referring to Table \ref{param_size} and Figure \ref{training_time}, the smallest 
models only needed around a third of the time training compared to the largest models 
TransUNet and ViT. From Table \ref{min_area_f1} we see that there 
is only a marginal increase in performance.

\begin{table}[!ht]
\resizebox{6cm}{!}{
    \centering
    \begin{tabular}{lll}
    \hline
        \textbf{Model} & \textbf{RGB} & \textbf{Full} \\ \hline
        UNet & 32.5 & 32.5 \\ 
        DeepLabV3+ & 26.7 & 26.7 \\ 
        Segformer & 82.0 & 82.0 \\ 
        Vision Transformer & 88.8 & 90.6 \\ 
        Transunet & 105 & 105 \\ \hline
    \end{tabular}
    }
    \caption{Number of parameters(millions) for models with RGB and all channels}
    \label{param_size}
\end{table}


![Training time for each model on 200 epochs\label{training_time}](./imgs/training_time.png){width=80%}

## Competition performance

Our chosen model for the competition was DeepLabV3+ trained on all channels.
It achieved an f1 score of **0.7367** on the validation data.
On the public leaderboard it achieved a score of **0.5851**,
and on the private leaderboard **0.5624**


# Discussion

## How do we interpret our results?

SegFormer (full) showed solid and consistent performance, particularly on the plantation and mining classes. While it didn't outperform DeepLabV3+, it remained competitive and stable across all classes. Its weaker results on logging and grassland_shrubland mirror trends seen in other models, likely due to the subtle patterns in those categories.

## Did we achieve our objectives?

Despite its relatively lightweight architecture and removal of explicit positional encodings, SegFormer delivers competitive results while maintaining significantly lower training time than models like ViT and TransUNet as seen in Figure 9. This efficiency is largely due to its simplified decoder and hierarchical encoder design. Given its faster training and strong performance across classes, SegFormer offers an excellent trade-off between complexity and accuracy—outperforming several more resource-intensive models in practical terms.

## Why did the larger models perform worse then the smaller ones?

Larger models like TransUNet and ViT underperformed compared to simpler architectures such as DeepLabV3+ and SegFormer. A key reason is underfitting—especially for TransUNet—which likely stems from limited data provided and insufficient training time to optimize such a complex architecture. TransUNet's low and unstable class-wise F1 scores indicate that it struggled to learn meaningful patterns from the dataset.

These larger models also depend heavily on precise hyperparameter tuning and benefit from large-scale datasets, which we did not have. Additionally, transformer-heavy models lack built-in spatial priors, making them less suited for tasks like satellite image segmentation unless paired with extensive pretraining.
Meanwhile, models like SegFormer and DeepLabV3+ balance capacity and efficiency well. They leverage inductive biases and hierarchical structures that are better aligned with the spatial nature of our task, allowing them to generalize more effectively with fewer resources.


<!--Ignore TODO-->
TODO:

- How do we interpret our results?
- Did we achieve our objectives?
- Why did the larger models perform worse then the smaller ones?
- Hyperparameter tuning

# References
