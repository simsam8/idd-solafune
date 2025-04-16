from ray import tune

# Common parameters
hyper_params = {
    "lr": 1e-4,
    "weight_decay": 1e-2,
    "num_workers": 12,
}

unet = {
    "model_name": "unet",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "unet",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "classes": 4,
    },
    "train_batch_size": 8,
    "val_batch_size": 8,
    "test_batch_size": 8,
    "num_workers": 12,
}

deeplab = {
    "model_name": "deeplab",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "deeplabv3plus",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "classes": 4,
    },
    "train_batch_size": 8,
    "val_batch_size": 8,
    "test_batch_size": 8,
    "num_workers": 12,
}

segformer = {
    "model_name": "segformer",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "segformer",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "classes": 4,
    },
    "train_batch_size": 4,
    "val_batch_size": 4,
    "test_batch_size": 4,
    "num_workers": 12,
}

transunet = {
    "model_name": "transunet",
    "model_type": "transunet",
    "model_params": {
        "segmentation_channels": 4,
    },
    "train_batch_size": 3,
    "val_batch_size": 1,
    "test_batch_size": 3,
    "num_workers": 12,
}

raytune_config = {
    "model_type": "pt_seg",
    "model_params": tune.choice(
        [
            {
                "arch": "unet",
                "encoder_name": "tu-tf_efficientnetv2_s",
                "encoder_weights": "imagenet",
                "in_channels": 12,
                "classes": 4,
            },
            {
                "arch": "deeplabv3plus",
                "encoder_name": "resnet50",
                "encoder_weights": "imagenet",
                "in_channels": 12,
                "classes": 4,
            },
        ]
    ),
    "lr": tune.uniform(1e-2, 1e-5),
    # "batch_size": tune.choice([2]),
    "batch_size": 2,
    "weight_decay": tune.uniform(1e-2, 1e-5),
    # "num_workers": tune.choice([10]),
    "num_workers": 4,
}
