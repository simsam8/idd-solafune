from ray import tune

unet = {
    "model_name": "unet",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "unet",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "in_channels": 12,
        "classes": 4,
    },
    "lr": 0.001,
    "batch_size": 8,
    "weight_decay": 0,
    "num_workers": 12,
}

deeplab = {
    "model_name": "deeplab",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "deeplabv3plus",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "in_channels": 12,
        "classes": 4,
    },
    "lr": 0.001,
    "batch_size": 8,
    "weight_decay": 0,
    "num_workers": 12,
}

segformer = {
    "model_name": "segformer",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "segformer",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "in_channels": 12,
        "classes": 4,
    },
    "lr": 0.001,
    "batch_size": 4,
    "weight_decay": 0,
    "num_workers": 12,
}

transunet = {
    "model_name": "transunet",
    "model_type": "transunet",
    "lr": 0.001,
    "batch_size": 2,
    "weight_decay": 0,
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
