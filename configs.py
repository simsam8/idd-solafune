from ray import tune

base_config = {
    "model_type": "pt_seg",
    "model_params": {
        "arch": "unet",
        "encoder_name": "tu-tf_efficientnetv2_s",  # use `tf_efficientnetv2_s` from timm
        "encoder_weights": "imagenet",  # always starts from imagenet pre-trained weight
        "in_channels": 12,
        "classes": 4,
    },
    "lr": tune.uniform(1e-2, 1e-5),
    "batch_size": tune.choice([2]),
    "weight_decay": tune.uniform(1e-2, 1e-5),
    "num_workers": tune.choice([10]),
}

deeplab_config = {
    "model_type": "pt_seg",
    "model_params": {
        "arch": "deeplabv3plus",
        "encoder_name": "resnet50",  # use `tf_efficientnetv2_s` from timm
        "encoder_weights": "imagenet",  # always starts from imagenet pre-trained weight
        "in_channels": 12,
        "classes": 4,
    },
    "lr": tune.uniform(1e-2, 1e-5),
    "batch_size": tune.choice([2]),
    "weight_decay": tune.uniform(1e-2, 1e-5),
    "num_workers": tune.choice([10]),
}
