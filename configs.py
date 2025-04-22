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
    "batch_accumulation": 2,
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
    "batch_accumulation": 2,
}

segformer = {
    "model_name": "segformer",
    "model_type": "pt_seg",
    "model_params": {
        "arch": "segformer",
        "encoder_name": "mit_b5",
        "encoder_weights": "imagenet",
        "classes": 4,
    },
    "train_batch_size": 3,
    "val_batch_size": 1,
    "test_batch_size": 3,
    "num_workers": 12,
    "batch_accumulation": 5,
}

transunet = {
    "model_name": "transunet",
    "model_type": "transunet",
    "model_params": {
        "segmentation_channels": 4,
    },
    "train_batch_size": 2,
    "val_batch_size": 1,
    "test_batch_size": 3,
    "num_workers": 12,
    "batch_accumulation": 8,
}

vit_seg = {
    "model_name": "vit_seg",
    "model_type": "vit_seg",
    "model_params": {},
    "train_batch_size": 3,
    "val_batch_size": 1,
    "test_batch_size": 3,
    "num_workers": 12,
    "batch_accumulation": 5,
}
