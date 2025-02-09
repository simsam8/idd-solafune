import numpy as np
import lightning as pl
import sklearn
import tifffile
import torch
from torch.utils.data import DataLoader


def load_mask(mask_path):
    mask = np.load(mask_path)  # (4, H, W), uint8
    assert mask.shape == (4, 1024, 1024)
    mask = mask.transpose(1, 2, 0)  # (H, W, 4)
    return mask.astype(np.float32) / 255.0  # normalize to [0, 1]


def load_image(image_path):
    image = tifffile.imread(image_path)  # (H, W, 12), float64
    assert image.shape == (1024, 1024, 12)
    image = np.nan_to_num(image)  # replace NaN with 0
    return image.astype(np.float32)


def normalize_image(image):
    # mean of train images
    mean = np.array(
        [
            285.8190561180765,
            327.22091430696577,
            552.9305957826701,
            392.1575148484924,
            914.3138803812591,
            2346.1184507500043,
            2884.4831706095824,
            2886.442429854111,
            3176.7501338557763,
            3156.934442092072,
            1727.1940075511282,
            848.573373995044,
        ],
        dtype=np.float32,
    )

    # std of train images
    std = np.array(
        [
            216.44975668759372,
            269.8880248304874,
            309.92790753407064,
            397.45655590699,
            400.22078920482215,
            630.3269651264278,
            789.8006920468097,
            810.4773696969773,
            852.9031432100967,
            807.5976198303886,
            631.7808113929271,
            502.66788721341396,
        ],
        dtype=np.float32,
    )

    mean = mean.reshape(12, 1, 1)
    std = std.reshape(12, 1, 1)

    return (image - mean) / std


class TrainValDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, sample_indices, augmentations=None):
        self.image_paths, self.mask_paths = [], []
        for i in sample_indices:
            self.image_paths.append(data_root / "train_images" / f"train_{i}.tif")
            self.mask_paths.append(data_root / "train_masks" / f"train_{i}.npy")
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {
            "image": load_image(self.image_paths[idx]),
            "mask": load_mask(self.mask_paths[idx]),
        }

        if self.augmentations is not None:
            sample = self.augmentations(**sample)

        sample["image"] = sample["image"].transpose(2, 0, 1)  # (12, H, W)
        sample["mask"] = sample["mask"].transpose(2, 0, 1)  # (4, H, W)

        sample["image"] = normalize_image(sample["image"])

        # add metadata
        sample["image_path"] = str(self.image_paths[idx])
        sample["mask_path"] = str(self.mask_paths[idx])

        return sample


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root):
        self.image_paths = []
        for i in range(118):  # evaluation_0.tif to evaluation_117.tif
            self.image_paths.append(
                data_root / "evaluation_images" / f"evaluation_{i}.tif"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        sample = {
            "image": load_image(self.image_paths[idx]),
        }

        sample["image"] = sample["image"].transpose(2, 0, 1)  # (12, H, W)
        sample["image"] = normalize_image(sample["image"])

        # add metadata
        sample["image_path"] = str(self.image_paths[idx])

        return sample


class IDDDataModule(pl.LightningDataModule):
    def __init__(self, data_root, augmentations=None, batch_size=8) -> None:
        super().__init__()
        self.data_dir = data_root
        self.batch_size = batch_size
        self.augmentations = augmentations

    def setup(self, stage=None) -> None:
        if stage == "fit":
            sample_indices = list(range(176))  # train_0.tif to train_175.tif
            train_indices, val_indices = sklearn.model_selection.train_test_split(
                sample_indices, test_size=0.2, random_state=42
            )

            self.idd_train = TrainValDataset(
                self.data_dir, train_indices, augmentations=self.augmentations
            )
            self.idd_val = TrainValDataset(
                self.data_dir, val_indices, augmentations=None
            )

        if stage == "test":
            self.idd_test = TestDataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.idd_train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.idd_val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.idd_test, batch_size=self.batch_size, num_workers=4)
