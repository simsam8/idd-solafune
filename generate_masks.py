import json
from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np
import tifffile
from tqdm import tqdm

data_dir = Path("./data")

train_file_names = [f"train_{i}.tif" for i in range(176)]  # train_0.tif ~ train_175.tif
class_names = ["grassland_shrubland", "logging", "mining", "plantation"]

# Load annotations and create masks/labels for training

with open(data_dir / "train_annotations.json", "r") as f:
    raw_annotations = json.load(f)

annotations: dict[str, dict[str, list[list[float]]]] = (
    {}
)  # file_name -> class_name -> polygons
for fn in tqdm(train_file_names):
    ann: dict[str, list[list[float]]] = {}  # class_name -> polygons
    for class_name in class_names:
        ann[class_name] = []

    for tmp_img in raw_annotations["images"]:
        if tmp_img["file_name"] == fn:
            for tmp_ann in tmp_img["annotations"]:
                ann[tmp_ann["class"]].append(tmp_ann["segmentation"])

    annotations[fn] = ann

mask_save_dir = data_dir / "train_masks"
mask_save_dir.mkdir(parents=True, exist_ok=True)

for fn in tqdm(train_file_names):
    mask = np.zeros((4, 1024, 1024), dtype=np.uint8)
    anns = annotations[fn]
    for class_idx, class_name in enumerate(class_names):
        polygons = anns[class_name]
        cv2.fillPoly(
            mask[class_idx],
            [np.array(poly).astype(np.int32).reshape(-1, 2) for poly in polygons],
            255,
        )

    np.save(mask_save_dir / fn.replace(".tif", ".npy"), mask)

# Create images for visualization

vis_save_dir = data_dir / "vis_train"
vis_save_dir.mkdir(parents=True, exist_ok=True)

for fn in tqdm(train_file_names):
    mask = np.load(mask_save_dir / fn.replace(".tif", ".npy"))  # (4, 1024, 1024)
    vis_masks = [
        np.zeros((1024, 1024, 3), dtype=np.uint8) for _ in range(4)
    ]  # 4: (glassland_shrubland, logging, mining, plantation)
    for class_idx, class_name in enumerate(class_names):
        vis_masks[class_idx][mask[class_idx] > 0] = np.array([255, 0, 0])  # blue
        # put class_name as text on the mask
        cv2.putText(
            vis_masks[class_idx],
            class_name,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

    vis_image = tifffile.imread(data_dir / "train_images" / fn)
    vis_image = vis_image[
        :, :, [1, 2, 3]
    ]  # extract BGR channels (B2, B3, and B4 band of Sentinel-2)
    vis_image = np.nan_to_num(vis_image, nan=0)
    vis_image = (vis_image / 8).clip(0, 255).astype(np.uint8)

    partition = np.ones((1024, 5, 3), dtype=np.uint8) * 255  # white partition
    vis = np.concatenate(
        [
            vis_image,
            partition,
            vis_masks[0],
            partition,
            vis_masks[1],
            partition,
            vis_masks[2],
            partition,
            vis_masks[3],
        ],
        axis=1,
    )
    cv2.imwrite(vis_save_dir / fn.replace(".tif", ".png"), vis)
