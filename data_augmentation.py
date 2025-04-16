
import json
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import affine_transform
from tqdm import tqdm
import tifffile

data_dir = Path("./data")
image_dir = data_dir / "train_images"
mask_dir = data_dir / "train_masks"
augmented_image_dir = data_dir / "aug_train_images"
augmented_mask_dir = data_dir / "aug_train_masks"
augmented_image_dir.mkdir(parents=True, exist_ok=True)
augmented_mask_dir.mkdir(parents=True, exist_ok=True)

# Load annotations
with open(data_dir / "train_annotations.json", "r") as f:
    raw_annotations = json.load(f)

# Improved Augmentation pipeline for 12-channel images
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.ShiftScaleRotate(p=0.5, border_mode=0),
    
    # Apply one of several distortions
    A.OneOf([
        A.OpticalDistortion(p=1),
        A.GridDistortion(p=1),
        A.ElasticTransform(p=1),
    ], p=0.3),
    
    # Apply one of several intensity changes
    A.OneOf([
        A.RandomBrightnessContrast(p=1),
        A.CLAHE(p=1),
        A.GaussNoise(var_limit=(10.0, 50.0), p=1),
    ], p=0.3),
    
    A.Normalize(mean=[0]*12, std=[1]*12),  # Adapted for 12-channel input
    ToTensorV2()
], additional_targets={"mask": "mask"})

new_annotations = {"images": []}

for img_info in tqdm(raw_annotations["images"], desc="Augmenting images"):
    filename = img_info["file_name"]
    image_path = image_dir / filename
    mask_path = mask_dir / filename.replace(".tif", ".npy")
    
    if not image_path.exists() or not mask_path.exists():
        continue
    
    image = tifffile.imread(image_path).astype(np.float32)  # Keep all 12 channels
    mask = np.load(mask_path).astype(np.float32)  # Ensure dtype consistency
    
    # Ensure both image and mask have shape (H, W, C)
    image = image  # Image is already in (H, W, C) format
    mask = mask.transpose(1, 2, 0)  # Convert mask from (C, H, W) to (H, W, C)
    
    # Apply augmentation
    augmented = augmentation(image=image, mask=mask)
    aug_image = augmented["image"].numpy()  # Already in (H, W, C)
    aug_mask = augmented["mask"].numpy().transpose(2, 0, 1)  # Convert back to (C, H, W)
    
    new_filename = "aug_" + filename
    new_image_path = augmented_image_dir / new_filename
    new_mask_path = augmented_mask_dir / new_filename.replace(".tif", ".npy")
    
    # Save augmented image and mask
    tifffile.imwrite(new_image_path, aug_image)
    np.save(new_mask_path, aug_mask)
    
    # Print augmentation info
    print(f"Augmented {filename} -> {new_filename}")
    
    # Modify annotations
    new_img_info = img_info.copy()
    new_img_info["file_name"] = new_filename
    new_img_info["annotations"] = []
    
    for ann in img_info["annotations"]:
        segmentation = ann["segmentation"]
        
        # Ensure segmentation is a valid list of coordinate pairs
        if isinstance(segmentation, list) and all(isinstance(pt, list) and len(pt) == 2 for pt in segmentation):
            orig_poly = Polygon(segmentation)
            if orig_poly.is_valid:
                transformed_poly = affine_transform(orig_poly, [1, 0, 0, 1, 0, 0])
                new_img_info["annotations"].append({
                    "class": ann["class"],
                    "segmentation": list(transformed_poly.exterior.coords)
                })
    
    new_annotations["images"].append(new_img_info)

# Save new annotations
with open(data_dir / "aug_train_annotations.json", "w") as f:
    json.dump(new_annotations, f, indent=4)
