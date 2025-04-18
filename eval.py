import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
from rasterio import features
from shapely.geometry import shape
from skimage import measure
from tqdm import tqdm

from datasets import TestDataset, TrainValDataset
from utils import load_models


def run_inference(model, loader, pred_output_dir):
    # def run_inference(model, loader):
    pred_output_dir = Path(pred_output_dir)
    pred_output_dir.mkdir(exist_ok=True, parents=True)

    for batch in tqdm(loader):
        img = batch["image"].cuda()

        with torch.no_grad():
            logits_mask = model(img)
            prob_mask = logits_mask.sigmoid()

        # save prob mask as numpy array
        for i in range(img.size(0)):
            file_name = os.path.basename(batch["image_path"][i])
            prob_mask_i = prob_mask[i].cpu().numpy()  # (4, 1024, 1024)

            np.save(
                pred_output_dir / file_name.replace(".tif", ".npy"),
                prob_mask_i.astype(np.float16),
            )


def compute_f1_score(pred_mask, truth_mask):
    # `pred_mask` is a binary numpy array of shape (H, W) = (1024, 1024)
    # `truth_mask` is a binaru numpy array of shape (H, W) = (1024, 1024)
    assert pred_mask.shape == (1024, 1024), f"{pred_mask.shape=}"
    assert truth_mask.shape == (1024, 1024), f"{truth_mask.shape=}"

    tp = ((pred_mask > 0) & (truth_mask > 0)).sum()
    fp = ((pred_mask > 0) & (truth_mask == 0)).sum()
    fn = ((pred_mask == 0) & (truth_mask > 0)).sum()
    precision = (
        tp / (tp + fp) if tp + fp > 0 else 1
    )  # if no prediction, precision is considered as 1
    recall = (
        tp / (tp + fn) if tp + fn > 0 else 1
    )  # if no ground truth, recall is considered as 1
    f1 = (
        2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    )  # if either precision or recall is 0, f1 is 0

    return f1


def detect_polygons(pred_dir, score_thresh, min_area, class_names):
    pred_dir = Path(pred_dir)
    pred_paths = list(pred_dir.glob("*.npy"))
    pred_paths = sorted(pred_paths)

    polygons_all_imgs = {}
    for pred_path in tqdm(pred_paths):
        polygons_all_classes = {}

        mask = np.load(pred_path)  # (4, 1024, 1024)
        mask = mask > score_thresh  # binarize
        for i, class_name in enumerate(class_names):
            mask_for_a_class = mask[i]
            if mask_for_a_class.sum() < min_area:
                mask_for_a_class = np.zeros_like(
                    mask_for_a_class
                )  # set all to zero if the predicted area is less than `min_area`

            # extract polygons from the binarized mask
            label = measure.label(
                mask_for_a_class, connectivity=2, background=0
            ).astype(np.uint8)
            polygons = []
            for p, value in features.shapes(label, label):
                p = shape(p).buffer(0.5)
                p = p.simplify(tolerance=0.5)
                polygons.append(p)
            polygons_all_classes[class_name] = polygons
        polygons_all_imgs[pred_path.name.replace(".npy", ".tif")] = polygons_all_classes

    return polygons_all_imgs


def run_validation(
    data_root, model, model_name, class_names, score_thresh, min_area, channels
):
    sample_indices = list(range(176))  # train_0.tif to train_175.tif
    _, val_indices = sklearn.model_selection.train_test_split(
        sample_indices, test_size=0.2, random_state=42
    )

    val_data = TrainValDataset(
        data_root, val_indices, augmentations=None, channels=channels
    )

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        num_workers=10,
        shuffle=False,
    )

    val_pred_dir = data_root / "val_preds" / model_name
    run_inference(model, val_loader, val_pred_dir)

    val_f1_scores = {}
    for idx in sorted(val_indices):
        fn = f"train_{idx}"
        # prepare prediction mask
        pred_mask = np.load(val_pred_dir / f"{fn}.npy")  # (4, 1024, 1024)
        pred_mask = pred_mask > score_thresh  # binarize
        # prepare ground truth mask
        truth_mask = np.load(data_root / "train_masks" / f"{fn}.npy")  # (4, 1024, 1024)
        # compute f1 score for each class
        val_f1_scores[fn] = {}
        for i, class_name in enumerate(class_names):
            pred_for_a_class = pred_mask[i]
            if pred_for_a_class.sum() < min_area:
                pred_for_a_class = np.zeros_like(
                    pred_for_a_class
                )  # set all to zero if the predicted area is less than `min_area`
            val_f1_scores[fn][class_name] = compute_f1_score(
                pred_for_a_class, truth_mask[i]
            )
    val_f1_scores = pd.DataFrame(val_f1_scores).T

    # add a column for average of all the 4 classes
    val_f1_scores["all_classes"] = val_f1_scores.mean(axis=1)
    # add a row for average of all the val images
    val_f1_scores.loc["all_images"] = val_f1_scores.mean()

    final_val_f1 = val_f1_scores.loc["all_images", "all_classes"]
    print(f"val f1 score: {final_val_f1:.4f}")
    return final_val_f1


def run_test(data_root, model, class_names, score_thresh, min_area):
    test_loader = torch.utils.data.DataLoader(
        TestDataset(data_root),
        batch_size=1,
        num_workers=10,
        shuffle=False,
    )

    test_pred_dir = data_root / "test_preds"
    run_inference(model, test_loader, test_pred_dir)

    test_pred_polygons = detect_polygons(
        test_pred_dir,
        score_thresh=score_thresh,
        min_area=min_area,
        class_names=class_names,
    )

    submission_save_path = data_root / "submission.json"

    images = []
    for img_id in range(118):  # evaluation_0.tif to evaluation_117.tif
        annotations = []
        for class_name in class_names:
            for poly in test_pred_polygons[f"evaluation_{img_id}.tif"][class_name]:
                seg: list[float] = []  # [x0, y0, x1, y1, ..., xN, yN]
                for xy in poly.exterior.coords:
                    seg.extend(xy)

                annotations.append({"class": class_name, "segmentation": seg})

        images.append(
            {"file_name": f"evaluation_{img_id}.tif", "annotations": annotations}
        )

    with open(submission_save_path, "w", encoding="utf-8") as f:
        json.dump({"images": images}, f, indent=4)


def run_selection(data_root, names, models, class_names, threshold=0.5, min_area=10000):

    f1_score = []
    for model_name, model in zip(names, models):
        print(model_name)
        model.eval()

        channels = "rgb" if model_name.endswith("rgb") else "full"

        with torch.no_grad():
            score = run_validation(
                data_root, model, model_name, class_names, threshold, min_area, channels
            )
            f1_score.append(score)

    best_idx = np.argmax(f1_score)

    print(f"Best model: {names[best_idx]}")
    print(f"F1 score: {f1_score[best_idx]}")
    return models[best_idx]


def main():
    data_root = Path("./data")
    models_path = Path("./data/training_result/")
    class_names = ["grassland_shrubland", "logging", "mining", "plantation"]

    model_names, models_list = load_models(models_path)

    score_thresh = 0.5  # threshold to binarize the prediction mask

    # if the predicted area of a class is less than this,
    # submit a zero mask because small predicted areas are often false positives
    min_area = 10000

    best_model = run_selection(
        data_root,
        model_names,
        models_list,
        class_names,
        threshold=score_thresh,
        min_area=min_area,
    )
    run_test(data_root, best_model, class_names, score_thresh, min_area)


if __name__ == "__main__":
    main()
