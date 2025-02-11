import json
import os
import shutil
from pathlib import Path

from ray.train import Checkpoint

from Models import Model


def convert_to_geojson(data):
    """
    Converts a list of dictionaries in the specified format to GeoJSON

    Args:
        data: A list of dictionaries containing 'class' and 'segmentation' keys

    Returns:
        A GeoJSON feature collection
    """
    features = []
    for item in data:
        polygon = []
        for i in range(0, len(item["segmentation"]), 2):
            polygon.append([item["segmentation"][i], item["segmentation"][i + 1]])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [polygon]},
                "properties": {"class": item["class"]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def load_model(model_path):
    model_path = Path(model_path)
    model_name = f"{model_path.name}.json"
    with open(model_path / model_name, "r") as f:
        settings = json.load(f)

    model = Model.load_from_checkpoint(
        model_path / "checkpoint.ckpt", config=settings["config"]
    )
    return model


def save_checkpoint(checkpoint_path, target_path):
    target_path = Path(target_path)
    checkpoint_path = (
        (Path(checkpoint_path.path)).resolve().relative_to(Path("./").resolve())
    )

    result_path = checkpoint_path.parent / "result.json"
    result = {}
    with open(result_path, "r") as f:
        for line in f.readlines():
            run_n = json.loads(line)
            if run_n["checkpoint_dir_name"] == checkpoint_path.name:
                result = run_n
                break

    key_rm = [
        "timestamp",
        "should_checkpoint",
        "done",
        "training_iteration",
        "time_this_iter_s",
        "pid",
        "hostname",
        "node_ip",
        "time_since_restore",
        "iterations_since_restore",
    ]

    for k in key_rm:
        del result[k]

    result["checkpoint_path"] = str(checkpoint_path)

    model_name = f"{result['trial_id']}_valf1_{result['val/f1']*100:.2f}"

    os.mkdir(target_path / model_name)

    shutil.copy2(
        checkpoint_path / "checkpoint.ckpt",
        target_path / model_name,
    )

    with open(
        target_path / model_name / f"{model_name}.json", "w", encoding="utf-8"
    ) as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":

    # Testing purposes
    checkpoint_path = "./logs/tune_idd_asha/train_idd_tune_7c201_00000_0_batch_size=2,lr=0.0083,num_workers=10,weight_decay=0.0084_2025-02-11_13-55-35/checkpoint_000001/"
    cp = Checkpoint.from_directory(checkpoint_path)

    save_checkpoint(cp, "./models")
