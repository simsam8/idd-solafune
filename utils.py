import os
from pathlib import Path

from ensemble import EnsembleModel
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
    model = Model.load_from_checkpoint(model_path / "best_f1.ckpt")
    return model


def load_models(models_path):
    names = []
    models = []
    models_rgb = []
    models_full = []
    for model_name in os.listdir(models_path):
        model = load_model(models_path / model_name)
        names.append(model_name)
        models.append(model)

        if model_name.endswith("rgb"):
            models_rgb.append(model)
        else:
            models_full.append(model)

    # Create ensemble of all models models
    names.append("ensemble_rgb")
    models.append(EnsembleModel(models_rgb))

    names.append("ensemble_full")
    models.append(EnsembleModel(models_full))
    return names, models


if __name__ == "__main__":
    pass
