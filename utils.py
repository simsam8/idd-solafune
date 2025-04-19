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
    ensemble1_rgb = []
    ensemble1_full = []
    ensemble2_rgb = []
    ensemble2_full = []
    for model_name in os.listdir(models_path):
        model = load_model(models_path / model_name)
        names.append(model_name)
        models.append(model)

        if model_name.endswith("rgb"):
            ensemble1_rgb.append(model)
            if not model_name.startswith("transunet"):
                ensemble2_rgb.append(model)
        else:
            ensemble1_full.append(model)
            if not model_name.startswith("transunet"):
                ensemble2_full.append(model)

    # Create ensemble of all models models
    names.append("ensemble1_rgb")
    models.append(EnsembleModel(ensemble1_rgb))
    names.append("ensemble1_full")
    models.append(EnsembleModel(ensemble1_full))

    names.append("ensemble2_rgb")
    models.append(EnsembleModel(ensemble2_rgb))
    names.append("ensemble2_full")
    models.append(EnsembleModel(ensemble2_full))
    return names, models


if __name__ == "__main__":
    pass
