import torch
import torch.nn as nn


class EnsembleModel(nn.Module):
    def __init__(self, models) -> None:
        super().__init__()

        self.models = nn.ModuleList(models)

    def forward(self, x):
        logits = []
        for model in self.models:
            with torch.no_grad():
                logits.append(model(x))

        stacked_logits = torch.stack(logits)
        return torch.mean(stacked_logits, dim=0)
