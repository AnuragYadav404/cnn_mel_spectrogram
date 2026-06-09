from __future__ import annotations

import torch
import torchmetrics


def as_class_ids(values: torch.Tensor) -> torch.Tensor:
    if values.ndim > 1:
        return torch.argmax(values, dim=-1)
    return values


def build_multiclass_metrics(num_classes: int) -> torchmetrics.MetricCollection:
    return torchmetrics.MetricCollection(
        metrics=[
            torchmetrics.Accuracy(average="macro", task="multiclass", num_classes=num_classes),
            torchmetrics.Recall(average="macro", task="multiclass", num_classes=num_classes),
            torchmetrics.Precision(average="macro", task="multiclass", num_classes=num_classes),
            torchmetrics.F1Score(average="macro", task="multiclass", num_classes=num_classes),
        ],
        compute_groups=False,
    )
