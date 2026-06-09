from __future__ import annotations

import torch

from esc50.utils.metrics import as_class_ids, build_multiclass_metrics


def evaluate_predictions(logits: torch.Tensor, targets: torch.Tensor, num_classes: int):
    metrics = build_multiclass_metrics(num_classes)
    metrics.update(as_class_ids(logits), as_class_ids(targets))
    return metrics.compute()
