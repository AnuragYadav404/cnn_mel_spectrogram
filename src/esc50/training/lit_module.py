from __future__ import annotations

from time import time

import lightning
import torch

from esc50.utils.metrics import as_class_ids


class LitTrainer(lightning.LightningModule):
    def __init__(
        self,
        model,
        forward,
        optimizer,
        scheduler,
        scheduler_params,
        batch_key,
        metric_input_key,
        metric_output_key,
        val_metrics,
        train_metrics=None,
    ):
        super().__init__()

        self.model = model
        self._forward = forward
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._scheduler_params = scheduler_params
        self._batch_key = batch_key
        self._metric_input_key = metric_input_key
        self._metric_output_key = metric_output_key
        self._val_metrics = val_metrics
        self._train_metrics = train_metrics

    def _aggregate_outputs(self, losses, inputs, outputs):
        united = losses.copy()
        united.update({"input_" + k: v for k, v in inputs.items()})
        united.update({"output_" + k: v for k, v in outputs.items()})
        return united

    def training_step(self, batch):
        start_time = time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time() - start_time

        if self._train_metrics is not None:
            self._train_metrics.update(
                as_class_ids(outputs[self._metric_output_key]),
                as_class_ids(inputs[self._metric_input_key]),
            )

        for key, value in losses.items():
            self.log(f"train_{key}", value, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=inputs[self._batch_key].shape[0], sync_dist=True)
            self.log(f"train_avg_{key}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=inputs[self._batch_key].shape[0], sync_dist=True)

        self.log("train_model_time", model_time, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=1, sync_dist=True)
        self.log("train_avg_model_time", model_time, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1, sync_dist=True)
        return self._aggregate_outputs(losses, inputs, outputs)

    def validation_step(self, batch, batch_idx):
        start_time = time()
        losses, inputs, outputs = self._forward(self, batch, epoch=self.current_epoch)
        model_time = time() - start_time

        if self._val_metrics is not None:
            self._val_metrics.update(
                as_class_ids(outputs[self._metric_output_key]),
                as_class_ids(inputs[self._metric_input_key]),
            )

        for key, value in losses.items():
            self.log(f"valid_{key}", value, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=inputs[self._batch_key].shape[0], sync_dist=True)
            self.log(f"valid_avg_{key}", value, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=inputs[self._batch_key].shape[0], sync_dist=True)

        self.log("valid_model_time", model_time, on_step=True, on_epoch=False, prog_bar=True, logger=True, batch_size=1, sync_dist=True)
        self.log("valid_avg_model_time", model_time, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=1, sync_dist=True)
        return self._aggregate_outputs(losses, inputs, outputs)

    def on_train_epoch_end(self):
        if self._train_metrics is None:
            return
        metric_values = self._train_metrics.compute()
        self.log_dict({"train_" + key: value for key, value in metric_values.items()}, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self._train_metrics.reset()

    def on_validation_epoch_end(self):
        if self._val_metrics is None:
            return
        metric_values = self._val_metrics.compute()
        self.log_dict({"valid_" + key: value for key, value in metric_values.items()}, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self._val_metrics.reset()

    def configure_optimizers(self):
        scheduler = {"scheduler": self._scheduler}
        scheduler.update(self._scheduler_params)
        return [self._optimizer], [scheduler]
