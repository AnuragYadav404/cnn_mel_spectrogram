from __future__ import annotations

import os

import lightning
import pandas as pd
from hydra.utils import instantiate, to_absolute_path
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import torch

from esc50.data.metadata import enrich_df_with_audiometa, enrich_df_with_filepaths
from esc50.data.split import assign_binary_split, assign_folds
from esc50.models import SpecCNNClassifier
from esc50.training.lit_module import LitTrainer
from esc50.utils.seed import seed_everything


def _resolve_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_experiment(cfg):
    seed_everything(int(cfg.seed))
    device = _resolve_device()

    csv_path = to_absolute_path(cfg.data.csv_path)
    data_root = to_absolute_path(cfg.data.data_root)
    exp_name = to_absolute_path(cfg.train.exp_name)

    meta_df = pd.read_csv(csv_path)
    class_name2idx = meta_df[["category", "target"]].drop_duplicates("target").set_index("category")["target"].to_dict()
    class_idx2name = {value: key for key, value in class_name2idx.items()}

    meta_df = meta_df.drop(columns=["fold"])
    meta_df = assign_binary_split(meta_df)
    train_df = meta_df[meta_df["is_train"]].reset_index(drop=True)
    test_df = meta_df[~meta_df["is_train"]].reset_index(drop=True)
    train_df = assign_folds(train_df)

    fold_id = int(cfg.train.fold_id)
    fold_df = train_df
    train_df = fold_df[fold_df["fold"] != fold_id].reset_index(drop=True)
    val_df = fold_df[fold_df["fold"] == fold_id].reset_index(drop=True)

    train_df = enrich_df_with_filepaths(train_df, data_root)
    val_df = enrich_df_with_filepaths(val_df, data_root)
    train_df = enrich_df_with_audiometa(train_df)
    val_df = enrich_df_with_audiometa(val_df)

    train_dataset = instantiate(cfg.data.train_dataset, input_df=train_df, num_classes=len(class_name2idx))
    val_dataset = instantiate(cfg.data.val_dataset, input_df=val_df, num_classes=len(class_name2idx))
    train_loader = torch.utils.data.DataLoader(train_dataset, **cfg.data.train_dataloader)
    val_loader = torch.utils.data.DataLoader(val_dataset, **cfg.data.val_dataloader)

    pool = instantiate(cfg.pooling)
    model = instantiate(cfg.model, device=device, pool=pool)
    optimizer = instantiate(cfg.optim, params=model.parameters())
    scheduler = instantiate(cfg.sched, optimizer=optimizer, T_max=len(train_loader) * int(cfg.train.max_epochs))
    forward = instantiate(cfg.train.forward)
    val_metrics = instantiate(cfg.metrics, num_classes=int(cfg.data.num_classes))
    train_metrics = instantiate(cfg.metrics, num_classes=int(cfg.data.num_classes)) if cfg.train.get("use_train_metrics", True) else None

    lightning_model = LitTrainer(
        model=model,
        forward=forward,
        optimizer=optimizer,
        scheduler=scheduler,
        scheduler_params=dict(cfg.train.scheduler_params),
        batch_key=cfg.train.batch_key,
        metric_input_key=cfg.train.metric_input_key,
        metric_output_key=cfg.train.metric_output_key,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(exp_name, "checkpoints"),
        save_top_k=int(cfg.train.n_checkpoints_to_save),
        mode=cfg.train.metric_mode,
        monitor=cfg.train.main_metric,
        save_last=True,
        auto_insert_metric_name=True,
        save_weights_only=True,
        save_on_train_epoch_end=True,
        filename="{epoch}-{step}-{valid_MulticlassF1Score:.3f}",
    )
    callbacks = [checkpoint_callback, LearningRateMonitor(logging_interval="step")]

    wandb_logger = pl_loggers.WandbLogger(save_dir=exp_name, name=exp_name, **dict(cfg.train.wandb_logger))
    trainer = lightning.Trainer(
        devices=-1,
        precision=cfg.train.precision_mode,
        strategy=cfg.train.train_strategy,
        max_epochs=int(cfg.train.max_epochs),
        logger=wandb_logger,
        log_every_n_steps=int(cfg.train.log_every_n_steps),
        callbacks=callbacks,
    )
    trainer.fit(model=lightning_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    wandb_logger.experiment.finish()
    return checkpoint_callback.best_model_path
