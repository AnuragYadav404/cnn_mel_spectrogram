from __future__ import annotations

import torch
from tqdm import tqdm


def create_inference_model(nn_model_class: torch.nn.Module, nn_model_config: dict, checkpoint_path: str, device: str = "cuda"):
    nn_model_initialized = nn_model_class(**nn_model_config, device=device)
    nn_model_initialized.eval()
    loaded_checkpoint = torch.load(checkpoint_path, map_location="cpu")
    nn_model_initialized.load_state_dict({k.replace("model.", ""): v for k, v in loaded_checkpoint["state_dict"].items()})
    return nn_model_initialized


@torch.inference_mode()
def run_inference_on_df(input_df, nn_model, dataset_class, dataset_config, dataloader_config, device: str = "cuda"):
    inf_dataset = dataset_class(input_df=input_df, **dataset_config)
    inf_loader = torch.utils.data.DataLoader(inf_dataset, **dataloader_config)

    targets, preds = [], []
    for batch_au, batch_tgt in tqdm(inf_loader):
        preds.append(nn_model(batch_au.to(device))["logits"].detach().cpu())
        targets.append(batch_tgt.argmax(dim=-1) if batch_tgt.ndim > 1 else batch_tgt)

    return torch.cat(preds), torch.cat(targets)
