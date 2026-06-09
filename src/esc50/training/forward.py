from __future__ import annotations

from torch import nn


class AudioForward(nn.Module):
    def __init__(self, loss_function, output_key: str = "logits", input_key: str = "targets"):
        super().__init__()
        self.loss_function = loss_function
        self.output_key = output_key
        self.input_key = input_key

    def forward(self, runner, batch, epoch=None):
        aus, targets = batch
        output = runner.model(aus)
        output["predictions"] = output["logits"]
        inputs = {"aus": aus, "targets": targets}
        losses = {
            "loss": self.loss_function(
                output[self.output_key],
                inputs[self.input_key],
            )
        }
        return losses, inputs, output
