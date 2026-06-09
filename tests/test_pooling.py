import torch

from esc50.models.pooling import GeM


def test_gem_produces_positive_parameters_and_shape():
    layer = GeM(p=3.0)
    x = torch.rand(2, 8, 4, 4)
    y = layer(x)

    assert y.shape == (2, 8, 1, 1)
    assert layer.p.item() > 0
