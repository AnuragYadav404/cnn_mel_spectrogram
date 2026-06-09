import torch

from esc50.utils.metrics import as_class_ids


def test_as_class_ids_converts_soft_labels():
    values = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    ids = as_class_ids(values)

    assert torch.equal(ids, torch.tensor([1, 0]))
