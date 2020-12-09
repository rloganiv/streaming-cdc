import unittest

import torch

from meercat import nn_thresh


def test_nn_thresh():
    example = torch.tensor([
        [1, 0],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],
    ])
    expected = torch.tensor([0, 1, 0, 0, 1])
    observed = nn_thresh.cluster(example, threshold=0.5)
    assert torch.equal(expected, observed)

