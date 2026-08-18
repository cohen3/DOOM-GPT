import pytest
import torch

import numpy as np

from activation_doom.experiment import activation_frame, image_uint8, loss_space_uint8, resize_gray, synthetic_target


def test_activation_frame_takes_first_values():
    hidden = torch.arange(12).reshape(1, 3, 4)
    assert activation_frame(hidden, 3, 2).tolist() == [[0, 1, 2], [3, 4, 5]]


def test_activation_frame_rejects_small_tensor():
    with pytest.raises(ValueError):
        activation_frame(torch.arange(5), 3, 2)


def test_image_uint8_constant_is_black():
    out = image_uint8(torch.ones(2, 2))
    assert out.dtype.name == "uint8"
    assert out.max() == 0


def test_synthetic_target_shape_and_range():
    target = synthetic_target(16, 8)
    assert target.shape == (8, 16)
    assert 0.0 <= target.min() <= target.max() <= 1.0


def test_resize_gray_shape_and_range():
    target = resize_gray(np.zeros((4, 8, 3), dtype=np.uint8), 16, 8)
    assert target.shape == (8, 16)
    assert 0.0 <= target.min() <= target.max() <= 1.0


def test_loss_space_uint8_uses_fixed_scale():
    assert loss_space_uint8(torch.tensor([[-1.0, 0.0, 1.0]])).tolist() == [[0, 127, 255]]
