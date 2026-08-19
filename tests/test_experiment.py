import pytest
import torch

import numpy as np
from PIL import Image

from activation_doom.data.common import dumps_json, read_jsonl, should_accept, split_episodes, validate_records, visual_difference
from activation_doom.experiment import activation_frame, image_uint8, loss_space_uint8, resize_gray, synthetic_target
from activation_doom.preprocess import save_target, target_gray


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


def test_preprocessing_is_deterministic(tmp_path):
    image = np.arange(8 * 4 * 3, dtype=np.uint8).reshape(4, 8, 3)
    first = target_gray(image)
    second = target_gray(Image.fromarray(image))
    assert np.array_equal(first, second)
    out = tmp_path / "processed.png"
    save_target(out, first)
    assert Image.open(out).mode == "L"
    assert Image.open(out).size == (64, 32)


def test_visual_difference_and_acceptance_logic():
    a = np.zeros((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32) * 0.02
    assert visual_difference(a, b) == pytest.approx(0.02)
    assert should_accept(3, 0.02, 3, 0.015, 12, []) == ["novelty"]
    assert should_accept(12, 0.0, 3, 0.015, 12, []) == ["forced_interval"]


def test_episode_split_has_no_leakage():
    split = split_episodes([0, 0, 1, 2, 3, 4], seed=42)
    assert set(split) == {0, 1, 2, 3, 4}
    assert all(value in {"train", "val", "test"} for value in split.values())
    assert {"train", "val", "test"} <= set(split.values())


def test_metadata_jsonl_roundtrip(tmp_path):
    path = tmp_path / "metadata.jsonl"
    path.write_text(dumps_json({"sample_id": "x", "episode_id": 1}) + "\n", encoding="utf-8")
    assert read_jsonl(path) == [{"sample_id": "x", "episode_id": 1}]


def test_validation_detects_split_leakage(tmp_path):
    original = tmp_path / "o.png"
    processed = tmp_path / "p.png"
    rgb = np.zeros((4, 8, 3), dtype=np.uint8)
    Image.fromarray(rgb, mode="RGB").save(original)
    save_target(processed, target_gray(rgb))
    records = [
        {"sample_id": "a", "episode_id": 1, "dataset_split": "train", "source_image_path": "o.png", "processed_image_path": "p.png"},
        {"sample_id": "b", "episode_id": 1, "dataset_split": "val", "source_image_path": "o.png", "processed_image_path": "p.png"},
    ]
    report = validate_records(tmp_path, records)
    assert not report["ok"]
    assert any("episodes in multiple splits" in error for error in report["errors"])
