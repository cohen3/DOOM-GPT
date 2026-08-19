import pytest
import torch

import numpy as np
from PIL import Image

from activation_doom.data.common import dumps_json, read_jsonl, should_accept, split_episodes, validate_records, visual_difference
from activation_doom.activation import ActivationConfig, target_loss_space
from activation_doom.dashboard import FINAL_RECT, RAW_RECT, compose_dashboard, pixel_from_point, pixel_source, source_token_map
from activation_doom.experiment import activation_frame, image_uint8, loss_space_uint8, resize_gray, synthetic_target
from activation_doom.live import activation_image, describe, key_action, scripted_actions, temporal_row
from activation_doom.preprocess import save_target, target_gray
from activation_doom.presentation import compose_presentation
from activation_doom.renderer import ImageEncoder, fixed_indices, load_encoder_checkpoint, random_prompts


def test_activation_frame_takes_first_values():
    hidden = torch.arange(12).reshape(1, 3, 4)
    assert activation_frame(hidden, 3, 2)[0].tolist() == [[0, 1, 2], [3, 4, 5]]


def test_activation_frame_keeps_batches_isolated():
    hidden = torch.arange(24).reshape(2, 3, 4)
    assert activation_frame(hidden, 3, 2).tolist() == [
        [[0, 1, 2], [3, 4, 5]],
        [[12, 13, 14], [15, 16, 17]],
    ]


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


def test_target_loss_space_is_fixed():
    assert target_loss_space(torch.tensor([0.0, 0.5, 1.0])).tolist() == [-1.0, 0.0, 1.0]


def test_encoder_shape_and_default_parameter_count():
    encoder = ImageEncoder(torch.zeros(4, 768))
    assert encoder(torch.zeros(2, 1, 32, 64)).shape == (2, 4, 768)
    assert sum(parameter.numel() for parameter in encoder.parameters()) == 1_337_344


def test_encoder_checkpoint_roundtrip_is_frozen(tmp_path):
    from activation_doom.activation import ActivationConfig

    encoder = ImageEncoder(torch.zeros(4, 768))
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "config": {
                "activation": ActivationConfig(activation_mean=1.0, activation_std=2.0).to_dict(),
                "encoder": {"channels": [16, 32, 64], "feature_dim": 256},
            },
        },
        path,
    )
    loaded, config, _ = load_encoder_checkpoint(path, torch.device("cpu"))
    assert config.activation_mean == 1.0
    assert not loaded.training
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_random_prompts_are_deterministic():
    from activation_doom.activation import ActivationConfig

    config = ActivationConfig()
    assert torch.equal(random_prompts(config, 0.0, 1.0, 2, 7), random_prompts(config, 0.0, 1.0, 2, 7))


def test_fixed_indices_spread_episodes():
    records = [{"episode_id": episode} for episode in [1, 1, 2, 2, 3, 3]]
    chosen = fixed_indices(records, 3, 7)
    assert {records[index]["episode_id"] for index in chosen} == {1, 2, 3}


def test_live_keys_support_simultaneous_controls():
    buttons = ["MOVE_FORWARD", "MOVE_BACKWARD", "MOVE_LEFT", "TURN_RIGHT", "ATTACK"]
    assert key_action(buttons, {"w", "a", "right", "space"}) == [1.0, 0.0, 1.0, 1.0, 1.0]


def test_fixed_live_trajectory_is_315_tics():
    actions = scripted_actions()
    assert len(actions) == 315
    assert {label for label, _ in actions} >= {"idle", "forward", "moving_turn", "forward_attack", "strafe_attack", "turn"}


def test_temporal_metrics_compare_frame_deltas():
    previous = {
        "target_loss": np.zeros((2, 2), dtype=np.float32),
        "prediction": np.zeros((2, 2), dtype=np.float32),
        "prompt": np.zeros((4, 3), dtype=np.float32),
    }
    result = {
        "target_loss": np.zeros((2, 2), dtype=np.float32),
        "prediction": np.ones((2, 2), dtype=np.float32),
        "prompt": np.ones((4, 3), dtype=np.float32),
        "timing": {},
    }
    row = temporal_row(result, previous)
    assert row["spatial_mse"] == 1.0
    assert row["target_change"] == 0.0
    assert row["prediction_change"] == 1.0
    assert row["temporal_error"] == 1.0
    assert row["prompt_l2_distance"] == pytest.approx(np.sqrt(12))


def test_live_display_uses_fixed_loss_space_scale():
    image = activation_image(np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32))
    assert np.asarray(image).tolist() == [[0, 127, 255]]


def test_dashboard_pixel_provenance_boundaries():
    assert [(pixel_source(index % 64, index // 64)["token"], pixel_source(index % 64, index // 64)["dimension"]) for index in [0, 767, 768, 1535, 1536, 2047]] == [
        (0, 0), (0, 767), (1, 0), (1, 767), (2, 0), (2, 511)
    ]
    tokens = source_token_map()
    assert [(tokens == token).sum() for token in range(4)] == [768, 768, 512, 0]
    with pytest.raises(ValueError):
        pixel_source(64, 0)


def test_dashboard_hit_testing_covers_both_framebuffers():
    assert pixel_from_point(FINAL_RECT[0], FINAL_RECT[1]) == (0, 0)
    assert pixel_from_point(RAW_RECT[0] + RAW_RECT[2] - 1, RAW_RECT[1] + RAW_RECT[3] - 1) == (63, 31)
    assert pixel_from_point(0, 0) is None


def test_dashboard_composes_the_exact_raw_activation_slice():
    hidden = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
    raw = hidden.reshape(-1)[:2048].reshape(32, 64)
    config = ActivationConfig(activation_mean=0.0, activation_std=1.0)
    dashboard = compose_dashboard(
        np.zeros((240, 320, 3), dtype=np.uint8),
        {"target": np.zeros((32, 64), dtype=np.float32), "hidden": hidden, "raw": raw, "prediction": raw},
        {"sequence": 7, "game_tick": 9},
        config,
    )
    assert dashboard.size == (1600, 900)
    assert raw[31, 63] == hidden[2, 511]


def test_presentation_composes_a_16_by_9_activation_flow():
    hidden = np.arange(4 * 768, dtype=np.float32).reshape(4, 768)
    raw = hidden.reshape(-1)[:2048].reshape(32, 64)
    result = {
        "target": np.zeros((32, 64), dtype=np.float32),
        "hidden": hidden,
        "raw": raw,
        "prediction": raw,
        "prompt": np.zeros((4, 768), dtype=np.float32),
    }
    image = compose_presentation(
        np.zeros((240, 320, 3), dtype=np.uint8),
        result,
        {"sequence": 1, "game_tick": 2},
        ActivationConfig(activation_mean=0.0, activation_std=1.0),
    )
    assert image.size == (1920, 1080)


def test_latency_description_has_requested_percentiles():
    stats = describe([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == 2.5
    assert set(stats) == {"mean", "p50", "p95", "p99"}


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
