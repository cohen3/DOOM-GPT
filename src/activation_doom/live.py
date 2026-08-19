from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from activation_doom.activation import ActivationConfig, activation_frame, prediction_loss_space, target_loss_space
from activation_doom.dashboard import compose_dashboard, pixel_from_point
from activation_doom.data.collect import button_name, make_game
from activation_doom.data.common import read_jsonl
from activation_doom.experiment import load_frozen, loss_space_uint8, parameter_hash, write_json
from activation_doom.preprocess import save_target, target_gray
from activation_doom.renderer import load_encoder_checkpoint, render_prompts


DEFAULT_CHECKPOINT = Path("experiments/m5_amortized_renderer/20260819-101654/full/best.pt")
EXPECTED_CHECKPOINT_HASH = "c76df8c03ce2ca96cbbb65821da8b727f3bc23aa386877463becea7ffad96edb"
EXPECTED_MODEL_HASH = "f87fb1b59f61d6f9c7b008f4a6933f023765d9706dd8c85ea31dcfe8243a96d2"
DISPLAY_SIZE = (640, 480)
TIC_RATE = 35.0


def file_hash(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def output_dir(out: str | None) -> Path:
    """Create a requested or timestamped V1-C artifact directory."""
    path = Path(out) if out else Path("experiments") / "m6_live" / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_device(name: str) -> torch.device:
    """Resolve auto mode to CUDA when available and CPU otherwise."""
    return torch.device("cuda" if name == "auto" and torch.cuda.is_available() else "cpu" if name == "auto" else name)


def load_runtime(checkpoint_path: Path, device: torch.device):
    """Load and verify the exact frozen V1-B encoder and transformer runtime."""
    checkpoint_sha = file_hash(checkpoint_path)
    if checkpoint_sha != EXPECTED_CHECKPOINT_HASH:
        raise RuntimeError(f"checkpoint SHA-256 mismatch: {checkpoint_sha}")
    encoder, config, checkpoint = load_encoder_checkpoint(checkpoint_path, device)
    expected = {
        "model": "distilbert/distilgpt2",
        "hidden_state_index": 3,
        "prompt_tokens": 4,
        "hidden_size": 768,
        "width": 64,
        "height": 32,
        "activation_mean": 1.004219851056348,
        "activation_std": 38.36865501756245,
    }
    if config.to_dict() != expected:
        raise RuntimeError(f"checkpoint activation configuration changed: {config.to_dict()}")
    _, model = load_frozen(config.model, device)
    model_sha = parameter_hash(model)
    if model_sha != EXPECTED_MODEL_HASH:
        raise RuntimeError(f"transformer SHA-256 mismatch: {model_sha}")
    if any(parameter.requires_grad for parameter in [*encoder.parameters(), *model.parameters()]):
        raise RuntimeError("live runtime contains trainable parameters")
    metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha,
        "transformer_sha256": model_sha,
        "encoder_sha256": parameter_hash(encoder),
        "activation": config.to_dict(),
        "encoder_parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "transformer_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "optimizer_present_in_checkpoint_but_unused": "optimizer_state_dict" in checkpoint,
    }
    return encoder, model, config, metadata


def key_action(buttons: list[str], pressed: set[str]) -> list[float]:
    """Convert currently held keys into one simultaneous ViZDoom action vector."""
    values = [0.0] * len(buttons)
    mapping = {
        "w": "MOVE_FORWARD",
        "s": "MOVE_BACKWARD",
        "a": "MOVE_LEFT",
        "d": "MOVE_RIGHT",
        "left": "TURN_LEFT",
        "right": "TURN_RIGHT",
        "space": "ATTACK",
        "shift_l": "SPEED",
        "shift_r": "SPEED",
    }
    for key, button in mapping.items():
        if key in pressed and button in buttons:
            values[buttons.index(button)] = 1.0
    return values


def named_action(buttons: list[str], names: set[str]) -> list[float]:
    """Build an action vector from explicit ViZDoom button names."""
    return [1.0 if button in names else 0.0 for button in buttons]


def scripted_actions() -> list[tuple[str, set[str]]]:
    """Return the fixed 315-tic movement, turning, and firing trajectory."""
    segments = [
        ("idle", set(), 35),
        ("forward", {"MOVE_FORWARD"}, 70),
        ("moving_turn", {"MOVE_FORWARD", "TURN_RIGHT"}, 35),
        ("forward_attack", {"MOVE_FORWARD", "ATTACK"}, 70),
        ("strafe_attack", {"MOVE_LEFT", "ATTACK"}, 35),
        ("turn", {"TURN_LEFT"}, 35),
        ("forward_attack", {"MOVE_FORWARD", "ATTACK"}, 35),
    ]
    return [(label, names) for label, names, count in segments for _ in range(count)]


def render_frame(original: np.ndarray, encoder, model, config: ActivationConfig, device: torch.device) -> dict:
    """Render one RGB ViZDoom frame and return arrays, prompt diagnostics, and stage timings."""
    preprocess_started = time.perf_counter()
    target = target_gray(original)
    cpu_image = torch.from_numpy(target.copy())[None, None].float()
    preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0
    with torch.inference_mode():
        if device.type == "cuda":
            events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
            events[0].record()
            image = cpu_image.to(device)
            events[1].record()
            prompt = encoder(image)
            events[2].record()
            outputs = model(
                inputs_embeds=prompt,
                attention_mask=torch.ones(prompt.shape[:2], device=device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            events[3].record()
            hidden = outputs.hidden_states[config.hidden_state_index]
            raw = activation_frame(hidden, config.width, config.height)
            prediction = prediction_loss_space(raw, config.activation_mean, config.activation_std)
            events[4].record()
            hidden_cpu = hidden[0].float().cpu().numpy().copy()
            raw_cpu = raw[0].float().cpu().numpy().copy()
            prediction_cpu = prediction[0].float().cpu().numpy().copy()
            prompt_cpu = prompt[0].float().cpu().numpy().copy()
            events[5].record()
            events[5].synchronize()
            gpu_times = [events[index].elapsed_time(events[index + 1]) for index in range(5)]
        else:
            started = time.perf_counter()
            image = cpu_image.to(device)
            h2d = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            prompt = encoder(image)
            encoder_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            outputs = model(
                inputs_embeds=prompt,
                attention_mask=torch.ones(prompt.shape[:2], device=device),
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )
            transformer_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            hidden = outputs.hidden_states[config.hidden_state_index]
            raw = activation_frame(hidden, config.width, config.height)
            prediction = prediction_loss_space(raw, config.activation_mean, config.activation_std)
            extraction_ms = (time.perf_counter() - started) * 1000.0
            started = time.perf_counter()
            hidden_cpu = hidden[0].float().cpu().numpy().copy()
            raw_cpu = raw[0].float().cpu().numpy().copy()
            prediction_cpu = prediction[0].float().cpu().numpy().copy()
            prompt_cpu = prompt[0].float().cpu().numpy().copy()
            d2h = (time.perf_counter() - started) * 1000.0
            gpu_times = [h2d, encoder_ms, transformer_ms, extraction_ms, d2h]
    timing = {
        "preprocessing_ms": preprocess_ms,
        "host_to_device_ms": gpu_times[0],
        "encoder_ms": gpu_times[1],
        "transformer_ms": gpu_times[2],
        "activation_conversion_ms": gpu_times[3],
        "device_to_host_ms": gpu_times[4],
    }
    timing["renderer_only_ms"] = sum(timing.values())
    return {
        "target": target,
        "target_loss": target * 2.0 - 1.0,
        "hidden": hidden_cpu,
        "raw": raw_cpu,
        "prediction": prediction_cpu,
        "prompt": prompt_cpu,
        "timing": timing,
    }


def temporal_row(result: dict, previous: dict | None) -> dict:
    """Calculate spatial, temporal, and prompt-motion diagnostics for one frame."""
    prediction = result["prediction"]
    target = result["target_loss"]
    prompt = result["prompt"]
    total_norm = float(np.linalg.norm(prompt))
    row = {
        "spatial_mse": float(np.mean((prediction - target) ** 2)),
        "prompt_norm": total_norm,
        "per_token_prompt_norm": [float(np.linalg.norm(token)) for token in prompt],
        "target_change": None,
        "prediction_change": None,
        "temporal_error": None,
        "prompt_norm_delta": None,
        "prompt_norm_change": None,
        "prompt_l2_distance": None,
        **result["timing"],
    }
    if previous is not None:
        target_delta = target - previous["target_loss"]
        prediction_delta = prediction - previous["prediction"]
        previous_norm = float(np.linalg.norm(previous["prompt"]))
        row |= {
            "target_change": float(np.mean(np.abs(target_delta))),
            "prediction_change": float(np.mean(np.abs(prediction_delta))),
            "temporal_error": float(np.mean(np.abs(prediction_delta - target_delta))),
            "prompt_norm_delta": total_norm - previous_norm,
            "prompt_norm_change": abs(total_norm - previous_norm),
            "prompt_l2_distance": float(np.linalg.norm(prompt - previous["prompt"])),
        }
    return row


def activation_image(prediction: np.ndarray, size: tuple[int, int] | None = None) -> Image.Image:
    """Convert loss-space activations to the fixed grayscale display without adaptive normalization."""
    image = Image.fromarray(loss_space_uint8(prediction), mode="L")
    return image.resize(size, Image.Resampling.NEAREST) if size else image


def comparison_image(original: np.ndarray, prediction: np.ndarray) -> Image.Image:
    """Place the raw RGB framebuffer beside the fixed-scale activation display."""
    left = Image.fromarray(original, mode="RGB").resize(DISPLAY_SIZE, Image.Resampling.NEAREST)
    right = activation_image(prediction, DISPLAY_SIZE).convert("RGB")
    image = Image.new("RGB", (DISPLAY_SIZE[0] * 2, DISPLAY_SIZE[1]))
    image.paste(left, (0, 0))
    image.paste(right, (DISPLAY_SIZE[0], 0))
    return image


class Recorder:
    """Write optional per-frame metrics and ordered diagnostic images."""

    def __init__(self, root: Path, every: int = 1, enabled: bool = True):
        """Prepare one recording directory and its image subdirectories."""
        self.root = root
        self.every = max(1, every)
        self.enabled = enabled
        self.rows: list[dict] = []
        for name in ["original", "target", "activation", "comparison"]:
            (root / "frames" / name).mkdir(parents=True, exist_ok=True)
        self.handle = (root / "metrics.jsonl").open("w", encoding="utf-8")

    def write(self, row: dict, original: np.ndarray, result: dict) -> None:
        """Append one metric row and save images at the configured interval."""
        if not self.enabled:
            return
        self.rows.append(row)
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.handle.flush()
        sequence = row["sequence"]
        if sequence % self.every:
            return
        name = f"frame_{sequence:06d}.png"
        Image.fromarray(original, mode="RGB").save(self.root / "frames" / "original" / name)
        save_target(self.root / "frames" / "target" / name, result["target"])
        activation_image(result["prediction"]).save(self.root / "frames" / "activation" / name)
        comparison_image(original, result["prediction"]).save(self.root / "frames" / "comparison" / name)

    def close(self) -> None:
        """Close the machine-readable metrics stream."""
        if not self.handle.closed:
            self.handle.close()


def describe(values: list[float]) -> dict | None:
    """Return mean and requested latency percentiles for finite numeric values."""
    clean = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if not clean.size:
        return None
    return {
        "mean": float(clean.mean()),
        "p50": float(np.percentile(clean, 50)),
        "p95": float(np.percentile(clean, 95)),
        "p99": float(np.percentile(clean, 99)),
    }


def correlation(rows: list[dict], first: str, second: str) -> float | None:
    """Return a Pearson correlation for rows where both requested values exist."""
    pairs = [(row[first], row[second]) for row in rows if row.get(first) is not None and row.get(second) is not None]
    if len(pairs) < 2:
        return None
    x, y = np.asarray(pairs, dtype=np.float64).T
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def summarize(rows: list[dict], wall_seconds: float | None = None) -> dict:
    """Summarize reconstruction, temporal, prompt, timing, and frame-rate measurements."""
    fields = [
        "spatial_mse",
        "target_change",
        "prediction_change",
        "temporal_error",
        "prompt_norm",
        "prompt_norm_change",
        "prompt_l2_distance",
        "simulation_update_ms",
        "framebuffer_acquisition_ms",
        "preprocessing_ms",
        "host_to_device_ms",
        "encoder_ms",
        "transformer_ms",
        "activation_conversion_ms",
        "device_to_host_ms",
        "display_upscale_ms",
        "renderer_only_ms",
        "end_to_end_ms",
    ]
    summary = {field: describe([row.get(field) for row in rows]) for field in fields}
    summary |= {
        "frames": len(rows),
        "prompt_distance_temporal_error_correlation": correlation(rows, "prompt_l2_distance", "temporal_error"),
        "prompt_distance_spatial_mse_correlation": correlation(rows, "prompt_l2_distance", "spatial_mse"),
    }
    if wall_seconds:
        summary["wall_seconds"] = wall_seconds
        summary["true_fps"] = len(rows) / wall_seconds
    renderer = summary.get("renderer_only_ms")
    summary["renderer_only_fps"] = 1000.0 / renderer["mean"] if renderer and renderer["mean"] else None
    return summary


def save_plots(root: Path, rows: list[dict]) -> None:
    """Save temporal-error and prompt-motion plots for one ordered sequence."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [row["sequence"] for row in rows]
    fig, axis = plt.subplots(figsize=(7, 3), dpi=150)
    axis.plot(x, [row["target_change"] for row in rows], label="target change")
    axis.plot(x, [row["prediction_change"] for row in rows], label="prediction change")
    axis.plot(x, [row["temporal_error"] for row in rows], label="temporal error")
    axis.set_xlabel("frame")
    axis.legend()
    axis.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(root / "temporal.png")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(7, 5), dpi=150, sharex=True)
    axes[0].plot(x, [row["prompt_norm"] for row in rows], label="total")
    token_norms = np.asarray([row["per_token_prompt_norm"] for row in rows])
    for token in range(token_norms.shape[1]):
        axes[0].plot(x, token_norms[:, token], label=f"token {token + 1}")
    axes[0].legend(ncol=3)
    axes[1].plot(x, [row["prompt_l2_distance"] for row in rows], label="prompt L2 distance")
    axes[1].plot(x, [row["prompt_norm_change"] for row in rows], label="norm change")
    axes[1].legend()
    axes[1].set_xlabel("frame")
    fig.tight_layout()
    fig.savefig(root / "prompt.png")
    plt.close(fig)


def finish_row(row: dict, total_started: float, simulation_ms: float, acquisition_ms: float, display_ms: float) -> dict:
    """Add simulation, display, and total end-to-end timings to one frame row."""
    row |= {
        "simulation_update_ms": simulation_ms,
        "framebuffer_acquisition_ms": acquisition_ms,
        "display_upscale_ms": display_ms,
        "end_to_end_ms": (time.perf_counter() - total_started) * 1000.0,
    }
    return row


def run_static(root: Path, encoder, model, config: ActivationConfig, device: torch.device, seed: int) -> dict:
    """Verify repeated identical input and fourth-token perturbation produce stable framebuffers."""
    static_root = root / "static"
    static_root.mkdir(exist_ok=True)
    game, _ = make_game("deathmatch.cfg", seed)
    game.new_episode()
    original = game.get_state().screen_buffer.copy()
    game.close()
    results = [render_frame(original, encoder, model, config, device) for _ in range(120)]
    predictions = np.stack([result["prediction"] for result in results])
    baseline = predictions[0]
    max_difference = float(np.max(np.abs(predictions - baseline)))
    max_variance = float(np.max(np.var(predictions.astype(np.float64), axis=0)))
    prompt = torch.from_numpy(results[0]["prompt"])[None].to(device)
    changed = prompt.clone()
    changed[:, 3] += torch.linspace(-100.0, 100.0, config.hidden_size, device=device)
    with torch.inference_mode():
        _, first = render_prompts(model, prompt, config)
        _, second = render_prompts(model, changed, config)
    fourth_difference = float((first - second).abs().max().item())
    activation_image(baseline).save(static_root / "activation.png")
    comparison_image(original, baseline).save(static_root / "comparison.png")
    report = {
        "repetitions": len(results),
        "maximum_output_difference": max_difference,
        "maximum_pixel_variance": max_variance,
        "fourth_token_perturbation": "linear [-100, 100] added to token four only",
        "fourth_token_maximum_framebuffer_difference": fourth_difference,
        "stable": max_difference <= 1e-6 and max_variance <= 1e-12,
        "fourth_token_inert": fourth_difference <= 1e-6,
    }
    write_json(static_root / "summary.json", report)
    if not report["stable"] or not report["fourth_token_inert"]:
        raise RuntimeError(f"static integrity test failed: {report}")
    return report


def run_trajectory(root: Path, encoder, model, config: ActivationConfig, device: torch.device, seed: int) -> dict:
    """Run and record the fixed live ViZDoom action trajectory without real-time throttling."""
    trajectory_root = root / "trajectory"
    recorder = Recorder(trajectory_root)
    game, scenario = make_game("deathmatch.cfg", seed)
    game.new_episode()
    buttons = [button_name(button) for button in game.get_available_buttons()]
    previous = None
    started = time.perf_counter()
    for sequence, (label, names) in enumerate(scripted_actions()):
        total_started = time.perf_counter()
        action = named_action(buttons, names)
        simulation_started = time.perf_counter()
        game.make_action(action, 1)
        simulation_ms = (time.perf_counter() - simulation_started) * 1000.0
        if game.is_episode_finished():
            break
        acquisition_started = time.perf_counter()
        state = game.get_state()
        original = state.screen_buffer.copy()
        acquisition_ms = (time.perf_counter() - acquisition_started) * 1000.0
        result = render_frame(original, encoder, model, config, device)
        display_started = time.perf_counter()
        activation_image(result["prediction"], DISPLAY_SIZE)
        display_ms = (time.perf_counter() - display_started) * 1000.0
        row = temporal_row(result, previous) | {
            "sequence": sequence,
            "game_tick": int(game.get_episode_time()),
            "action_label": label,
            "action_vector": action,
        }
        finish_row(row, total_started, simulation_ms, acquisition_ms, display_ms)
        recorder.write(row, original, result)
        previous = result
    game.close()
    recorder.close()
    wall = time.perf_counter() - started
    summary = summarize(recorder.rows, wall) | {
        "seed": seed,
        "scenario": scenario,
        "requested_tics": len(scripted_actions()),
        "completed_tics": len(recorder.rows),
        "ended_early": len(recorder.rows) < len(scripted_actions()),
    }
    write_json(trajectory_root / "summary.json", summary)
    save_plots(trajectory_root, recorder.rows)
    return summary


def run_close_enemy(root: Path, dataset: Path, encoder, model, config: ActivationConfig, device: torch.device) -> dict:
    """Replay the known held-out episode-31 close-enemy trajectory through the unchanged renderer."""
    replay_root = root / "close_enemy"
    recorder = Recorder(replay_root)
    records = [
        record
        for record in read_jsonl(dataset / "metadata.jsonl")
        if record["episode_id"] == 31 and 400 <= int(record["sample_id"].rsplit("f", 1)[1]) <= 459
    ]
    records.sort(key=lambda record: record["game_tick"])
    previous = None
    previous_tick = None
    for sequence, record in enumerate(records):
        total_started = time.perf_counter()
        acquisition_started = time.perf_counter()
        original = np.asarray(Image.open(dataset / record["source_image_path"]).convert("RGB"))
        acquisition_ms = (time.perf_counter() - acquisition_started) * 1000.0
        result = render_frame(original, encoder, model, config, device)
        display_started = time.perf_counter()
        activation_image(result["prediction"], DISPLAY_SIZE)
        display_ms = (time.perf_counter() - display_started) * 1000.0
        frame_number = int(record["sample_id"].rsplit("f", 1)[1])
        row = temporal_row(result, previous) | {
            "sequence": sequence,
            "sample_id": record["sample_id"],
            "game_tick": record["game_tick"],
            "tick_delta": None if previous_tick is None else record["game_tick"] - previous_tick,
            "action_label": record["action_label"],
            "action_vector": record["action_vector"],
            "close_enemy_subset": 438 <= frame_number <= 459,
        }
        finish_row(row, total_started, 0.0, acquisition_ms, display_ms)
        recorder.write(row, original, result)
        previous = result
        previous_tick = record["game_tick"]
    recorder.close()
    close_rows = [row for row in recorder.rows if row["close_enemy_subset"]]
    summary = {
        "source": "held-out V1-B test episode 31, ordered accepted frames 400..459",
        "full_sequence": summarize(recorder.rows),
        "close_enemy_frames_438_to_459": summarize(close_rows),
    }
    write_json(replay_root / "summary.json", summary)
    save_plots(replay_root, recorder.rows)
    return summary


def run_evaluate(args: argparse.Namespace) -> Path:
    """Run all automated V1-C integrity, temporal, stress, and timing experiments."""
    device = resolve_device(args.device)
    root = output_dir(args.out)
    encoder, model, config, runtime = load_runtime(Path(args.checkpoint), device)
    encoder_before = runtime["encoder_sha256"]
    model_before = runtime["transformer_sha256"]
    write_json(root / "config.json", runtime | {"command": "evaluate", "seed": args.seed, "dataset": args.dataset})
    static = run_static(root, encoder, model, config, device, args.seed)
    trajectory = run_trajectory(root, encoder, model, config, device, args.seed)
    close_enemy = run_close_enemy(root, Path(args.dataset), encoder, model, config, device)
    integrity = {
        "checkpoint_sha256_unchanged": file_hash(Path(args.checkpoint)) == EXPECTED_CHECKPOINT_HASH,
        "encoder_sha256_before": encoder_before,
        "encoder_sha256_after": parameter_hash(encoder),
        "encoder_unchanged": encoder_before == parameter_hash(encoder),
        "transformer_sha256_before": model_before,
        "transformer_sha256_after": parameter_hash(model),
        "transformer_unchanged": model_before == parameter_hash(model),
        "trainable_parameter_count": sum(parameter.numel() for parameter in [*encoder.parameters(), *model.parameters()] if parameter.requires_grad),
        "parameter_gradients_absent": all(parameter.grad is None for parameter in [*encoder.parameters(), *model.parameters()]),
        "iterative_optimization_used": False,
        "decoder_used": False,
    }
    write_json(root / "integrity.json", integrity)
    write_json(root / "automated_summary.json", {"static": static, "trajectory": trajectory, "close_enemy": close_enemy, "integrity": integrity})
    print(json.dumps({"completed": str(root), "static": static, "trajectory": trajectory, "close_enemy": close_enemy, "integrity": integrity}, indent=2), flush=True)
    return root


class LiveApp:
    """Run the activation-only Tkinter display and continuous ViZDoom controls."""

    def __init__(self, args: argparse.Namespace):
        """Load the locked renderer, initialize ViZDoom, and create the display window."""
        import tkinter as tk

        from PIL import ImageTk

        self.tk = tk
        self.ImageTk = ImageTk
        self.args = args
        self.device = resolve_device(args.device)
        self.root_path = output_dir(args.out)
        self.encoder, self.model, self.config, self.runtime = load_runtime(Path(args.checkpoint), self.device)
        self.encoder_before = self.runtime["encoder_sha256"]
        self.model_before = self.runtime["transformer_sha256"]
        self.game, self.scenario = make_game("deathmatch.cfg", args.seed)
        self.game.new_episode()
        self.buttons = [button_name(button) for button in self.game.get_available_buttons()]
        self.window = tk.Tk()
        self.window.title("ActivationDoom V1-C")
        self.status = tk.Label(self.window, anchor="w", font=("Consolas", 10))
        self.status.pack(fill="x")
        self.image_label = tk.Label(self.window, borderwidth=0)
        self.image_label.pack()
        self.pressed: set[str] = set()
        self.debug = args.debug
        self.dashboard = args.dashboard
        self.dashboard_options = {"original": True, "hidden": True, "provenance": True, "auto": False, "live": True}
        self.selected = (0, 0)
        self.recorder = Recorder(self.root_path / "interactive", args.record_every, args.record)
        self.previous = None
        self.photo = None
        self.last_frame = None
        self.sequence = 0
        self.episode = 0
        self.started = time.perf_counter()
        self.next_deadline = self.started
        self.closed = False
        self.window.bind("<KeyPress>", self.on_press)
        self.window.bind("<KeyRelease>", self.on_release)
        self.image_label.bind("<Button-1>", self.on_click)
        self.window.protocol("WM_DELETE_WINDOW", self.close)
        self.image_label.focus_set()
        write_json(self.root_path / "config.json", self.runtime | {"command": "play", "seed": args.seed, "scenario": self.scenario})

    def on_press(self, event) -> None:
        """Track held controls and handle mode, recording, and quit commands."""
        key = event.keysym.lower()
        first_press = key not in self.pressed
        self.pressed.add(key)
        if not first_press:
            return
        if key == "c":
            self.debug = not self.debug
        elif key == "tab":
            self.dashboard = not self.dashboard
        elif key == "r":
            self.recorder.enabled = not self.recorder.enabled
        elif key == "f12":
            self.save_dashboard()
        elif key == "escape":
            self.close()

    def on_release(self, event) -> None:
        """Remove a released key from the continuous action state."""
        self.pressed.discard(event.keysym.lower())

    def on_click(self, event) -> None:
        """Select one displayed activation pixel while the dashboard is visible."""
        if self.dashboard:
            selected = pixel_from_point(event.x, event.y, self.config.width, self.config.height)
            if selected is not None:
                self.selected = selected
        self.image_label.focus_set()

    def save_dashboard(self) -> Path | None:
        """Save the most recently rendered live frame as a full dashboard screenshot."""
        if self.last_frame is None:
            return None
        original, result, row = self.last_frame
        image = compose_dashboard(
            original,
            result,
            row,
            self.config,
            self.selected,
            self.dashboard_options,
            "LIVE | exact hidden-state provenance | transformer and encoder frozen",
        )
        path = self.root_path / "interactive" / "dashboard" / "screenshots" / f"dashboard_{row['sequence']:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        return path

    def tick(self) -> None:
        """Advance one game tic, render one activation frame, and schedule the next update."""
        if self.closed:
            return
        if self.args.max_seconds and time.perf_counter() - self.started >= self.args.max_seconds:
            self.close()
            return
        if self.game.is_episode_finished():
            self.game.new_episode()
            self.episode += 1
            self.previous = None
        total_started = time.perf_counter()
        action = key_action(self.buttons, self.pressed)
        simulation_started = time.perf_counter()
        self.game.make_action(action, 1)
        simulation_ms = (time.perf_counter() - simulation_started) * 1000.0
        if self.game.is_episode_finished():
            self.window.after(0, self.tick)
            return
        acquisition_started = time.perf_counter()
        state = self.game.get_state()
        original = state.screen_buffer.copy()
        acquisition_ms = (time.perf_counter() - acquisition_started) * 1000.0
        result = render_frame(original, self.encoder, self.model, self.config, self.device)
        display_mode = "dashboard" if self.dashboard else "comparison" if self.debug else "activation_only"
        row = temporal_row(result, self.previous) | {
            "sequence": self.sequence,
            "episode": self.episode,
            "game_tick": int(self.game.get_episode_time()),
            "pressed_keys": sorted(self.pressed),
            "action_vector": action,
            "display_mode": display_mode,
        }
        display_started = time.perf_counter()
        if self.dashboard:
            image = compose_dashboard(
                original,
                result,
                row,
                self.config,
                self.selected,
                self.dashboard_options,
                "LIVE | exact hidden-state provenance | transformer and encoder frozen",
            )
        else:
            image = comparison_image(original, result["prediction"]) if self.debug else activation_image(result["prediction"], DISPLAY_SIZE)
        self.photo = self.ImageTk.PhotoImage(image)
        self.image_label.configure(image=self.photo)
        display_ms = (time.perf_counter() - display_started) * 1000.0
        finish_row(row, total_started, simulation_ms, acquisition_ms, display_ms)
        self.recorder.write(row, original, result)
        self.last_frame = original, result, row
        elapsed = max(time.perf_counter() - self.started, 1e-9)
        self.status.configure(
            text=(
                f"mode={display_mode.replace('_', ' ')}  "
                f"recording={'on' if self.recorder.enabled else 'off'}  "
                f"fps={self.sequence / elapsed:5.1f}  renderer={row['renderer_only_ms']:5.2f} ms  "
                f"tick={row['game_tick']}"
            )
        )
        self.previous = result
        self.sequence += 1
        self.next_deadline += 1.0 / TIC_RATE
        now = time.perf_counter()
        if self.next_deadline < now - 1.0 / TIC_RATE:
            self.next_deadline = now
        self.window.after(max(0, int((self.next_deadline - now) * 1000.0)), self.tick)

    def close(self) -> None:
        """Close the game and write interactive timing and integrity summaries once."""
        if self.closed:
            return
        self.closed = True
        wall = time.perf_counter() - self.started
        self.game.close()
        self.recorder.close()
        summary = summarize(self.recorder.rows, wall) | {
            "displayed_frames": self.sequence,
            "recorded_frames": len(self.recorder.rows),
            "activation_only_frames": sum(row["display_mode"] == "activation_only" for row in self.recorder.rows),
            "comparison_frames": sum(row["display_mode"] == "comparison" for row in self.recorder.rows),
            "dashboard_frames": sum(row["display_mode"] == "dashboard" for row in self.recorder.rows),
            "minimum_requested_play_seconds": 60,
            "duration_requirement_met": wall >= 60.0,
        }
        write_json(self.root_path / "interactive" / "summary.json", summary)
        if self.recorder.rows:
            save_plots(self.root_path / "interactive", self.recorder.rows)
        integrity = {
            "encoder_sha256_before": self.encoder_before,
            "encoder_sha256_after": parameter_hash(self.encoder),
            "encoder_unchanged": self.encoder_before == parameter_hash(self.encoder),
            "transformer_sha256_before": self.model_before,
            "transformer_sha256_after": parameter_hash(self.model),
            "transformer_unchanged": self.model_before == parameter_hash(self.model),
            "trainable_parameter_count": sum(parameter.numel() for parameter in [*self.encoder.parameters(), *self.model.parameters()] if parameter.requires_grad),
        }
        write_json(self.root_path / "integrity.json", integrity)
        self.window.destroy()

    def run(self) -> None:
        """Enter the Tk event loop and start the native-rate game updates."""
        self.window.after(0, self.tick)
        self.window.mainloop()


def parser() -> argparse.ArgumentParser:
    """Build the V1-C automated-evaluation and playable-mode command-line interface."""
    root = argparse.ArgumentParser()
    subparsers = root.add_subparsers(dest="command", required=True)
    for name in ["evaluate", "play"]:
        command = subparsers.add_parser(name)
        command.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
        command.add_argument("--device", default="auto")
        command.add_argument("--seed", type=int, default=20260820)
        command.add_argument("--out")
    evaluate = subparsers.choices["evaluate"]
    evaluate.add_argument("--dataset", default="data/vizdoom_v1")
    play = subparsers.choices["play"]
    play.add_argument("--record", action="store_true")
    play.add_argument("--record-every", type=int, default=5)
    play.add_argument("--debug", action="store_true")
    play.add_argument("--dashboard", action="store_true")
    play.add_argument("--max-seconds", type=float, default=0.0)
    return root


def main() -> None:
    """Run the selected automated V1-C experiment or playable application."""
    args = parser().parse_args()
    if args.command == "evaluate":
        run_evaluate(args)
    else:
        LiveApp(args).run()


if __name__ == "__main__":
    main()
