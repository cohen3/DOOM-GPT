from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from activation_doom.activation import (
    ActivationConfig,
    predict_activation_frame,
    prediction_loss_space,
    target_loss_space,
)
from activation_doom.data.common import read_jsonl
from activation_doom.experiment import (
    fit_target,
    image_uint8,
    load_frozen,
    loss_space_uint8,
    parameter_hash,
    set_seed,
    write_json,
)
from activation_doom.preprocess import HEIGHT, WIDTH, load_target, save_target, target_gray


class FrameDataset(Dataset):
    """Keep one metadata split and its small processed frames in CPU memory."""

    def __init__(self, root: Path, records: list[dict]):
        """Load processed grayscale PNGs as one compact uint8 tensor."""
        self.records = records
        arrays = [np.asarray(Image.open(root / record["processed_image_path"]).convert("L"), dtype=np.uint8) for record in records]
        self.images = torch.from_numpy(np.stack(arrays, axis=0)[:, None])

    def __len__(self) -> int:
        """Return the number of frames in this split."""
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Return one normalized grayscale tensor and its stable split index."""
        return self.images[index].float().div(255.0), index


class ImageEncoder(nn.Module):
    """Encode a low-resolution grayscale frame into four DistilGPT2 embeddings."""

    def __init__(self, base_prompt: torch.Tensor, channels: tuple[int, int, int] = (16, 32, 64), feature_dim: int = 256):
        """Build the compact convolutional encoder with a conservative prompt head."""
        super().__init__()
        c1, c2, c3 = channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(c1, c2, 3, 2, 1),
            nn.GELU(),
            nn.Conv2d(c2, c3, 3, 2, 1),
            nn.GELU(),
        )
        with torch.no_grad():
            flattened = self.features(torch.zeros(1, 1, HEIGHT, WIDTH)).numel()
        self.project = nn.Sequential(nn.Flatten(), nn.Linear(flattened, feature_dim), nn.GELU())
        self.head = nn.Linear(feature_dim, base_prompt.numel())
        with torch.no_grad():
            self.head.weight.normal_(0.0, 1e-4)
            self.head.bias.copy_(base_prompt.reshape(-1))
        self.prompt_shape = tuple(base_prompt.shape)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return one soft prompt per input image."""
        return self.head(self.project(self.features(images))).reshape(images.shape[0], *self.prompt_shape)


def load_encoder_checkpoint(path: Path, device: torch.device) -> tuple[ImageEncoder, ActivationConfig, dict]:
    """Reconstruct a frozen encoder and activation configuration from one training checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    saved = checkpoint["config"]
    config = ActivationConfig(**saved["activation"])
    architecture = saved["encoder"]
    encoder = ImageEncoder(
        torch.zeros(config.prompt_tokens, config.hidden_size),
        tuple(architecture["channels"]),
        architecture["feature_dim"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval().requires_grad_(False)
    return encoder, config, checkpoint


def experiment_dir(out: str | None) -> Path:
    """Create the requested or timestamped amortized-renderer output directory."""
    path = Path(out) if out else Path("experiments") / "m5_amortized_renderer" / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def render_prompts(model, prompts: torch.Tensor, config: ActivationConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Return raw and globally normalized framebuffers produced only from soft prompts."""
    raw = predict_activation_frame(model, prompts, config.hidden_state_index, config.width, config.height)
    return raw, prediction_loss_space(raw, config.activation_mean, config.activation_std)


def embedding_distribution(model) -> tuple[float, float]:
    """Return scalar mean and population standard deviation of the input embedding table."""
    embedding = model.get_input_embeddings().weight.detach().float()
    return float(embedding.mean().item()), float(embedding.std(unbiased=False).item())


def random_prompts(config: ActivationConfig, mean: float, std: float, count: int, seed: int) -> torch.Tensor:
    """Draw deterministic CPU soft prompts from the frozen embedding distribution."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(count, config.prompt_tokens, config.hidden_size, generator=generator) * std + mean


def calibrate(model, config: ActivationConfig, device: torch.device, seed: int, count: int = 1024, batch_size: int = 64) -> tuple[ActivationConfig, dict, torch.Tensor]:
    """Compute and describe one global activation calibration from random soft prompts."""
    embedding_mean, embedding_std = embedding_distribution(model)
    prompts = random_prompts(config, embedding_mean, embedding_std, count, seed)
    total = total_square = 0.0
    values = 0
    with torch.no_grad():
        for start in range(0, count, batch_size):
            raw = predict_activation_frame(
                model,
                prompts[start : start + batch_size].to(device),
                config.hidden_state_index,
                config.width,
                config.height,
            ).double()
            total += float(raw.sum().item())
            total_square += float(raw.square().sum().item())
            values += raw.numel()
    activation_mean = total / values
    activation_std = math.sqrt(max(total_square / values - activation_mean**2, 1e-12))
    calibrated = replace(config, activation_mean=activation_mean, activation_std=activation_std)
    metadata = {
        "method": "population mean/std over direct activation framebuffers from deterministic random soft prompts",
        "seed": seed,
        "sample_count": count,
        "batch_size": batch_size,
        "embedding_mean": embedding_mean,
        "embedding_std": embedding_std,
        "activation_mean": activation_mean,
        "activation_std": activation_std,
        "framebuffer_values": values,
        "target_transform": "target * 2 - 1",
        "prediction_transform": "(raw - activation_mean) / activation_std",
    }
    return calibrated, metadata, prompts[0]


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    """Group metadata records by their immutable episode-level dataset split."""
    result: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        result[record["dataset_split"]].append(record)
    return dict(result)


def fixed_indices(records: list[dict], count: int, seed: int) -> list[int]:
    """Select deterministic examples while spreading choices across episodes."""
    groups: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[record["episode_id"]].append(index)
    rng = random.Random(seed)
    episodes = list(groups)
    rng.shuffle(episodes)
    for indices in groups.values():
        rng.shuffle(indices)
    chosen = []
    while len(chosen) < min(count, len(records)):
        added = False
        for episode in episodes:
            if groups[episode]:
                chosen.append(groups[episode].pop())
                added = True
                if len(chosen) == min(count, len(records)):
                    break
        if not added:
            break
    return chosen


def loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    """Build a reproducible in-memory batch loader without worker-process overhead."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, pin_memory=torch.cuda.is_available())


def evaluate(encoder: nn.Module, model, dataset: Dataset, config: ActivationConfig, device: torch.device, batch_size: int, collect: bool = False) -> tuple[dict, list[dict], np.ndarray | None]:
    """Measure per-frame MSE and MAE and optionally retain loss-space predictions."""
    encoder.eval()
    rows = []
    predictions = []
    with torch.no_grad():
        for images, indices in loader(dataset, batch_size, False, 0):
            images = images.to(device, non_blocking=True)
            _, pred = render_prompts(model, encoder(images), config)
            target = target_loss_space(images[:, 0])
            mse = (pred - target).square().flatten(1).mean(1)
            mae = (pred - target).abs().flatten(1).mean(1)
            rows.extend({"index": int(index), "mse": float(m), "mae": float(a)} for index, m, a in zip(indices, mse.cpu(), mae.cpu()))
            if collect:
                predictions.append(pred.float().cpu().numpy())
    mses = np.asarray([row["mse"] for row in rows])
    maes = np.asarray([row["mae"] for row in rows])
    metrics = {
        "frames": len(rows),
        "mean_mse": float(mses.mean()),
        "median_mse": float(np.median(mses)),
        "mean_mae": float(maes.mean()),
        "median_mae": float(np.median(maes)),
        "mse_p90": float(np.percentile(mses, 90)),
        "mse_p95": float(np.percentile(mses, 95)),
        "mse_p99": float(np.percentile(mses, 99)),
    }
    return metrics, rows, np.concatenate(predictions) if predictions else None


def save_checkpoint(path: Path, encoder: nn.Module, optimizer: torch.optim.Optimizer, epoch: int, step: int, best_mse: float, config: dict) -> None:
    """Save encoder training state and every value required to reconstruct inference."""
    torch.save(
        {
            "encoder_state_dict": encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "best_validation_mse": best_mse,
            "config": config,
        },
        path,
    )


def save_metrics_csv(path: Path, rows: list[dict]) -> None:
    """Write consistently keyed training metrics as CSV."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_curves(path: Path, rows: list[dict]) -> None:
    """Plot training and validation MSE against optimizer steps."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3), dpi=150)
    ax.plot([row["step"] for row in rows], [row["train_mse"] for row in rows], label="train")
    ax.plot([row["step"] for row in rows], [row["validation_mse"] for row in rows], label="validation")
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def normalized_display(values: np.ndarray) -> np.ndarray:
    """Return a display-only min-max normalized uint8 frame."""
    return image_uint8(values)


def save_panel(path: Path, dataset: FrameDataset, indices: list[int], predictions: np.ndarray, title: str) -> None:
    """Save target, fixed loss-space, and display-normalized columns for fixed samples."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(indices), 3, figsize=(7, max(3, len(indices) * 1.25)), dpi=120, squeeze=False)
    for row, index in enumerate(indices):
        target = dataset.images[index, 0].numpy()
        axes[row, 0].imshow(target, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].imshow(predictions[row], cmap="gray", vmin=-1, vmax=1)
        axes[row, 2].imshow(normalized_display(predictions[row]), cmap="gray", vmin=0, vmax=255)
        for axis in axes[row]:
            axis.axis("off")
    axes[0, 0].set_title("target")
    axes[0, 1].set_title("prediction: loss space")
    axes[0, 2].set_title("prediction: display only")
    fig.suptitle(title, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path)
    plt.close(fig)


def panel_predictions(encoder: nn.Module, model, dataset: FrameDataset, indices: list[int], config: ActivationConfig, device: torch.device) -> np.ndarray:
    """Render a fixed list of dataset indices for visual comparison."""
    images = dataset.images[indices].float().div(255.0).to(device)
    encoder.eval()
    with torch.no_grad():
        _, predictions = render_prompts(model, encoder(images), config)
    return predictions.float().cpu().numpy()


def train_run(
    name: str,
    out: Path,
    model,
    train_data: Dataset,
    validation_data: FrameDataset,
    validation_indices: list[int],
    config: ActivationConfig,
    base_prompt: torch.Tensor,
    experiment_config: dict,
    device: torch.device,
    channels: tuple[int, int, int],
    feature_dim: int,
    batch_size: int,
    learning_rate: float,
    max_epochs: int,
    seed: int,
    evaluate_every: int = 1,
    patience: int | None = None,
    target_ratio: float | None = None,
    save_epoch_panels: bool = False,
) -> tuple[ImageEncoder, dict]:
    """Train one fresh encoder run while enforcing gradient and frozen-model checks."""
    set_seed(seed)
    run_out = out / name
    run_out.mkdir(parents=True, exist_ok=True)
    panels = run_out / "validation_panels"
    panels.mkdir(exist_ok=True)
    encoder = ImageEncoder(base_prompt, channels, feature_dim).to(device)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    before_hash = parameter_hash(model)
    initial, _, _ = evaluate(encoder, model, validation_data, config, device, batch_size)
    best = initial["mean_mse"]
    save_checkpoint(run_out / "best.pt", encoder, optimizer, 0, 0, best, experiment_config)
    stale = 0
    step = 0
    rows = []
    first_token_grad_norms = None
    gradient_reached_features = False
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        encoder.train()
        squared_error = absolute_error = pixels = frames = 0.0
        grad_norm_sum = prompt_norm_sum = active_prompt_norm_sum = 0.0
        batches = 0
        epoch_started = time.perf_counter()
        for images, _ in loader(train_data, batch_size, True, seed + epoch):
            images = images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prompts = encoder(images)
            if first_token_grad_norms is None:
                prompts.retain_grad()
            _, prediction = render_prompts(model, prompts, config)
            target = target_loss_space(images[:, 0])
            if not torch.isfinite(prediction).all():
                raise RuntimeError(f"{name}: non-finite activation prediction")
            loss = nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise RuntimeError(f"{name}: non-finite loss")
            loss.backward()
            if first_token_grad_norms is None:
                first_token_grad_norms = prompts.grad.detach().norm(dim=2).mean(0).cpu().tolist()
                if first_token_grad_norms[-1] != 0.0:
                    raise RuntimeError(f"{name}: fourth causal prompt token unexpectedly received gradient")
            gradient_reached_features |= any(
                parameter.grad is not None and bool(torch.count_nonzero(parameter.grad).item())
                for parameter in encoder.features.parameters()
            )
            if not gradient_reached_features:
                raise RuntimeError(f"{name}: gradients did not reach convolutional encoder")
            grad_norm = float(nn.utils.clip_grad_norm_(encoder.parameters(), 5.0).item())
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"{name}: non-finite encoder gradient")
            optimizer.step()
            error = prediction.detach() - target
            squared_error += float(error.square().sum().item())
            absolute_error += float(error.abs().sum().item())
            pixels += error.numel()
            frames += images.shape[0]
            grad_norm_sum += grad_norm
            prompt_norm_sum += float(prompts.detach().flatten(1).norm(dim=1).mean().item())
            active_prompt_norm_sum += float(prompts.detach()[:, :3].flatten(1).norm(dim=1).mean().item())
            batches += 1
            step += 1

        if epoch % evaluate_every:
            continue
        validation, _, _ = evaluate(encoder, model, validation_data, config, device, batch_size)
        elapsed = max(time.perf_counter() - epoch_started, 1e-9)
        row = {
            "epoch": epoch,
            "step": step,
            "train_mse": squared_error / pixels,
            "train_mae": absolute_error / pixels,
            "validation_mse": validation["mean_mse"],
            "validation_mae": validation["mean_mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "gradient_norm": grad_norm_sum / batches,
            "soft_prompt_norm": prompt_norm_sum / batches,
            "active_prompt_norm": active_prompt_norm_sum / batches,
            "frames_per_second": frames / elapsed,
        }
        rows.append(row)
        improved = validation["mean_mse"] < best - 1e-5
        if improved:
            best = validation["mean_mse"]
            stale = 0
            save_checkpoint(run_out / "best.pt", encoder, optimizer, epoch, step, best, experiment_config)
        else:
            stale += 1
        save_checkpoint(run_out / "last.pt", encoder, optimizer, epoch, step, best, experiment_config)
        if save_epoch_panels:
            predictions = panel_predictions(encoder, model, validation_data, validation_indices, config, device)
            save_panel(panels / f"epoch_{epoch:03d}.png", validation_data, validation_indices, predictions, f"{name}, epoch {epoch}")
        print(json.dumps({"run": name, **row}), flush=True)
        if target_ratio is not None and validation["mean_mse"] <= initial["mean_mse"] * target_ratio:
            break
        if patience is not None and stale >= patience:
            break

    checkpoint = torch.load(run_out / "best.pt", map_location=device, weights_only=False)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    final, _, _ = evaluate(encoder, model, validation_data, config, device, batch_size)
    final_predictions = panel_predictions(encoder, model, validation_data, validation_indices, config, device)
    save_panel(run_out / "final_panel.png", validation_data, validation_indices, final_predictions, f"{name}, best checkpoint")
    save_metrics_csv(run_out / "metrics.csv", rows)
    save_curves(run_out / "curves.png", rows)
    after_hash = parameter_hash(model)
    summary = {
        "name": name,
        "encoder_parameter_count": sum(parameter.numel() for parameter in encoder.parameters()),
        "initial_validation": initial,
        "best_validation_mse": best,
        "final_validation": final,
        "epochs": rows[-1]["epoch"],
        "steps": step,
        "seconds": time.perf_counter() - started,
        "first_prompt_token_gradient_norms": first_token_grad_norms,
        "gradient_reached_features": gradient_reached_features,
        "transformer_trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "transformer_parameter_hash_before": before_hash,
        "transformer_parameter_hash_after": after_hash,
        "transformer_parameter_hash_unchanged": before_hash == after_hash,
        "cuda_peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / 1024**2) if device.type == "cuda" else None,
    }
    write_json(run_out / "summary.json", summary)
    return encoder, summary


def save_error_histogram(path: Path, errors: list[float]) -> None:
    """Save the held-out per-frame MSE distribution."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.hist(errors, bins=50)
    ax.set_xlabel("per-frame MSE")
    ax.set_ylabel("frames")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_contact_sheet(path: Path, dataset: FrameDataset, predictions: np.ndarray, indices: list[int], columns: int = 10) -> None:
    """Tile target, loss-space prediction, and display prediction triplets."""
    rows = math.ceil(len(indices) / columns)
    sheet = Image.new("L", (columns * WIDTH * 3, rows * HEIGHT), 0)
    for position, index in enumerate(indices):
        x = (position % columns) * WIDTH * 3
        y = (position // columns) * HEIGHT
        sheet.paste(Image.fromarray(dataset.images[index, 0].numpy(), mode="L"), (x, y))
        sheet.paste(Image.fromarray(loss_space_uint8(predictions[index]), mode="L"), (x + WIDTH, y))
        sheet.paste(Image.fromarray(normalized_display(predictions[index]), mode="L"), (x + WIDTH * 2, y))
    sheet.save(path)


def failure_categories(record: dict, target: np.ndarray) -> list[str]:
    """Attach conservative metadata-supported labels to a difficult test frame."""
    labels = []
    if record.get("player_dead") or record.get("episode_finished"):
        labels.append("death_or_episode_end")
    if record.get("action_label") == "combat":
        labels.append("combat")
    if float(target.mean()) < 64:
        labels.append("dark_scene")
    if record.get("health", 100) <= 20:
        labels.append("low_health")
    return labels or ["visual_inspection_required"]


def evaluate_test(out: Path, encoder: nn.Module, model, dataset: FrameDataset, config: ActivationConfig, device: torch.device, batch_size: int, seed: int) -> tuple[dict, list[dict], np.ndarray]:
    """Evaluate the selected checkpoint once on the held-out test split and save diagnostics."""
    evaluation_out = out / "test_evaluation"
    evaluation_out.mkdir(exist_ok=True)
    metrics, rows, predictions = evaluate(encoder, model, dataset, config, device, batch_size, collect=True)
    for row in rows:
        row["sample_id"] = dataset.records[row["index"]]["sample_id"]
    save_metrics_csv(evaluation_out / "per_frame_metrics.csv", rows)
    write_json(evaluation_out / "metrics.json", metrics)
    save_error_histogram(evaluation_out / "mse_histogram.png", [row["mse"] for row in rows])
    random_indices = random.Random(seed).sample(range(len(dataset)), min(100, len(dataset)))
    save_contact_sheet(evaluation_out / "test_contact_sheet.png", dataset, predictions, random_indices)
    worst = sorted(rows, key=lambda row: row["mse"], reverse=True)[:20]
    worst_indices = [row["index"] for row in worst]
    save_contact_sheet(evaluation_out / "worst_20.png", dataset, predictions, worst_indices, columns=5)
    write_json(
        evaluation_out / "worst_20.json",
        {
            "frames": [
                row
                | {
                    "categories": failure_categories(dataset.records[row["index"]], dataset.images[row["index"], 0].numpy()),
                    "metadata": dataset.records[row["index"]],
                }
                for row in worst
            ]
        },
    )
    return metrics, rows, predictions


def run_oracle(out: Path, tokenizer, model, encoder_rows: list[dict], dataset: FrameDataset, indices: list[int], config: ActivationConfig, device: torch.device, steps: int) -> dict:
    """Compare one encoder pass with iterative inversion on a small fixed test subset."""
    oracle_out = out / "oracle"
    oracle_out.mkdir(exist_ok=True)
    encoder_error = {row["index"]: row["mse"] for row in encoder_rows}
    results = []
    for index in indices:
        record = dataset.records[index]
        frame_out = oracle_out / record["sample_id"]
        frame_out.mkdir(exist_ok=True)
        target = dataset.images[index, 0].numpy().astype(np.float32) / 255.0
        save_target(frame_out / "target.png", target)
        args = argparse.Namespace(
            seed=1234,
            prompt_tokens=config.prompt_tokens,
            lr=0.03,
            steps=steps,
            layer=config.hidden_state_index,
            width=config.width,
            height=config.height,
            model=config.model,
        )
        metadata = fit_target(
            args,
            tokenizer,
            model,
            device,
            target,
            frame_out,
            f"held-out test frame {record['sample_id']}",
            normalization=(config.activation_mean, config.activation_std),
        )
        iterative = metadata["final_loss"]
        encoded = encoder_error[index]
        results.append(
            {
                "index": index,
                "sample_id": record["sample_id"],
                "encoder_mse": encoded,
                "iterative_inversion_mse": iterative,
                "ratio": encoded / iterative if iterative else None,
                "absolute_gap": encoded - iterative,
            }
        )
        print(json.dumps({"oracle": results[-1]}), flush=True)
    save_metrics_csv(oracle_out / "comparison.csv", results)
    summary = {
        "frames": len(results),
        "steps_per_frame": steps,
        "seed": 1234,
        "mean_encoder_mse": float(np.mean([row["encoder_mse"] for row in results])),
        "mean_iterative_mse": float(np.mean([row["iterative_inversion_mse"] for row in results])),
        "median_ratio": float(np.median([row["ratio"] for row in results])),
        "results": results,
    }
    write_json(oracle_out / "comparison.json", summary)
    return summary


def timed(function, warmup: int, iterations: int, synchronize: bool) -> dict:
    """Benchmark a callable after warmup and return synchronized latency statistics."""
    for _ in range(warmup):
        function()
    if synchronize:
        torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        started = time.perf_counter()
        function()
        if synchronize:
            torch.cuda.synchronize()
        values.append((time.perf_counter() - started) * 1000.0)
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)), "p95_ms": float(np.percentile(values, 95))}


def benchmark(out: Path, root: Path, record: dict, encoder: nn.Module, model, config: ActivationConfig, device: torch.device, iterations: int = 500) -> dict:
    """Measure each batch-1 renderer component and the complete raw-frame pipeline."""
    source = Image.open(root / record["source_image_path"]).convert("RGB")
    cpu_image = torch.from_numpy(target_gray(source)[None, None]).float()
    image = cpu_image.to(device)
    encoder.eval()
    with torch.no_grad():
        prompt = encoder(image)
        outputs = model(inputs_embeds=prompt, attention_mask=torch.ones(prompt.shape[:2], device=device), output_hidden_states=True, use_cache=False, return_dict=True)
        hidden = outputs.hidden_states[config.hidden_state_index]
        raw, _ = render_prompts(model, prompt, config)

        stages = {
            "preprocessing": timed(lambda: target_gray(source), 10, iterations, False),
            "host_to_device": timed(lambda: cpu_image.to(device), 50, iterations, device.type == "cuda"),
            "encoder": timed(lambda: encoder(image), 50, iterations, device.type == "cuda"),
            "transformer": timed(
                lambda: model(inputs_embeds=prompt, attention_mask=torch.ones(prompt.shape[:2], device=device), output_hidden_states=True, use_cache=False, return_dict=True),
                50,
                iterations,
                device.type == "cuda",
            ),
            "activation_extraction": timed(lambda: hidden.reshape(1, -1)[:, : config.width * config.height].reshape(1, config.height, config.width), 50, iterations, device.type == "cuda"),
            "loss_space_conversion": timed(lambda: prediction_loss_space(raw, config.activation_mean, config.activation_std), 50, iterations, device.type == "cuda"),
        }

        def complete() -> None:
            """Render one stored raw frame through preprocessing, encoder, transformer, and activation mapping."""
            tensor = torch.from_numpy(target_gray(source)[None, None]).float().to(device)
            render_prompts(model, encoder(tensor), config)

        total = timed(complete, 50, iterations, device.type == "cuda")
    report = {
        "batch_size": 1,
        "warmup_iterations": 50,
        "timed_iterations": iterations,
        "precision": "float32",
        "stages": stages,
        "total": total,
        "theoretical_fps": 1000.0 / total["mean_ms"],
    }
    write_json(out / "latency.json", report)
    return report


def verify_preprocessing(root: Path, records: list[dict], count: int = 16) -> bool:
    """Confirm stored targets exactly match fresh shared preprocessing for sampled records."""
    for record in records[:count]:
        regenerated = target_gray(Image.open(root / record["source_image_path"]))
        stored = load_target(root / record["processed_image_path"])
        if not np.array_equal(np.rint(regenerated * 255).astype(np.uint8), np.rint(stored * 255).astype(np.uint8)):
            return False
    return True


def parser() -> argparse.ArgumentParser:
    """Build the V1-B phased experiment command-line interface."""
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("stage", choices=["overfit", "smoke", "full", "all"], default="all", nargs="?")
    argument_parser.add_argument("--dataset", default="data/vizdoom_v1")
    argument_parser.add_argument("--out")
    argument_parser.add_argument("--device", default="auto")
    argument_parser.add_argument("--seed", type=int, default=20260819)
    argument_parser.add_argument("--batch-size", type=int, default=64)
    argument_parser.add_argument("--learning-rate", type=float, default=1e-3)
    argument_parser.add_argument("--max-epochs", type=int, default=30)
    argument_parser.add_argument("--channels", type=int, nargs=3, default=[16, 32, 64])
    argument_parser.add_argument("--feature-dim", type=int, default=256)
    argument_parser.add_argument("--oracle-count", type=int, default=16)
    argument_parser.add_argument("--oracle-steps", type=int, default=1500)
    argument_parser.add_argument("--benchmark-iterations", type=int, default=500)
    return argument_parser


def main() -> None:
    """Run the requested V1-B gates, full training, held-out evaluation, and benchmarks."""
    args = parser().parse_args()
    root = Path(args.dataset)
    metadata_path = root / "metadata.jsonl"
    if not metadata_path.exists():
        raise FileNotFoundError(f"dataset metadata not found: {metadata_path}")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    set_seed(args.seed)
    out = experiment_dir(args.out)
    records = read_jsonl(metadata_path)
    grouped = split_records(records)
    if {"train", "val", "test"} - grouped.keys():
        raise ValueError("dataset must contain train, val, and test episode splits")
    print(json.dumps({"loading_frames": {key: len(value) for key, value in grouped.items()}}), flush=True)
    datasets = {key: FrameDataset(root, value) for key, value in grouped.items()}
    tokenizer, model = load_frozen("distilbert/distilgpt2", device)
    activation = ActivationConfig()
    activation, calibration, base_prompt = calibrate(model, activation, device, args.seed)
    channels = tuple(args.channels)
    probe_encoder = ImageEncoder(base_prompt, channels, args.feature_dim)
    validation_indices = fixed_indices(grouped["val"], 16, args.seed)
    oracle_indices = fixed_indices(grouped["test"], args.oracle_count, args.seed + 1)
    experiment_config = {
        "activation": activation.to_dict(),
        "calibration": calibration,
        "encoder": {
            "channels": list(channels),
            "feature_dim": args.feature_dim,
            "parameter_count": sum(parameter.numel() for parameter in probe_encoder.parameters()),
            "output_head_weight_std": 1e-4,
            "output_head_bias": "first calibration prompt",
        },
        "preprocessing": {"width": WIDTH, "height": HEIGHT, "mode": "grayscale", "scale": "[0, 1]"},
        "training": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "gradient_clip": 5.0,
            "precision": "float32",
            "max_epochs": args.max_epochs,
            "early_stopping_patience": 5,
        },
        "dataset": str(root),
        "split_counts": {key: len(value) for key, value in grouped.items()},
        "fixed_validation_indices": validation_indices,
        "fixed_validation_sample_ids": [grouped["val"][index]["sample_id"] for index in validation_indices],
        "oracle_indices": oracle_indices,
        "oracle_sample_ids": [grouped["test"][index]["sample_id"] for index in oracle_indices],
    }
    write_json(out / "config.json", experiment_config)
    write_json(out / "calibration.json", calibration)
    del probe_encoder

    overfit_summary = smoke_summary = full_summary = None
    if args.stage in {"overfit", "all"}:
        tiny_indices = fixed_indices(grouped["train"], 32, args.seed)
        tiny = FrameDataset(root, [grouped["train"][index] for index in tiny_indices])
        tiny_panel_indices = list(range(len(tiny)))
        _, overfit_summary = train_run(
            "tiny_overfit",
            out,
            model,
            tiny,
            tiny,
            tiny_panel_indices,
            activation,
            base_prompt,
            experiment_config,
            device,
            channels,
            args.feature_dim,
            32,
            args.learning_rate,
            3000,
            args.seed,
            evaluate_every=100,
            target_ratio=0.25,
        )
        overfit_summary["gate_passed"] = overfit_summary["best_validation_mse"] <= overfit_summary["initial_validation"]["mean_mse"] * 0.25
        write_json(out / "tiny_overfit" / "summary.json", overfit_summary)
        if args.stage == "all" and not overfit_summary["gate_passed"]:
            raise RuntimeError("tiny overfit gate failed; full training was not started")

    if args.stage in {"smoke", "all"}:
        smoke_train_indices = fixed_indices(grouped["train"], 1024, args.seed + 2)
        smoke_validation_indices = fixed_indices(grouped["val"], 256, args.seed + 3)
        smoke_train = Subset(datasets["train"], smoke_train_indices)
        smoke_validation = FrameDataset(root, [grouped["val"][index] for index in smoke_validation_indices])
        smoke_panel_indices = list(range(min(16, len(smoke_validation))))
        _, smoke_summary = train_run(
            "smoke",
            out,
            model,
            smoke_train,
            smoke_validation,
            smoke_panel_indices,
            activation,
            base_prompt,
            experiment_config,
            device,
            channels,
            args.feature_dim,
            args.batch_size,
            args.learning_rate,
            2,
            args.seed + 1,
            save_epoch_panels=True,
        )
        smoke_summary["gate_passed"] = smoke_summary["best_validation_mse"] < smoke_summary["initial_validation"]["mean_mse"]
        write_json(out / "smoke" / "summary.json", smoke_summary)
        if args.stage == "all" and not smoke_summary["gate_passed"]:
            raise RuntimeError("smoke gate failed; full training was not started")

    if args.stage in {"full", "all"}:
        encoder, full_summary = train_run(
            "full",
            out,
            model,
            datasets["train"],
            datasets["val"],
            validation_indices,
            activation,
            base_prompt,
            experiment_config,
            device,
            channels,
            args.feature_dim,
            args.batch_size,
            args.learning_rate,
            args.max_epochs,
            args.seed + 2,
            patience=5,
            save_epoch_panels=True,
        )
        test_metrics, test_rows, predictions = evaluate_test(out, encoder, model, datasets["test"], activation, device, args.batch_size, args.seed)
        oracle = run_oracle(out, tokenizer, model, test_rows, datasets["test"], oracle_indices, activation, device, args.oracle_steps)
        latency = benchmark(out, root, grouped["test"][0], encoder, model, activation, device, args.benchmark_iterations)
        final_hash = parameter_hash(model)
        integrity = {
            "trainable_transformer_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "transformer_hash_matches_full_training_start": final_hash == full_summary["transformer_parameter_hash_before"],
            "gradients_reached_encoder": full_summary["gradient_reached_features"],
            "fourth_prompt_token_gradient_is_zero": full_summary["first_prompt_token_gradient_norms"][-1] == 0.0,
            "renderer_accepts_only_soft_prompts_after_encoder": list(inspect.signature(render_prompts).parameters) == ["model", "prompts", "config"],
            "direct_hidden_state_framebuffer": True,
            "transformer_gradients_absent": all(parameter.grad is None for parameter in model.parameters()),
            "preprocessing_identical": all(verify_preprocessing(root, grouped[split]) for split in ["train", "val", "test"]),
        }
        write_json(out / "integrity.json", integrity)
        final = {
            "encoder": experiment_config["encoder"],
            "full_training": full_summary,
            "test": test_metrics,
            "oracle": oracle,
            "latency": latency,
            "integrity": integrity,
            "qualitative_decision": "requires inspection of saved panels",
        }
        write_json(out / "final_metrics.json", final)
        print(json.dumps({"completed": str(out), "test": test_metrics, "oracle": oracle, "latency": latency}), flush=True)


if __name__ == "__main__":
    main()
