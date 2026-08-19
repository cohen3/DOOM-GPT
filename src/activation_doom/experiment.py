from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import shutil
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_doom.preprocess import save_target, target_gray


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable experiment runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_for(name: str) -> torch.device:
    """Return the requested torch device or CUDA when available in auto mode."""
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_frozen(model_name: str, device: torch.device):
    """Load a Hugging Face causal LM and freeze every model parameter."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return tokenizer, model


def activation_frame(hidden: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Flatten a hidden-state tensor and reshape its first values into a framebuffer."""
    need = width * height
    flat = hidden.reshape(-1)
    if flat.numel() < need:
        raise ValueError(f"activation has {flat.numel()} values, need {need}")
    return flat[:need].reshape(height, width)


def image_uint8(values) -> np.ndarray:
    """Convert values to uint8 with per-image min-max normalization for display only."""
    arr = values.detach().float().cpu().numpy() if torch.is_tensor(values) else np.asarray(values, dtype=np.float32)
    arr = np.nan_to_num(arr.astype(np.float32), copy=False)
    lo = float(arr.min())
    hi = float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def loss_space_uint8(values) -> np.ndarray:
    """Convert loss-space values in [-1, 1] to uint8 without per-image renormalization."""
    arr = values.detach().float().cpu().numpy() if torch.is_tensor(values) else np.asarray(values, dtype=np.float32)
    return np.clip((arr + 1.0) * 127.5, 0, 255).astype(np.uint8)


def save_gray(path: Path, values) -> None:
    """Save values as a grayscale PNG using display-only normalization."""
    Image.fromarray(image_uint8(values), mode="L").save(path)


def save_loss_gray(path: Path, values) -> None:
    """Save loss-space values as a grayscale PNG using the fixed [-1, 1] scale."""
    Image.fromarray(loss_space_uint8(values), mode="L").save(path)


def synthetic_target(width: int, height: int) -> np.ndarray:
    """Create a simple outlined A target as float grayscale values in [0, 1]."""
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    thick = max(1, min(width, height) // 10)
    draw.rectangle(
        [width // 5, height // 5, width * 4 // 5, height * 4 // 5],
        outline=180,
        width=max(1, thick // 2),
    )
    draw.line([width // 4, height * 3 // 4, width // 2, height // 4, width * 3 // 4, height * 3 // 4], fill=255, width=thick)
    draw.line([width * 3 // 8, height // 2, width * 5 // 8, height // 2], fill=255, width=max(1, thick // 2))
    return np.asarray(img, dtype=np.float32) / 255.0


def experiment_dir(kind: str, out: str | None) -> Path:
    """Create and return a timestamped experiment output directory."""
    path = Path(out) if out else Path("experiments") / kind / time.strftime("%Y%m%d-%H%M%S")
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: dict) -> None:
    """Write a metadata dictionary as pretty JSON."""
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def save_loss_curve(path: Path, losses: list[float]) -> None:
    """Save a simple MSE-over-steps line plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3), dpi=150)
    ax.plot(losses)
    ax.set_xlabel("step")
    ax.set_ylabel("MSE")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def parameter_hash(model) -> str:
    """Return a SHA-256 hash over every tensor in a model state dict."""
    h = hashlib.sha256()
    for tensor in model.state_dict().values():
        arr = tensor.detach().cpu().contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def model_metadata(model, device: torch.device, layer: int) -> dict:
    """Collect model configuration and freezing metadata for an experiment run."""
    forward_params = inspect.signature(model.forward).parameters
    params = list(model.parameters())
    return {
        "device": str(device),
        "input_embeds_supported": "inputs_embeds" in forward_params,
        "all_model_parameters_frozen": all(not p.requires_grad for p in params),
        "parameter_count": int(sum(p.numel() for p in params)),
        "trainable_model_parameter_count": int(sum(p.numel() for p in params if p.requires_grad)),
        "selected_hidden_state_index": layer,
        "config": {
            "model_type": getattr(model.config, "model_type", None),
            "n_layer": getattr(model.config, "n_layer", None),
            "n_embd": getattr(model.config, "n_embd", None),
            "n_head": getattr(model.config, "n_head", None),
            "n_positions": getattr(model.config, "n_positions", None),
        },
    }


def predict_activation_frame(model, soft: torch.Tensor, mask: torch.Tensor, layer: int, width: int, height: int) -> torch.Tensor:
    """Run a soft prompt through the frozen model and return the selected activation framebuffer."""
    outputs = model(
        inputs_embeds=soft,
        attention_mask=mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return activation_frame(outputs.hidden_states[layer], width, height)


def fit_target(args: argparse.Namespace, tokenizer, model, device: torch.device, target: np.ndarray, out: Path, label: str) -> dict:
    """Optimize one soft prompt so a frozen-model activation framebuffer matches a target image."""
    set_seed(args.seed)
    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    target_t = torch.tensor(target * 2.0 - 1.0, dtype=torch.float32, device=device)
    before = parameter_hash(model)
    emb = model.get_input_embeddings().weight.detach()
    soft = torch.nn.Parameter(
        torch.randn(1, args.prompt_tokens, emb.shape[1], device=device) * emb.float().std().to(device)
        + emb.float().mean().to(device)
    )
    opt = torch.optim.Adam([soft], lr=args.lr)
    mask = torch.ones(1, args.prompt_tokens, device=device)

    with torch.no_grad():
        initial_raw = predict_activation_frame(model, soft, mask, args.layer, args.width, args.height)
        mean = initial_raw.mean().detach()
        std = initial_raw.std().clamp_min(1e-6).detach()
        initial_loss = torch.nn.functional.mse_loss((initial_raw - mean) / std, target_t).item()
    save_gray(out / "initial_activation.png", initial_raw)

    losses: list[float] = []
    first_grad_norm = None
    for _ in range(args.steps):
        opt.zero_grad(set_to_none=True)
        pred = (predict_activation_frame(model, soft, mask, args.layer, args.width, args.height) - mean) / std
        loss = torch.nn.functional.mse_loss(pred, target_t)
        loss.backward()
        grad_norm = float(soft.grad.detach().norm().item())
        if first_grad_norm is None:
            first_grad_norm = grad_norm
        opt.step()
        losses.append(float(loss.detach().item()))

    with torch.no_grad():
        final_raw = predict_activation_frame(model, soft, mask, args.layer, args.width, args.height)
        final_loss_space = (final_raw - mean) / std
        final_loss = torch.nn.functional.mse_loss(final_loss_space, target_t).item()
        shape = list(model(inputs_embeds=soft, attention_mask=mask, output_hidden_states=True, use_cache=False, return_dict=True).hidden_states[args.layer].shape)

    save_loss_gray(out / "final_loss_space.png", final_loss_space)
    save_gray(out / "final_display.png", final_raw)
    save_gray(out / "final_activation.png", final_raw)
    save_loss_curve(out / "loss_curve.png", [initial_loss] + losses + [final_loss])
    with (out / "losses.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for i, loss in enumerate([initial_loss] + losses + [final_loss]):
            writer.writerow([i, loss])

    after = parameter_hash(model)
    meta = model_metadata(model, device, args.layer) | {
        "model": args.model,
        "tokenizer_loaded": tokenizer.__class__.__name__,
        "selected_hidden_state_shape": shape,
        "framebuffer_shape": [args.height, args.width],
        "soft_prompt_shape": list(soft.shape),
        "target": label,
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.lr,
        "loss_normalization": "fixed mean/std from initial activation frame",
        "loss_space_image_scale": "fixed [-1, 1] mapped to [0, 255]",
        "visualization_normalization": "per-image min-max only",
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_decreased": final_loss < initial_loss,
        "first_soft_prompt_grad_norm": first_grad_norm,
        "model_parameter_hash_unchanged": before == after,
        "cuda_peak_memory_mb": (
            float(torch.cuda.max_memory_allocated(device) / 1024**2)
            if torch.cuda.is_available() and device.type == "cuda"
            else None
        ),
        "artifacts": {
            "initial_activation": str(out / "initial_activation.png"),
            "final_loss_space": str(out / "final_loss_space.png"),
            "final_display": str(out / "final_display.png"),
            "loss_curve": str(out / "loss_curve.png"),
            "losses": str(out / "losses.csv"),
        },
    }
    write_json(out / "metadata.json", meta)
    return meta


def resize_gray(image: Image.Image | np.ndarray, width: int, height: int) -> np.ndarray:
    """Convert an image to grayscale, resize it, and return float values in [0, 1]."""
    return target_gray(image, width, height)


def capture_doom_frame(seed: int):
    """Capture one deterministic RGB frame from ViZDoom's bundled basic scenario."""
    import vizdoom as vzd

    game = vzd.DoomGame()
    game.load_config(str(Path(vzd.scenarios_path) / "basic.cfg"))
    game.set_window_visible(False)
    game.set_sound_enabled(False)
    game.set_seed(seed)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    game.init()
    try:
        return game.get_state().screen_buffer.copy()
    finally:
        game.close()


def run_viewer(args: argparse.Namespace) -> Path:
    """Run a text prompt through a frozen transformer and save one activation image."""
    set_seed(args.seed)
    out = experiment_dir("m1_activation_viewer", args.out)
    device = device_for(args.device)
    tokenizer, model = load_frozen(args.model, device)
    tokens = tokenizer(args.text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**tokens, output_hidden_states=True, use_cache=False, return_dict=True)

    hidden = outputs.hidden_states[args.layer]
    frame = activation_frame(hidden, args.width, args.height)
    save_gray(out / "activation.png", frame)

    meta = model_metadata(model, device, args.layer) | {
        "model": args.model,
        "text": args.text,
        "input_token_count": int(tokens["input_ids"].shape[1]),
        "hidden_state_count": len(outputs.hidden_states),
        "selected_hidden_state_shape": list(hidden.shape),
        "framebuffer_shape": [args.height, args.width],
        "visualization_normalization": "per-image min-max only",
        "artifact": str(out / "activation.png"),
    }
    write_json(out / "metadata.json", meta)
    print(json.dumps(meta, indent=2))
    return out


def run_invert(args: argparse.Namespace) -> Path:
    """Fit the synthetic target with the soft-prompt activation inversion loop."""
    set_seed(args.seed)
    out = experiment_dir("m2_activation_inversion", args.out)
    device = device_for(args.device)
    tokenizer, model = load_frozen(args.model, device)

    target = synthetic_target(args.width, args.height)
    save_gray(out / "target.png", target)
    meta = fit_target(args, tokenizer, model, device, target, out, "outlined A shape")
    print(json.dumps(meta, indent=2))
    return out


def run_doom(args: argparse.Namespace) -> Path:
    """Fit one captured ViZDoom frame with three soft-prompt random seeds."""
    out = experiment_dir("m3_doom_frame", args.out)
    original = capture_doom_frame(args.frame_seed)
    Image.fromarray(original, mode="RGB").save(out / "original_doom_frame.png")
    target = resize_gray(original, args.width, args.height)
    save_target(out / "resized_grayscale_target.png", target)

    device = device_for(args.device)
    tokenizer, model = load_frozen(args.model, device)
    results = []
    for seed in args.seeds:
        seed_out = out / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out / "original_doom_frame.png", seed_out / "original_doom_frame.png")
        shutil.copy2(out / "resized_grayscale_target.png", seed_out / "resized_grayscale_target.png")
        args.seed = seed
        meta = fit_target(args, tokenizer, model, device, target, seed_out, "ViZDoom basic.cfg frame")
        meta["artifacts"] |= {
            "original_doom_frame": str(seed_out / "original_doom_frame.png"),
            "resized_grayscale_target": str(seed_out / "resized_grayscale_target.png"),
        }
        write_json(seed_out / "metadata.json", meta)
        results.append(meta)
        print(json.dumps(meta, indent=2))

    summary = {
        "command": "doom",
        "model": args.model,
        "frame_seed": args.frame_seed,
        "seeds": args.seeds,
        "results": [
            {
                "seed": meta["seed"],
                "initial_loss": meta["initial_loss"],
                "final_loss": meta["final_loss"],
                "loss_decreased": meta["loss_decreased"],
                "model_parameter_hash_unchanged": meta["model_parameter_hash_unchanged"],
            }
            for meta in results
        ],
    }
    write_json(out / "summary.json", summary)
    return out


def parser() -> argparse.ArgumentParser:
    """Build the command-line parser for all experiment entry points."""
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["viewer", "invert", "doom"])
    p.add_argument("--model", default="distilbert/distilgpt2")
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--text", default="Activation Doom probe")
    p.add_argument("--prompt-tokens", type=int, default=4)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--seeds", type=int, nargs="+", default=[1234, 1235, 1236])
    p.add_argument("--frame-seed", type=int, default=777)
    return p


def main() -> None:
    """Parse CLI arguments, fill command defaults, and run the chosen experiment."""
    args = parser().parse_args()
    if args.command == "viewer":
        args.width = args.width or 32
        args.height = args.height or 24
        run_viewer(args)
    elif args.command == "invert":
        args.width = args.width or 64
        args.height = args.height or 32
        run_invert(args)
    else:
        args.width = args.width or 64
        args.height = args.height or 32
        if args.steps == 200:
            args.steps = 1500
        run_doom(args)


if __name__ == "__main__":
    main()
