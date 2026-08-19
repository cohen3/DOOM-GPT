from __future__ import annotations

import argparse
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from activation_doom.activation import ActivationConfig
from activation_doom.data.common import read_jsonl
from activation_doom.experiment import loss_space_uint8


SIZE = (1600, 900)
TIC_RATE = 35.0
ORIGINAL_RECT = (20, 90, 320, 240)
TARGET_RECT = (370, 90, 290, 145)
METRICS_RECT = (370, 270, 290, 105)
HIDDEN_RECT = (35, 405, 625, 85)
FINAL_RECT = (690, 90, 890, 445)
RAW_RECT = (20, 590, 500, 250)
PROVENANCE_RECT = (540, 590, 500, 250)
INFO_RECT = (1060, 590, 520, 250)
TOKEN_COLORS = np.asarray([(39, 174, 96), (52, 152, 219), (241, 196, 15), (80, 85, 95)], dtype=np.uint8)
BG = (11, 14, 20)
PANEL = (20, 25, 34)
EDGE = (65, 73, 88)
TEXT = (235, 239, 245)
MUTED = (154, 163, 178)
ACCENT = (246, 99, 62)


@lru_cache(maxsize=None)
def font(size: int, bold: bool = False):
    """Use Windows' presentation fonts with Pillow's portable fallback."""
    name = "segoeuib.ttf" if bold else "consola.ttf"
    try:
        return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size)
    except OSError:
        return ImageFont.load_default()


def pixel_source(x: int, y: int, width: int = 64, height: int = 32, hidden_size: int = 768) -> dict:
    """Map one framebuffer coordinate to the exact flattened hidden-state value."""
    if not 0 <= x < width or not 0 <= y < height:
        raise ValueError(f"pixel ({x}, {y}) is outside {width}x{height}")
    flat = y * width + x
    return {"x": x, "y": y, "flat": flat, "token": flat // hidden_size, "dimension": flat % hidden_size}


def source_token_map(width: int = 64, height: int = 32, hidden_size: int = 768) -> np.ndarray:
    """Return the prompt-token source of every selected framebuffer pixel."""
    return (np.arange(width * height) // hidden_size).reshape(height, width)


def pixel_from_point(px: int, py: int, width: int = 64, height: int = 32) -> tuple[int, int] | None:
    """Convert a click in either framebuffer panel into native pixel coordinates."""
    for x, y, w, h in (FINAL_RECT, RAW_RECT):
        if x <= px < x + w and y <= py < y + h:
            return min(width - 1, (px - x) * width // w), min(height - 1, (py - y) * height // h)
    return None


def heatmap(values: np.ndarray, low: float, high: float) -> np.ndarray:
    """Apply a tiny blue-white-red heatmap without adding a plotting dependency."""
    if high <= low:
        high = low + 1.0
    t = np.clip((values - low) / (high - low), 0.0, 1.0)
    red = np.where(t < 0.5, 32 + t * 446, 255)
    green = np.where(t < 0.5, 64 + t * 382, 255 - (t - 0.5) * 430)
    blue = np.where(t < 0.5, 255, 255 - (t - 0.5) * 446)
    return np.stack((red, green, blue), axis=-1).astype(np.uint8)


def _panel(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int], title: str) -> None:
    x, y, w, h = rect
    draw.rounded_rectangle((x - 8, y - 30, x + w + 8, y + h + 8), 8, fill=PANEL, outline=EDGE, width=1)
    draw.text((x, y - 25), title, font=font(14, True), fill=TEXT)


def _image(canvas: Image.Image, draw: ImageDraw.ImageDraw, rect, values: Image.Image | None, title: str) -> None:
    _panel(draw, rect, title)
    x, y, w, h = rect
    if values is None:
        draw.rectangle((x, y, x + w, y + h), fill=(7, 9, 13))
        draw.text((x + 15, y + h // 2 - 8), "panel hidden", font=font(14), fill=MUTED)
    else:
        canvas.paste(values.resize((w, h), Image.Resampling.NEAREST), (x, y))


def _outline(draw: ImageDraw.ImageDraw, rect, x: int, y: int, width: int, height: int) -> None:
    left, top, w, h = rect
    x0, x1 = left + x * w // width, left + (x + 1) * w // width - 1
    y0, y1 = top + y * h // height, top + (y + 1) * h // height - 1
    draw.rectangle((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)), outline=ACCENT, width=2)


def _number(value, digits: int = 4) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


def compose_dashboard(
    original: np.ndarray,
    result: dict,
    row: dict,
    config: ActivationConfig,
    selected: tuple[int, int] = (0, 0),
    options: dict | None = None,
    status: str = "",
) -> Image.Image:
    """Compose one synchronized dashboard frame entirely with Pillow."""
    options = options or {}
    show_original = options.get("original", True)
    show_hidden = options.get("hidden", True)
    show_provenance = options.get("provenance", True)
    auto = options.get("auto", False)
    hidden = np.asarray(result["hidden"])
    raw = np.asarray(result["raw"])
    prediction = np.asarray(result["prediction"])
    if hidden.shape != (config.prompt_tokens, config.hidden_size) or raw.shape != (config.height, config.width):
        raise ValueError(f"dashboard received hidden {hidden.shape} and raw {raw.shape}")

    low, high = (
        (float(hidden.min()), float(hidden.max()))
        if auto
        else (config.activation_mean - 3 * config.activation_std, config.activation_mean + 3 * config.activation_std)
    )
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "DOOM-GPT  |  ACTIVATION PROVENANCE", font=font(24, True), fill=TEXT)
    draw.text(
        (610, 20),
        "H = hidden_state[3] in R[4,768]  ->  P = flatten(H)[:2048]  ->  F = reshape(P,32,64)",
        font=font(13),
        fill=(188, 199, 216),
    )
    if status:
        draw.text((20, 51), status, font=font(12), fill=(99, 203, 139))

    original_image = Image.fromarray(original.astype(np.uint8), mode="RGB") if show_original else None
    target_image = Image.fromarray(np.clip(np.rint(result["target"] * 255), 0, 255).astype(np.uint8), mode="L").convert("RGB")
    final_image = Image.fromarray(loss_space_uint8(prediction), mode="L").convert("RGB")
    _image(canvas, draw, ORIGINAL_RECT, original_image, "1  ORIGINAL VIZDOOM FRAME")
    _image(canvas, draw, TARGET_RECT, target_image, "2  PREPROCESSED FRAME  [32,64]")
    _image(canvas, draw, FINAL_RECT, final_image, "3  FINAL ACTIVATION-ONLY DISPLAY  |  fixed scale + nearest upscale")

    hidden_rgb = heatmap(hidden, low, high)
    used = np.arange(config.prompt_tokens * config.hidden_size).reshape(hidden.shape) < config.width * config.height
    hidden_rgb[~used] = (hidden_rgb[~used].astype(np.float32) * 0.22).astype(np.uint8)
    hidden_image = Image.fromarray(hidden_rgb, mode="RGB") if show_hidden else None
    _image(canvas, draw, HIDDEN_RECT, hidden_image, f"4  SELECTED HIDDEN STATE  [4,768]  |  x=hidden dim, y=token  |  {'auto' if auto else 'fixed'} [{low:.2f},{high:.2f}]")
    for token in range(config.prompt_tokens):
        draw.text((18, HIDDEN_RECT[1] + token * HIDDEN_RECT[3] // 4 + 4), str(token), font=font(11), fill=MUTED)
    for dimension in (0, 192, 384, 576, 767):
        x = HIDDEN_RECT[0] + dimension * HIDDEN_RECT[2] // 768
        draw.text((x, HIDDEN_RECT[1] + HIDDEN_RECT[3] + 1), str(dimension), font=font(10), fill=MUTED)

    raw_image = Image.fromarray(heatmap(raw, low, high), mode="RGB")
    _image(canvas, draw, RAW_RECT, raw_image, "5  RAW ACTIVATION FRAMEBUFFER  |  exact flatten(H)[:2048] -> [32,64]")
    tokens = source_token_map(config.width, config.height, config.hidden_size)
    provenance = Image.fromarray(TOKEN_COLORS[tokens], mode="RGB") if show_provenance else None
    _image(canvas, draw, PROVENANCE_RECT, provenance, "6  PIXEL PROVENANCE / SOURCE TOKEN MAP")

    sx, sy = selected
    source = pixel_source(sx, sy, config.width, config.height, config.hidden_size)
    for rect in (FINAL_RECT, RAW_RECT, PROVENANCE_RECT):
        _outline(draw, rect, sx, sy, config.width, config.height)
    _outline(draw, HIDDEN_RECT, source["dimension"], source["token"], config.hidden_size, config.prompt_tokens)

    _panel(draw, METRICS_RECT, "TIMELINE / METRICS")
    action = row.get("action_label") or "+".join(row.get("pressed_keys", [])) or "idle"
    token_norms = row.get("per_token_prompt_norm") or []
    metric_lines = [
        f"frame {row.get('sequence', '--')}   tick {row.get('game_tick', '--')}",
        f"action: {action}",
        f"prompt norm: {_number(row.get('prompt_norm'), 2)}",
        "token norms: " + (" / ".join(f"{v:.1f}" for v in token_norms) if token_norms else "--"),
        f"reconstruction MSE: {_number(row.get('spatial_mse'), 6)}",
        f"renderer latency: {_number(row.get('renderer_only_ms'), 2)} ms",
        f"temporal error: {_number(row.get('temporal_error'), 6)}",
    ]
    for index, line in enumerate(metric_lines):
        draw.text((METRICS_RECT[0] + 6, METRICS_RECT[1] + index * 14), line, font=font(11), fill=TEXT if index < 2 else MUTED)

    _panel(draw, INFO_RECT, "PIXEL INSPECTOR  |  click final display or raw framebuffer")
    loss_value = float(prediction[sy, sx])
    display_value = float(np.clip((loss_value + 1.0) / 2.0, 0.0, 1.0))
    info_lines = [
        f"display pixel (x,y)       ({sx}, {sy})",
        f"flat activation index     {source['flat']}",
        f"source                    token {source['token']}, hidden dim {source['dimension']}",
        f"raw activation value      {float(raw[sy, sx]):.8f}",
        f"loss-space value          {loss_value:.8f}",
        f"display-normalized value  {display_value:.8f}",
        f"display uint8             {int(loss_space_uint8(np.asarray([[loss_value]]))[0, 0])}",
        "",
        "used pixels: token 0 = 768 | token 1 = 768 | token 2 = 512",
        "token 3 = 0 pixels / UNUSED (neutral); dim 512..767 of token 2 unused",
    ]
    for index, line in enumerate(info_lines):
        draw.text((INFO_RECT[0] + 6, INFO_RECT[1] + index * 19), line, font=font(12), fill=TEXT if index < 7 else MUTED)
    for token, color in enumerate(TOKEN_COLORS):
        x = INFO_RECT[0] + 8 + token * 117
        draw.rectangle((x, INFO_RECT[1] + 215, x + 13, INFO_RECT[1] + 228), fill=tuple(color))
        draw.text((x + 18, INFO_RECT[1] + 213), f"token {token}", font=font(11), fill=MUTED)

    footer = (
        "Live: W/S forward/back  A/D strafe  arrows turn  Space attack  Shift speed  Tab dashboard  C compare  R record  F12 screenshot  Esc quit"
        if options.get("live")
        else "Replay: Space play/pause  Left/Right step  +/- speed  J jump  O/H/P panels  T scale  S screenshot  G GIF  Esc quit"
    )
    draw.text((20, 872), footer, font=font(12), fill=MUTED)
    return canvas


class ReplayApp:
    """Replay sparse V1-C recordings while recomputing their exact hidden states."""

    def __init__(self, args: argparse.Namespace):
        import tkinter as tk
        from tkinter import simpledialog

        from PIL import ImageTk
        from activation_doom.live import activation_image, load_runtime, render_frame, resolve_device

        self.tk, self.simpledialog, self.ImageTk = tk, simpledialog, ImageTk
        requested = Path(args.recording)
        self.run_root = requested if (requested / "interactive").is_dir() else requested.parent
        self.session = requested / "interactive" if (requested / "interactive").is_dir() else requested
        config_path, metrics_path = self.run_root / "config.json", self.session / "metrics.jsonl"
        if not config_path.is_file() or not metrics_path.is_file():
            raise FileNotFoundError("recording must contain config.json and interactive/metrics.jsonl")
        self.files = sorted((self.session / "frames" / "original").glob("frame_*.png"))
        if not self.files:
            raise FileNotFoundError("recording has no saved original frames")
        self.metrics = {int(row["sequence"]): row for row in read_jsonl(metrics_path)}
        import json

        saved_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.device = resolve_device(args.device)
        self.encoder, self.model, self.config, _ = load_runtime(Path(saved_config["checkpoint"]), self.device)
        self.render_frame, self.activation_image = render_frame, activation_image
        self.output = Path(args.out) if args.out else Path("experiments") / "m7_dashboard" / time.strftime("%Y%m%d-%H%M%S")
        self.cache: dict[int, tuple[np.ndarray, dict, dict, str]] = {}
        self.options = {"original": True, "hidden": True, "provenance": True, "auto": False}
        self.selected = (0, 0)
        self.index, self.speed, self.playing = 0, 1.0, True
        self.timer = None
        self.message = "recomputing hidden states from the locked checkpoint"
        self.window = tk.Tk()
        self.window.title("DOOM-GPT Activation Dashboard — Replay")
        self.label = tk.Label(self.window, borderwidth=0)
        self.label.pack()
        self.photo = self.image = None
        self.window.bind("<KeyPress>", self.on_key)
        self.label.bind("<Button-1>", self.on_click)
        self.label.focus_set()

    @staticmethod
    def sequence(path: Path) -> int:
        return int(path.stem.rsplit("_", 1)[1])

    def frame(self, index: int):
        if index not in self.cache:
            path = self.files[index]
            original = np.asarray(Image.open(path).convert("RGB"))
            result = self.render_frame(original, self.encoder, self.model, self.config, self.device)
            sequence = self.sequence(path)
            saved = np.asarray(Image.open(self.session / "frames" / "activation" / path.name).convert("L"))
            current = np.asarray(self.activation_image(result["prediction"]))
            delta = np.abs(saved.astype(np.int16) - current.astype(np.int16))
            verified = "historical activation verified exact" if not delta.any() else f"WARNING: historical max uint8 delta {int(delta.max())}"
            self.cache[index] = original, result, self.metrics.get(sequence, {"sequence": sequence}), verified
        return self.cache[index]

    def show(self) -> None:
        original, result, row, verified = self.frame(self.index)
        status = f"REPLAY {self.index + 1}/{len(self.files)}  |  {self.speed:g}x  |  {verified}  |  {self.message}"
        self.image = compose_dashboard(original, result, row, self.config, self.selected, self.options, status)
        self.photo = self.ImageTk.PhotoImage(self.image)
        self.label.configure(image=self.photo)

    def delay(self) -> int:
        if self.index + 1 >= len(self.files):
            return 100
        ticks = self.sequence(self.files[self.index + 1]) - self.sequence(self.files[self.index])
        return max(1, int(ticks / TIC_RATE / self.speed * 1000))

    def advance(self) -> None:
        self.timer = None
        if not self.playing:
            return
        if self.index + 1 >= len(self.files):
            self.playing = False
            self.message = "end of replay"
            self.show()
            return
        self.index += 1
        self.show()
        self.schedule(self.delay())

    def schedule(self, delay: int) -> None:
        """Keep exactly one playback callback active."""
        if self.timer is not None:
            self.window.after_cancel(self.timer)
        self.timer = self.window.after(delay, self.advance)

    def cancel(self) -> None:
        if self.timer is not None:
            self.window.after_cancel(self.timer)
            self.timer = None

    def on_click(self, event) -> None:
        selected = pixel_from_point(event.x, event.y, self.config.width, self.config.height)
        if selected is not None:
            self.selected = selected
            self.show()
        self.label.focus_set()

    def screenshot(self) -> Path:
        path = self.output / "screenshots" / f"dashboard_{self.sequence(self.files[self.index]):06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)
        self.message = f"saved {path}"
        return path

    def export_gif(self) -> Path:
        start_sequence = self.sequence(self.files[self.index])
        indices = [i for i in range(self.index, len(self.files)) if self.sequence(self.files[i]) <= start_sequence + int(5 * TIC_RATE)]
        images, durations = [], []
        for position, index in enumerate(indices):
            original, result, row, verified = self.frame(index)
            frame = compose_dashboard(original, result, row, self.config, self.selected, self.options, verified)
            images.append(frame.resize((1280, 720), Image.Resampling.LANCZOS).quantize(colors=128))
            next_index = indices[min(position + 1, len(indices) - 1)]
            ticks = max(1, self.sequence(self.files[next_index]) - self.sequence(self.files[index]))
            durations.append(max(20, int(ticks / TIC_RATE / self.speed * 1000)))
        path = self.output / "exports" / f"dashboard_{start_sequence:06d}.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=2)
        self.message = f"saved {len(images)}-frame GIF to {path}"
        return path

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        if key == "space":
            self.playing = not self.playing
            self.message = "playing" if self.playing else "paused"
            if self.playing:
                self.schedule(1)
            else:
                self.cancel()
        elif key in {"right", "left"}:
            self.playing = False
            self.cancel()
            self.index = min(len(self.files) - 1, max(0, self.index + (1 if key == "right" else -1)))
        elif key in {"plus", "equal", "kp_add"}:
            self.speed = min(8.0, self.speed * 2)
            if self.playing:
                self.schedule(self.delay())
        elif key in {"minus", "kp_subtract"}:
            self.speed = max(0.25, self.speed / 2)
            if self.playing:
                self.schedule(self.delay())
        elif key in {"o", "h", "p"}:
            name = {"o": "original", "h": "hidden", "p": "provenance"}[key]
            self.options[name] = not self.options[name]
        elif key == "t":
            self.options["auto"] = not self.options["auto"]
        elif key == "j":
            target = self.simpledialog.askinteger("Jump", "Recorded sequence number:", minvalue=0)
            if target is not None:
                self.index = min(range(len(self.files)), key=lambda i: abs(self.sequence(self.files[i]) - target))
                self.playing = False
                self.cancel()
        elif key == "s":
            self.screenshot()
        elif key == "g":
            self.export_gif()
        elif key == "escape":
            self.window.destroy()
            return
        self.show()

    def run(self) -> None:
        self.show()
        self.schedule(self.delay())
        self.window.mainloop()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Explain DOOM-GPT frames through their exact activation provenance.")
    root.add_argument("command", choices=["replay"])
    root.add_argument("recording", nargs="?", default="experiments/m6_live/20260820-interactive-user")
    root.add_argument("--device", default="auto")
    root.add_argument("--out")
    return root


def main() -> None:
    ReplayApp(parser().parse_args()).run()


if __name__ == "__main__":
    main()
