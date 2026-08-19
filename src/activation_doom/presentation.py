from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from activation_doom.activation import ActivationConfig
from activation_doom.dashboard import ACCENT, BG, EDGE, MUTED, PANEL, TEXT, ReplayApp, font, heatmap
from activation_doom.experiment import loss_space_uint8


SIZE = (1920, 1080)
ORANGE = (246, 99, 62)
CYAN = (69, 196, 214)
GREEN = (83, 211, 143)


def card(draw: ImageDraw.ImageDraw, box, title: str, number: str) -> None:
    """Draw one of the three presentation stages."""
    draw.rounded_rectangle(box, 18, fill=PANEL, outline=EDGE, width=2)
    draw.text((box[0] + 28, box[1] + 22), number, font=font(17, True), fill=ORANGE)
    draw.text((box[0] + 75, box[1] + 18), title, font=font(24, True), fill=TEXT)


def arrow(draw: ImageDraw.ImageDraw, start, end, label: str, phase: float = 0.0) -> None:
    """Draw a directional flow arrow with one animated signal pulse."""
    draw.line((start, end), fill=(99, 113, 135), width=5)
    x1, y1 = end
    draw.polygon(((x1, y1), (x1 - 18, y1 - 11), (x1 - 18, y1 + 11)), fill=(99, 113, 135))
    px = int(start[0] + (end[0] - start[0]) * phase)
    py = int(start[1] + (end[1] - start[1]) * phase)
    draw.ellipse((px - 8, py - 8, px + 8, py + 8), fill=ORANGE)
    draw.text(((start[0] + end[0]) // 2 - 45, start[1] - 35), label, font=font(13, True), fill=MUTED)


def compose_presentation(
    original: np.ndarray,
    result: dict,
    row: dict,
    config: ActivationConfig,
    status: str = "",
) -> Image.Image:
    """Render a clean input -> DOOM-GPT -> activation-frame story."""
    hidden = np.asarray(result["hidden"])
    raw = np.asarray(result["raw"])
    prediction = np.asarray(result["prediction"])
    if hidden.shape != (4, 768) or raw.shape != (32, 64):
        raise ValueError(f"presentation expects hidden [4,768] and raw [32,64], got {hidden.shape} and {raw.shape}")
    low, high = config.activation_mean - 3 * config.activation_std, config.activation_mean + 3 * config.activation_std
    phase = (int(row.get("sequence", 0)) % 30) / 29.0
    canvas = Image.new("RGB", SIZE, BG)
    draw = ImageDraw.Draw(canvas)

    draw.text((55, 38), "HOW DOOM-GPT DRAWS WITH TRANSFORMER ACTIVATIONS", font=font(34, True), fill=TEXT)
    draw.text(
        (58, 94),
        "A real ViZDoom frame becomes a soft prompt for a frozen transformer. Its hidden-state values become the pixels you see.",
        font=font(18),
        fill=MUTED,
    )
    action = row.get("action_label") or "+".join(row.get("pressed_keys", [])) or "idle"
    draw.text(
        (1450, 55),
        f"FRAME {row.get('sequence', '--')}  |  TICK {row.get('game_tick', '--')}  |  {action.upper()}",
        font=font(15, True),
        fill=CYAN,
    )
    if status:
        draw.text((1450, 89), status, font=font(12), fill=GREEN)

    left, center, right = (50, 180, 550, 925), (650, 145, 1270, 950), (1370, 180, 1870, 925)
    card(draw, left, "REAL DOOM INPUT", "01")
    card(draw, center, "DOOM-GPT", "02")
    card(draw, right, "ACTIVATIONS AS FRAME", "03")
    arrow(draw, (565, 545), (635, 545), "ENCODE", phase)
    arrow(draw, (1285, 545), (1355, 545), "RESHAPE", phase)

    original_image = Image.fromarray(original.astype(np.uint8), mode="RGB").resize((430, 323), Image.Resampling.NEAREST)
    canvas.paste(original_image, (85, 275))
    draw.rectangle((84, 274, 516, 599), outline=(82, 93, 110), width=2)
    draw.text((85, 620), "ViZDoom RGB framebuffer  [240 x 320 x 3]", font=font(14), fill=TEXT)
    draw.line((300, 665, 300, 700), fill=(99, 113, 135), width=3)
    draw.polygon(((300, 710), (290, 696), (310, 696)), fill=(99, 113, 135))
    target = Image.fromarray(np.clip(np.rint(result["target"] * 255), 0, 255).astype(np.uint8), mode="L").convert("RGB")
    canvas.paste(target.resize((240, 120), Image.Resampling.NEAREST), (180, 720))
    draw.rectangle((179, 719, 421, 841), outline=(82, 93, 110), width=2)
    draw.text((197, 860), "grayscale + resize  ->  [32 x 64]", font=font(14), fill=MUTED)

    canvas.paste(target.resize((155, 78), Image.Resampling.NEAREST), (685, 230))
    draw.text((684, 320), "image input", font=font(12), fill=MUTED)
    draw.rounded_rectangle((875, 220, 1060, 320), 12, fill=(31, 40, 54), outline=CYAN, width=2)
    draw.text((905, 244), "IMAGE", font=font(17, True), fill=TEXT)
    draw.text((900, 275), "ENCODER", font=font(17, True), fill=TEXT)
    arrow(draw, (845, 270), (868, 270), "", phase)

    prompt_norms = row.get("per_token_prompt_norm") or [float(np.linalg.norm(token)) for token in result["prompt"]]
    max_prompt = max(prompt_norms) or 1.0
    for token, norm in enumerate(prompt_norms):
        x, y = 1110 + token * 34, 270
        radius = 10 + int(8 * norm / max_prompt)
        color = ORANGE if token < 3 else (77, 83, 94)
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=TEXT, width=1)
        draw.text((x - 4, 298), str(token), font=font(11), fill=MUTED)
    draw.text((1080, 220), "SOFT PROMPT", font=font(13, True), fill=TEXT)
    draw.text((1093, 320), "4 tokens x 768", font=font(11), fill=MUTED)

    draw.line((960, 360, 960, 395), fill=(99, 113, 135), width=3)
    draw.polygon(((960, 405), (950, 391), (970, 391)), fill=(99, 113, 135))
    draw.rounded_rectangle((680, 405, 1240, 725), 14, fill=(15, 20, 28), outline=(82, 93, 110), width=2)
    draw.text((705, 425), "FROZEN DISTILGPT2  |  HIDDEN STATE 3", font=font(17, True), fill=CYAN)
    draw.text((705, 458), "96 representative neurons of the actual 3,072 activations", font=font(12), fill=MUTED)
    dimensions = np.linspace(0, 767, 24, dtype=int)
    colors = heatmap(hidden[:, dimensions], low, high)
    for token in range(4):
        y = 520 + token * 46
        draw.text((705, y - 7), f"T{token}", font=font(12, True), fill=TEXT if token < 3 else MUTED)
        for column, dimension in enumerate(dimensions):
            x = 755 + column * 19
            color = tuple(int(value) for value in colors[token, column])
            if token == 3 or (token == 2 and dimension >= 512):
                color = tuple(int(value * 0.25) for value in color)
            draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
    draw.text((705, 692), "bright = positive   |   blue = negative   |   dimmed = not selected for pixels", font=font(11), fill=MUTED)

    full_heatmap = Image.fromarray(heatmap(hidden, low, high), mode="RGB").resize((520, 86), Image.Resampling.NEAREST)
    canvas.paste(full_heatmap, (700, 760))
    draw.rectangle((699, 759, 1221, 847), outline=(82, 93, 110), width=2)
    draw.text((700, 862), "H = hidden_state[3]  [4 x 768]  |  every color is a real activation value", font=font(12), fill=TEXT)
    draw.text((810, 907), "GPT WEIGHTS STAY FROZEN", font=font(15, True), fill=GREEN)

    raw_image = Image.fromarray(heatmap(raw, low, high), mode="RGB").resize((420, 160), Image.Resampling.NEAREST)
    canvas.paste(raw_image, (1410, 285))
    draw.rectangle((1409, 284, 1831, 446), outline=(82, 93, 110), width=2)
    draw.text((1410, 462), "flatten(H)[:2048]  ->  raw [32 x 64] framebuffer", font=font(13), fill=MUTED)
    draw.line((1620, 500, 1620, 535), fill=(99, 113, 135), width=3)
    draw.polygon(((1620, 545), (1610, 531), (1630, 531)), fill=(99, 113, 135))
    final = Image.fromarray(loss_space_uint8(prediction), mode="L").convert("RGB").resize((420, 315), Image.Resampling.NEAREST)
    canvas.paste(final, (1410, 560))
    draw.rectangle((1409, 559, 1831, 876), outline=ORANGE, width=3)
    draw.text((1410, 892), "fixed normalization + nearest upscale", font=font(13), fill=TEXT)

    draw.rounded_rectangle((50, 975, 1870, 1045), 14, fill=(23, 29, 39), outline=EDGE, width=2)
    draw.text((85, 991), "H = hidden_state[3]", font=font(18, True), fill=CYAN)
    draw.text((385, 991), "->", font=font(18, True), fill=ORANGE)
    draw.text((440, 991), "P = flatten(H)[:2048]", font=font(18, True), fill=CYAN)
    draw.text((810, 991), "->", font=font(18, True), fill=ORANGE)
    draw.text((865, 991), "F = reshape(P, 32, 64)", font=font(18, True), fill=CYAN)
    draw.text((1290, 991), "NO DECODER  •  NO GPT WEIGHT UPDATES", font=font(16, True), fill=GREEN)
    draw.text(
        (85, 1021),
        f"MSE {_number(row.get('spatial_mse'), 6)}   |   prompt norm {_number(row.get('prompt_norm'), 1)}   |   renderer {_number(row.get('renderer_only_ms'), 2)} ms",
        font=font(12),
        fill=MUTED,
    )
    return canvas


def _number(value, digits: int) -> str:
    return "--" if value is None else f"{float(value):.{digits}f}"


class PresentationApp(ReplayApp):
    """Run the verified replay as a clean presentation rather than a debugger."""

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        if not args.out:
            self.output = Path("experiments") / "m8_presentation" / time.strftime("%Y%m%d-%H%M%S")
        self.window.title("DOOM-GPT — Activation Flow Presentation")
        self.label.unbind("<Button-1>")

    def show(self) -> None:
        original, result, row, verified = self.frame(self.index)
        status = f"{self.index + 1}/{len(self.files)}  |  {self.speed:g}x  |  {verified}"
        self.image = compose_presentation(original, result, row, self.config, status)
        self.photo = self.ImageTk.PhotoImage(self.image.resize((1600, 900), Image.Resampling.LANCZOS))
        self.label.configure(image=self.photo)

    def screenshot(self) -> Path:
        path = self.output / "screenshots" / f"presentation_{self.sequence(self.files[self.index]):06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(path)
        self.message = f"saved {path}"
        return path

    def export_gif(self) -> Path:
        start = self.sequence(self.files[self.index])
        indices = [i for i in range(self.index, len(self.files)) if self.sequence(self.files[i]) <= start + 175]
        images, durations = [], []
        for position, index in enumerate(indices):
            original, result, row, verified = self.frame(index)
            image = compose_presentation(original, result, row, self.config, verified)
            images.append(image.resize((1280, 720), Image.Resampling.LANCZOS).quantize(colors=128))
            following = indices[min(position + 1, len(indices) - 1)]
            ticks = max(1, self.sequence(self.files[following]) - self.sequence(self.files[index]))
            durations.append(max(20, int(ticks / 35.0 / self.speed * 1000)))
        path = self.output / "exports" / f"presentation_{start:06d}.gif"
        path.parent.mkdir(parents=True, exist_ok=True)
        images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, disposal=2)
        self.message = f"saved {path}"
        return path

    def on_key(self, event) -> None:
        key = event.keysym.lower()
        if key == "space":
            self.playing = not self.playing
            self.schedule(1) if self.playing else self.cancel()
        elif key in {"right", "left"}:
            self.playing = False
            self.cancel()
            self.index = min(len(self.files) - 1, max(0, self.index + (1 if key == "right" else -1)))
        elif key in {"plus", "equal", "kp_add", "minus", "kp_subtract"}:
            self.speed = min(8.0, self.speed * 2) if key in {"plus", "equal", "kp_add"} else max(0.25, self.speed / 2)
            if self.playing:
                self.schedule(self.delay())
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
            self.cancel()
            self.window.destroy()
            return
        self.show()


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Present DOOM-GPT as an input-to-activation visual story.")
    root.add_argument("recording", nargs="?", default="experiments/m6_live/20260820-interactive-user")
    root.add_argument("--device", default="auto")
    root.add_argument("--out")
    return root


def main() -> None:
    PresentationApp(parser().parse_args()).run()


if __name__ == "__main__":
    main()
