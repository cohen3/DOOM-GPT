from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ActivationConfig:
    """Describe the direct hidden-state framebuffer and its fixed loss space."""

    model: str = "distilbert/distilgpt2"
    hidden_state_index: int = 3
    prompt_tokens: int = 4
    hidden_size: int = 768
    width: int = 64
    height: int = 32
    activation_mean: float | None = None
    activation_std: float | None = None

    def to_dict(self) -> dict:
        """Return this activation configuration as JSON-compatible values."""
        return asdict(self)


def activation_frame(hidden: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """Select and reshape the first activation values independently per sample."""
    squeeze = hidden.ndim < 3
    samples = hidden.reshape(1, -1) if squeeze else hidden.reshape(hidden.shape[0], -1)
    need = width * height
    if samples.shape[1] < need:
        raise ValueError(f"activation has {samples.shape[1]} values per sample, need {need}")
    frames = samples[:, :need].reshape(samples.shape[0], height, width)
    return frames[0] if squeeze else frames


def prediction_loss_space(frame: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Map raw activation values through one fixed global mean and standard deviation."""
    if std <= 0:
        raise ValueError("activation standard deviation must be positive")
    return (frame - mean) / std


def target_loss_space(target: torch.Tensor) -> torch.Tensor:
    """Map normalized grayscale targets from [0, 1] into the fixed [-1, 1] objective."""
    return target * 2.0 - 1.0


def predict_activation_frame(model, soft: torch.Tensor, layer: int, width: int, height: int) -> torch.Tensor:
    """Run soft prompts through a frozen transformer and return direct activation frames."""
    outputs = model(
        inputs_embeds=soft,
        attention_mask=torch.ones(soft.shape[:2], device=soft.device),
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    return activation_frame(outputs.hidden_states[layer], width, height)
