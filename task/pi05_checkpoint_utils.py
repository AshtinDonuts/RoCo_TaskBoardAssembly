"""Dependency-free checkpoint inspection shared by pi0.5 tools and tests."""
from __future__ import annotations

from pathlib import Path


def checkpoint_kind(checkpoint: str | Path) -> str:
    """Return ``full`` or ``lora`` from checkpoint file markers."""
    root = Path(checkpoint)
    if (root / "adapter_config.json").is_file():
        return "lora"
    if (root / "model.safetensors").is_file():
        return "full"
    raise FileNotFoundError(
        f"{root} is neither a full checkpoint (model.safetensors) "
        "nor a PEFT adapter (adapter_config.json)"
    )
