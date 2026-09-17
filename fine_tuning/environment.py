"""Print the environment selected for local fine-tuning."""

# ruff: noqa: EM101, EM102, T201, TRY003

import argparse
import importlib.metadata
import json
import platform
import typing
from pathlib import Path

import psutil
import torch

from fine_tuning.config import Config, get_path, get_section, load_config


def xpu_is_available() -> bool:
    """Return whether PyTorch exposes an available Intel XPU device."""
    return hasattr(torch, "xpu") and torch.xpu.is_available()


def select_device(requested: str) -> torch.device:
    """Select CPU or the first available GPU backend."""
    available: typing.Final = {
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
        "xpu": xpu_is_available(),
    }
    if requested == "cpu":
        return torch.device("cpu")
    if requested in available:
        if not available[requested]:
            raise RuntimeError(f"Устройство {requested} недоступно")
        return torch.device(requested)
    if requested not in {"auto", "gpu"}:
        raise ValueError("device должен быть auto, gpu, cpu, cuda, mps или xpu")
    for name, is_available in available.items():
        if is_available:
            return torch.device(name)
    if requested == "gpu":
        raise RuntimeError("GPU недоступен; используйте device=cpu")
    return torch.device("cpu")


def inspect_environment(config: Config, requested_device: str | None = None) -> dict[str, object]:
    """Collect the environment values required by the assignment."""
    model: typing.Final = get_section(config, "model")
    model_path: typing.Final = get_path(config, "base_model")
    device: typing.Final = select_device(requested_device or str(model["device"]))
    return {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("transformers", "peft", "accelerate", "pyyaml")
        },
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "xpu_available": xpu_is_available(),
        "cpu_available": True,
        "selected_device": device.type,
        "selected_dtype": model["dtype"],
        "system_memory_gib": round(psutil.virtual_memory().total / 1024**3, 1),
        "model_exists": (model_path / "model.safetensors").is_file(),
        "tokenizer_exists": (model_path / "tokenizer.json").is_file(),
    }


def run_environment_inspection_cli() -> None:
    """Print the configured local environment."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    parser.add_argument("--device", default=None)
    args: typing.Final = parser.parse_args()
    report: typing.Final = inspect_environment(load_config(args.config), args.device)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_environment_inspection_cli()
