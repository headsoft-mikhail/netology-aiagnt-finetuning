"""Проверка окружения и локальной модели без обучения и сетевых запросов."""

import argparse
import importlib.metadata
import json
import platform
import sys
import typing
from pathlib import Path

import peft
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def select_torch_device(requested_device: str) -> torch.device:
    """Выбрать CPU или доступный GPU backend для значения auto/cpu/gpu."""
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.xpu.is_available():
        return torch.device("xpu")
    if requested_device == "gpu":
        message: typing.Final = "GPU недоступен. Повторите с --device cpu или --device auto."
        raise RuntimeError(message)
    return torch.device("cpu")


def inspect_environment(model_path: Path, requested_device: str) -> dict[str, object]:
    """Загрузить модель, проверить chat template, модули LoRA и короткую генерацию."""
    mps_available: typing.Final = torch.backends.mps.is_available()
    device: typing.Final = select_torch_device(requested_device)
    if not model_path.is_dir():
        message = f"Нет локальной модели: {model_path}. Выполните just download_model."
        raise FileNotFoundError(message)

    # float32 — консервативный вариант для первого небольшого LoRA-эксперимента.
    dtype: typing.Final = torch.float32
    tokenizer: typing.Final = typing.cast(
        "typing.Any",
        AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False),
    )
    if not tokenizer.chat_template:
        message = "У tokenizer нет chat template."
        raise ValueError(message)
    model: typing.Final = typing.cast(
        "typing.Any",
        AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            dtype=dtype,
            attn_implementation="eager",
        ),
    ).to(device)
    model.eval()

    # Проверяем совместимость с выбранными target_modules, не создавая адаптер.
    module_names: typing.Final = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing_modules: typing.Final = {"q_proj", "v_proj"} - module_names
    if missing_modules:
        message = f"В модели нет модулей LoRA: {sorted(missing_modules)}"
        raise ValueError(message)

    inputs = typing.cast(
        "dict[str, torch.Tensor]",
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "Ответь одним словом: привет!"}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ),
    )
    inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
    with torch.inference_mode():
        generated: typing.Final = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=8,
            pad_token_id=tokenizer.eos_token_id,
        )
    if device.type == "mps":
        torch.mps.synchronize()
    answer: typing.Final = typing.cast(
        "str",
        tokenizer.decode(generated[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True),
    )
    if not answer.strip():
        message = "Проверочная генерация не вернула текста."
        raise RuntimeError(message)

    # Проверка autograd на устройстве, без изменения весов модели.
    probe: typing.Final = torch.ones(2, device=device, requires_grad=True)
    probe.square().sum().backward()
    if probe.grad is None or not torch.isfinite(probe.grad).all().item():
        message = "Проверка autograd не пройдена."
        raise RuntimeError(message)
    cpu_available: typing.Final = torch.ones(1, device="cpu").item() == 1.0
    memory: typing.Final = psutil.virtual_memory()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "libraries": {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "peft", "accelerate", "huggingface-hub", "pyyaml")
        },
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_available,
        "xpu_available": torch.xpu.is_available(),
        "cpu_available": cpu_available,
        "selected_device": device.type,
        "selected_dtype": str(dtype),
        "system_memory_gib": round(memory.total / 1024**3, 2),
        "available_system_memory_gib": round(memory.available / 1024**3, 2),
        "mps_recommended_memory_gib": (
            round(torch.mps.recommended_max_memory() / 1024**3, 2) if mps_available else None
        ),
        "model_path": str(model_path.resolve()),
        "model_exists": True,
        "tokenizer_loaded": True,
        "model_loaded": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "lora_target_modules_found": ["q_proj", "v_proj"],
        "peft_available": hasattr(peft, "get_peft_model"),
        "generation_check": answer,
        "autograd_check": True,
        "training_performed": False,
    }


def run_environment_inspection_cli() -> None:
    """Разобрать аргументы CLI, проверить окружение и вывести JSON-сводку."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", type=Path, default=Path("data/models/base/Qwen2.5-0.5B-Instruct")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"), default="auto")
    args: typing.Final = parser.parse_args()
    report: typing.Final = inspect_environment(args.model_path, args.device)
    sys.stdout.write(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n")


if __name__ == "__main__":
    run_environment_inspection_cli()
