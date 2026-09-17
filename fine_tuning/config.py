"""Load the YAML configuration used by the training scripts."""

# ruff: noqa: EM101, EM102, T201, TRY003, TRY004

import argparse
import json
import typing
from pathlib import Path

import yaml

type Config = dict[str, typing.Any]

REQUIRED_KEYS: typing.Final = {
    "run": ("name", "seed"),
    "paths": ("train_dataset", "eval_dataset", "base_model", "runs", "adapters", "reports"),
    "model": ("name", "device", "dtype", "max_seq_length"),
    "lora": ("method", "r", "lora_alpha", "lora_dropout", "bias", "target_modules"),
    "training": (
        "learning_rate",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "num_train_epochs",
        "logging_steps",
        "eval_strategy",
        "save_strategy",
        "save_total_limit",
    ),
    "generation": ("do_sample", "max_new_tokens"),
}


def get_section(config: Config, name: str) -> Config:
    """Return one mapping from the configuration."""
    section: typing.Final = config.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"В конфигурации отсутствует раздел {name!r}")
    return section


def get_path(config: Config, name: str) -> Path:
    """Return one path from the paths section."""
    value: typing.Final = get_section(config, "paths").get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Некорректный путь paths.{name}")
    return Path(value)


def load_config(path: Path) -> Config:
    """Read YAML and check that the values required by the pipeline exist."""
    if not path.is_file():
        raise FileNotFoundError(f"Конфигурация не найдена: {path}")
    document: typing.Final = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("Конфигурация должна быть YAML-объектом")

    config: typing.Final[Config] = document
    for section_name, keys in REQUIRED_KEYS.items():
        section = get_section(config, section_name)
        missing = [key for key in keys if key not in section]
        if missing:
            raise ValueError(f"В разделе {section_name!r} нет ключей: {missing}")

    if get_section(config, "lora")["method"] != "lora":
        raise ValueError("Минимальная реализация поддерживает только LoRA")
    return config


def run_config_validation_cli() -> None:
    """Validate the configured YAML file from the command line."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    args: typing.Final = parser.parse_args()
    config: typing.Final = load_config(args.config)
    print(json.dumps(config, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_config_validation_cli()
