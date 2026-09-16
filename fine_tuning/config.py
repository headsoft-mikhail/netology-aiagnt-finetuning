"""Загрузка и проверка YAML-конфигурации fine-tuning."""

import argparse
import dataclasses
import json
import sys
import typing
from collections.abc import Mapping
from pathlib import Path

import yaml

from fine_tuning.dataset import FineTuningDatasetManager, ValidationReport

type ConfigSection = Mapping[str, object]


@dataclasses.dataclass(frozen=True, slots=True)
class RunSettings:
    """Идентификатор и seed одного запуска."""

    name: str
    seed: int


@dataclasses.dataclass(frozen=True, slots=True)
class PathSettings:
    """Пути ко входным данным и каталогам артефактов."""

    train_dataset: Path
    eval_dataset: Path
    base_model: Path
    runs: Path
    adapters: Path
    reports: Path


@dataclasses.dataclass(frozen=True, slots=True)
class ModelSettings:
    """Параметры базовой модели и её загрузки."""

    name: str
    revision: str
    device: str
    dtype: str
    max_seq_length: int
    trust_remote_code: bool


@dataclasses.dataclass(frozen=True, slots=True)
class LoraSettings:
    """Параметры LoRA-адаптера."""

    method: str
    r: int
    lora_alpha: int
    lora_dropout: float
    bias: str
    target_modules: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingSettings:
    """Параметры Trainer для минимального учебного запуска."""

    learning_rate: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    eval_strategy: str
    save_strategy: str
    save_total_limit: int
    gradient_checkpointing: bool


@dataclasses.dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Параметры одинаковой генерации до и после обучения."""

    do_sample: bool
    max_new_tokens: int


@dataclasses.dataclass(frozen=True, slots=True)
class FineTuningConfig:
    """Полная проверенная конфигурация эксперимента."""

    run: RunSettings
    paths: PathSettings
    model: ModelSettings
    lora: LoraSettings
    training: TrainingSettings
    generation: GenerationSettings

    def build_serializable_summary(self) -> dict[str, object]:
        """Преобразовать настройки в словарь, пригодный для JSON-отчёта."""
        return typing.cast("dict[str, object]", _convert_paths_to_strings(dataclasses.asdict(self)))


class ConfigValidationError(ValueError):
    """Ошибка структуры или значения YAML-конфигурации."""


class FineTuningConfigManager:
    """Загружает YAML и возвращает типизированную конфигурацию."""

    def __init__(self, config_path: Path) -> None:
        """Сохранить путь к YAML для последующей загрузки."""
        self.config_path = config_path
        self.project_root = config_path.resolve().parent.parent

    def load_and_validate(self) -> FineTuningConfig:
        """Прочитать YAML, проверить все секции и вернуть настройки."""
        document: typing.Final = self._read_yaml_mapping()
        config: typing.Final = FineTuningConfig(
            run=self._build_run_settings(self._require_section(document, "run")),
            paths=self._build_path_settings(self._require_section(document, "paths")),
            model=self._build_model_settings(self._require_section(document, "model")),
            lora=self._build_lora_settings(self._require_section(document, "lora")),
            training=self._build_training_settings(self._require_section(document, "training")),
            generation=self._build_generation_settings(
                self._require_section(document, "generation")
            ),
        )
        self._validate_relationships(config)
        return config

    def _read_yaml_mapping(self) -> ConfigSection:
        """Прочитать корневой YAML-объект без неявного сетевого доступа."""
        if not self.config_path.is_file():
            message = f"Файл конфигурации не найден: {self.config_path}"
            raise ConfigValidationError(message)
        try:
            with self.config_path.open(encoding="utf-8") as source:
                document: typing.Final[object] = yaml.safe_load(source)
        except yaml.YAMLError as error:
            message = f"Некорректный YAML в {self.config_path}: {error}"
            raise ConfigValidationError(message) from error
        if not isinstance(document, Mapping):
            message = "Корнем конфигурации должен быть YAML-объект"
            raise ConfigValidationError(message)
        return document

    def _build_run_settings(self, section: ConfigSection) -> RunSettings:
        """Проверить секцию run и построить её настройки."""
        return RunSettings(
            name=self._require_string(section, "name", "run"),
            seed=self._require_positive_int(section, "seed", "run", allow_zero=True),
        )

    def _build_path_settings(self, section: ConfigSection) -> PathSettings:
        """Проверить секцию paths и разрешить пути от корня проекта."""
        return PathSettings(
            train_dataset=self._require_path(section, "train_dataset"),
            eval_dataset=self._require_path(section, "eval_dataset"),
            base_model=self._require_path(section, "base_model"),
            runs=self._require_path(section, "runs"),
            adapters=self._require_path(section, "adapters"),
            reports=self._require_path(section, "reports"),
        )

    def _build_model_settings(self, section: ConfigSection) -> ModelSettings:
        """Проверить секцию model и построить её настройки."""
        device: typing.Final = self._require_string(section, "device", "model")
        dtype: typing.Final = self._require_string(section, "dtype", "model")
        self._require_choice(device, "model.device", {"auto", "cpu", "gpu"})
        self._require_choice(dtype, "model.dtype", {"float32"})
        return ModelSettings(
            name=self._require_string(section, "name", "model"),
            revision=self._require_string(section, "revision", "model"),
            device=device,
            dtype=dtype,
            max_seq_length=self._require_positive_int(section, "max_seq_length", "model"),
            trust_remote_code=self._require_bool(section, "trust_remote_code", "model"),
        )

    def _build_lora_settings(self, section: ConfigSection) -> LoraSettings:
        """Проверить секцию lora и построить её настройки."""
        method: typing.Final = self._require_string(section, "method", "lora")
        bias: typing.Final = self._require_string(section, "bias", "lora")
        self._require_choice(method, "lora.method", {"lora"})
        self._require_choice(bias, "lora.bias", {"none", "all", "lora_only"})
        target_modules: typing.Final = self._require_string_tuple(section, "target_modules", "lora")
        if len(set(target_modules)) != len(target_modules):
            message: typing.Final = "lora.target_modules содержит повторяющиеся значения"
            raise ConfigValidationError(message)
        return LoraSettings(
            method=method,
            r=self._require_positive_int(section, "r", "lora"),
            lora_alpha=self._require_positive_int(section, "lora_alpha", "lora"),
            lora_dropout=self._require_ratio(section, "lora_dropout", "lora"),
            bias=bias,
            target_modules=target_modules,
        )

    def _build_training_settings(self, section: ConfigSection) -> TrainingSettings:
        """Проверить секцию training и построить её настройки."""
        eval_strategy: typing.Final = self._require_string(section, "eval_strategy", "training")
        save_strategy: typing.Final = self._require_string(section, "save_strategy", "training")
        allowed_strategies: typing.Final = {"no", "steps", "epoch"}
        self._require_choice(eval_strategy, "training.eval_strategy", allowed_strategies)
        self._require_choice(save_strategy, "training.save_strategy", allowed_strategies)
        return TrainingSettings(
            learning_rate=self._require_positive_float(section, "learning_rate", "training"),
            per_device_train_batch_size=self._require_positive_int(
                section, "per_device_train_batch_size", "training"
            ),
            per_device_eval_batch_size=self._require_positive_int(
                section, "per_device_eval_batch_size", "training"
            ),
            gradient_accumulation_steps=self._require_positive_int(
                section, "gradient_accumulation_steps", "training"
            ),
            num_train_epochs=self._require_positive_int(section, "num_train_epochs", "training"),
            warmup_ratio=self._require_ratio(section, "warmup_ratio", "training"),
            logging_steps=self._require_positive_int(section, "logging_steps", "training"),
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=self._require_positive_int(section, "save_total_limit", "training"),
            gradient_checkpointing=self._require_bool(
                section, "gradient_checkpointing", "training"
            ),
        )

    def _build_generation_settings(self, section: ConfigSection) -> GenerationSettings:
        """Проверить секцию generation и построить её настройки."""
        return GenerationSettings(
            do_sample=self._require_bool(section, "do_sample", "generation"),
            max_new_tokens=self._require_positive_int(section, "max_new_tokens", "generation"),
        )

    def _validate_relationships(self, config: FineTuningConfig) -> None:
        """Проверить связи между секциями и существование входных путей."""
        missing_inputs: typing.Final = [
            path
            for path in (
                config.paths.train_dataset,
                config.paths.eval_dataset,
                config.paths.base_model,
            )
            if not path.exists()
        ]
        if missing_inputs:
            message = f"Не найдены входные пути: {[str(path) for path in missing_inputs]}"
            raise ConfigValidationError(message)
        if config.training.eval_strategy != config.training.save_strategy:
            message = "training.eval_strategy и training.save_strategy должны совпадать"
            raise ConfigValidationError(message)

    def _require_path(self, section: ConfigSection, key: str) -> Path:
        """Прочитать непустой путь и разрешить его относительно корня проекта."""
        raw_path: typing.Final = self._require_string(section, key, "paths")
        path: typing.Final = Path(raw_path).expanduser()
        return path if path.is_absolute() else self.project_root / path

    @staticmethod
    def _require_section(document: ConfigSection, key: str) -> ConfigSection:
        """Получить обязательную секцию конфигурации."""
        section: typing.Final = document.get(key)
        if not isinstance(section, Mapping):
            message: typing.Final = f"Секция {key!r} должна быть YAML-объектом"
            raise ConfigValidationError(message)
        return section

    @staticmethod
    def _require_string(section: ConfigSection, key: str, section_name: str) -> str:
        """Получить обязательную непустую строку."""
        value: typing.Final = section.get(key)
        if not isinstance(value, str) or not value.strip():
            message: typing.Final = f"{section_name}.{key} должен быть непустой строкой"
            raise ConfigValidationError(message)
        return value.strip()

    @staticmethod
    def _require_positive_int(
        section: ConfigSection,
        key: str,
        section_name: str,
        *,
        allow_zero: bool = False,
    ) -> int:
        """Получить целое число больше нуля либо неотрицательный seed."""
        value: typing.Final = section.get(key)
        minimum: typing.Final = 0 if allow_zero else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            condition: typing.Final = "неотрицательным" if allow_zero else "положительным"
            message: typing.Final = f"{section_name}.{key} должен быть {condition} целым числом"
            raise ConfigValidationError(message)
        return value

    @staticmethod
    def _require_positive_float(section: ConfigSection, key: str, section_name: str) -> float:
        """Получить положительное вещественное число."""
        value: typing.Final = section.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
            message: typing.Final = f"{section_name}.{key} должен быть положительным числом"
            raise ConfigValidationError(message)
        return float(value)

    @staticmethod
    def _require_ratio(section: ConfigSection, key: str, section_name: str) -> float:
        """Получить числовое значение из диапазона от нуля до единицы."""
        value: typing.Final = section.get(key)
        if isinstance(value, bool) or not isinstance(value, int | float) or not 0 <= value <= 1:
            message: typing.Final = f"{section_name}.{key} должен быть числом от 0 до 1"
            raise ConfigValidationError(message)
        return float(value)

    @staticmethod
    def _require_bool(section: ConfigSection, key: str, section_name: str) -> bool:
        """Получить обязательное логическое значение."""
        value: typing.Final = section.get(key)
        if not isinstance(value, bool):
            message: typing.Final = f"{section_name}.{key} должен быть true или false"
            raise ConfigValidationError(message)
        return value

    @staticmethod
    def _require_string_tuple(
        section: ConfigSection,
        key: str,
        section_name: str,
    ) -> tuple[str, ...]:
        """Получить непустой список непустых строк."""
        value: typing.Final = section.get(key)
        if (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, str) or not item.strip() for item in value)
        ):
            message: typing.Final = f"{section_name}.{key} должен быть непустым списком строк"
            raise ConfigValidationError(message)
        return tuple(item.strip() for item in value if isinstance(item, str))

    @staticmethod
    def _require_choice(value: str, field: str, allowed: set[str]) -> None:
        """Проверить строку по набору допустимых значений."""
        if value not in allowed:
            message: typing.Final = (
                f"{field}: ожидается одно из значений {sorted(allowed)}, получено {value!r}"
            )
            raise ConfigValidationError(message)


def _convert_paths_to_strings(value: object) -> object:
    """Рекурсивно заменить Path на строки для JSON-сериализации."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _convert_paths_to_strings(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_convert_paths_to_strings(item) for item in value]
    return value


def validate_configured_datasets(config: FineTuningConfig) -> ValidationReport:
    """Проверить train/eval по путям и лимиту из загруженной конфигурации."""
    return FineTuningDatasetManager(
        train_path=config.paths.train_dataset,
        eval_path=config.paths.eval_dataset,
        model_path=config.paths.base_model,
        max_seq_length=config.model.max_seq_length,
    ).validate_and_summarize()


def run_config_validation_cli() -> None:
    """Разобрать аргументы CLI и вывести проверенную конфигурацию с данными."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    args: typing.Final = parser.parse_args()
    config: typing.Final = FineTuningConfigManager(args.config).load_and_validate()
    report: typing.Final = {
        "valid": True,
        "config_path": str(args.config.resolve()),
        "config": config.build_serializable_summary(),
        "dataset": validate_configured_datasets(config),
    }
    sys.stdout.write(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n")


if __name__ == "__main__":
    run_config_validation_cli()
