"""Baseline, assistant-only tokenization and LoRA training for the local model."""

import argparse
import json
import math
import sys
import time
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    set_seed,
)

from fine_tuning.config import FineTuningConfig, FineTuningConfigManager
from fine_tuning.dataset import FineTuningDatasetManager, ValidatedExample
from fine_tuning.environment import select_torch_device

IGNORE_INDEX: typing.Final = -100
EVALUATION_CRITERIA: typing.Final = (
    "Ответ на русском языке.",
    "Ответ краткий и по существу.",
    "Ответ соответствует правилам из эталона.",
    "Ответ не добавляет выдуманных условий.",
)


class TokenizedExample(typing.TypedDict):
    """Тензоры одного диалога до добавления padding."""

    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]


@dataclass(frozen=True, slots=True)
class MaskingSummary:
    """Количество prompt- и assistant-токенов одного примера."""

    example_id: str
    prompt_tokens: int
    assistant_tokens: int
    total_tokens: int


def apply_chat_template_to_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Применить chat template и извлечь плоский список input_ids."""
    tokenized: typing.Final = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    token_ids: typing.Final = (
        tokenized.get("input_ids") if isinstance(tokenized, Mapping) else tokenized
    )
    if not isinstance(token_ids, list) or not all(
        isinstance(token_id, int) for token_id in token_ids
    ):
        message: typing.Final = "Chat template вернул неожиданный формат input_ids"
        raise TypeError(message)
    return typing.cast("list[int]", token_ids)


class AssistantResponseDataset(Dataset[TokenizedExample]):
    """Токенизирует диалоги и оставляет в loss только ответ assistant."""

    def __init__(
        self,
        examples: list[ValidatedExample],
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> None:
        """Построить токены и проверить точную границу prompt/answer."""
        self.items: list[TokenizedExample] = []
        self.masking_summaries: list[MaskingSummary] = []
        for example in examples:
            item, summary = self._tokenize_example(example, tokenizer, max_seq_length)
            self.items.append(item)
            self.masking_summaries.append(summary)

    def __len__(self) -> int:
        """Вернуть число подготовленных примеров."""
        return len(self.items)

    def __getitem__(self, index: int) -> TokenizedExample:
        """Вернуть токены одного примера."""
        return self.items[index]

    @staticmethod
    def _tokenize_example(
        example: ValidatedExample,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_length: int,
    ) -> tuple[TokenizedExample, MaskingSummary]:
        """Применить chat template и замаскировать system/user часть."""
        prompt_messages: typing.Final = typing.cast("list[dict[str, str]]", example.messages[:-1])
        full_messages: typing.Final = typing.cast("list[dict[str, str]]", example.messages)
        prompt_ids: typing.Final = apply_chat_template_to_ids(
            tokenizer,
            prompt_messages,
            add_generation_prompt=True,
        )
        full_ids: typing.Final = apply_chat_template_to_ids(
            tokenizer,
            full_messages,
            add_generation_prompt=False,
        )
        if full_ids[: len(prompt_ids)] != prompt_ids:
            message = f"{example.example_id}: prompt не является префиксом полного диалога"
            raise ValueError(message)
        if len(full_ids) > max_seq_length:
            message = (
                f"{example.example_id}: {len(full_ids)} токенов превышают лимит {max_seq_length}"
            )
            raise ValueError(message)

        assistant_tokens: typing.Final = len(full_ids) - len(prompt_ids)
        if assistant_tokens <= 0:
            message = f"{example.example_id}: после masking не осталось assistant-токенов"
            raise ValueError(message)
        labels: typing.Final = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        if len(labels) != len(full_ids):
            message = f"{example.example_id}: длины input_ids и labels не совпадают"
            raise ValueError(message)
        return (
            {
                "input_ids": full_ids,
                "attention_mask": [1] * len(full_ids),
                "labels": labels,
            },
            MaskingSummary(
                example_id=example.example_id,
                prompt_tokens=len(prompt_ids),
                assistant_tokens=assistant_tokens,
                total_tokens=len(full_ids),
            ),
        )


class CausalLanguageModelCollator:
    """Добавляет padding к input_ids, attention_mask и labels."""

    def __init__(self, pad_token_id: int) -> None:
        """Сохранить ID padding-токена."""
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[TokenizedExample]) -> dict[str, torch.Tensor]:
        """Собрать список примеров в batch одинаковой длины."""
        max_length: typing.Final = max(len(feature["input_ids"]) for feature in features)
        input_ids: typing.Final = [
            self._pad(feature["input_ids"], max_length, self.pad_token_id) for feature in features
        ]
        attention_mask: typing.Final = [
            self._pad(feature["attention_mask"], max_length, 0) for feature in features
        ]
        labels: typing.Final = [
            self._pad(feature["labels"], max_length, IGNORE_INDEX) for feature in features
        ]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    @staticmethod
    def _pad(values: list[int], max_length: int, padding_value: int) -> list[int]:
        """Дополнить список справа до требуемой длины."""
        return values + [padding_value] * (max_length - len(values))


def load_local_model_and_tokenizer(
    config: FineTuningConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Загрузить локальные модель и tokenizer на выбранное устройство."""
    device: typing.Final = select_torch_device(config.model.device)
    tokenizer: typing.Final = typing.cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            config.paths.base_model,
            local_files_only=True,
            trust_remote_code=config.model.trust_remote_code,
        ),
    )
    if not tokenizer.chat_template:
        message = "У tokenizer отсутствует chat template"
        raise ValueError(message)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        message = "Не удалось определить pad_token_id"
        raise ValueError(message)

    loaded_model: typing.Final = typing.cast(
        "PreTrainedModel",
        AutoModelForCausalLM.from_pretrained(
            config.paths.base_model,
            local_files_only=True,
            trust_remote_code=config.model.trust_remote_code,
            dtype=torch.float32,
            attn_implementation="eager",
        ),
    )
    model: typing.Final = typing.cast(
        "PreTrainedModel", typing.cast("typing.Any", loaded_model).to(device)
    )
    return model, tokenizer, device


def evaluate_model_loss(
    model: PreTrainedModel,
    dataset: AssistantResponseDataset,
    collator: CausalLanguageModelCollator,
    batch_size: int,
    device: torch.device,
) -> float:
    """Посчитать средний loss тем же способом, что и Trainer.evaluate."""
    loader: typing.Final = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=collator
    )
    weighted_loss = 0.0
    evaluated_examples = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            device_batch = {name: tensor.to(device) for name, tensor in batch.items()}
            outputs = model(**device_batch)
            loss = outputs.loss
            if loss is None or not torch.isfinite(loss).item():
                message = "Модель вернула некорректный eval loss"
                raise RuntimeError(message)
            if not (device_batch["labels"][:, 1:] != IGNORE_INDEX).any().item():
                message = "В eval batch не найдено target-токенов"
                raise RuntimeError(message)
            batch_examples = device_batch["labels"].shape[0]
            weighted_loss += float(loss.item()) * batch_examples
            evaluated_examples += batch_examples
    if evaluated_examples == 0:
        message = "Eval dataset оказался пустым"
        raise RuntimeError(message)
    return weighted_loss / evaluated_examples


def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[ValidatedExample],
    config: FineTuningConfig,
    device: torch.device,
) -> list[dict[str, object]]:
    """Сгенерировать ответы на eval без передачи эталонного assistant."""
    generated_examples: typing.Final[list[dict[str, object]]] = []
    model.eval()
    for example in examples:
        prompt_messages = typing.cast("list[dict[str, str]]", example.messages[:-1])
        inputs = typing.cast(
            "dict[str, torch.Tensor]",
            tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ),
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
        with torch.inference_mode():
            generated = typing.cast("typing.Any", model).generate(
                **inputs,
                do_sample=config.generation.do_sample,
                max_new_tokens=config.generation.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        answer = typing.cast(
            "str",
            tokenizer.decode(
                generated[0, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            ),
        ).strip()
        if not answer:
            message = f"Baseline не вернул текст для {example.example_id}"
            raise RuntimeError(message)
        generated_examples.append(
            {
                "id": example.example_id,
                "system_prompt": example.messages[0]["content"],
                "user_prompt": example.messages[1]["content"],
                "reference_answer": example.messages[2]["content"],
                "generated_answer": answer,
                "manual_evaluation": {"passed": None, "reason": None},
            }
        )
    return generated_examples


def configure_lora_model(
    model: PreTrainedModel,
    config: FineTuningConfig,
) -> tuple[PreTrainedModel, int, int]:
    """Подключить LoRA и проверить число обучаемых и замороженных параметров."""
    module_names: typing.Final = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    missing_modules: typing.Final = set(config.lora.target_modules) - module_names
    if missing_modules:
        message = f"В модели нет LoRA target modules: {sorted(missing_modules)}"
        raise ValueError(message)
    lora_config: typing.Final = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.lora_alpha,
        lora_dropout=config.lora.lora_dropout,
        bias=typing.cast("typing.Any", config.lora.bias),
        target_modules=list(config.lora.target_modules),
    )
    lora_model: typing.Final = typing.cast("PreTrainedModel", get_peft_model(model, lora_config))
    total_parameters: typing.Final = sum(parameter.numel() for parameter in lora_model.parameters())
    trainable_parameters: typing.Final = sum(
        parameter.numel() for parameter in lora_model.parameters() if parameter.requires_grad
    )
    if trainable_parameters <= 0 or trainable_parameters >= total_parameters:
        message = "LoRA должна оставить часть параметров обучаемыми, а базовые веса замороженными"
        raise RuntimeError(message)
    return lora_model, trainable_parameters, total_parameters


def build_training_arguments(
    config: FineTuningConfig,
    output_dir: Path,
    device: torch.device,
) -> TrainingArguments:
    """Построить TrainingArguments только из проверенной конфигурации."""
    training: typing.Final = config.training
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=training.per_device_train_batch_size,
        per_device_eval_batch_size=training.per_device_eval_batch_size,
        gradient_accumulation_steps=training.gradient_accumulation_steps,
        num_train_epochs=training.num_train_epochs,
        learning_rate=training.learning_rate,
        lr_scheduler_type=training.lr_scheduler_type,
        optim=training.optimizer,
        weight_decay=training.weight_decay,
        max_grad_norm=training.max_grad_norm,
        warmup_steps=training.warmup_steps,
        logging_strategy=training.logging_strategy,
        logging_steps=training.logging_steps,
        logging_first_step=True,
        eval_strategy=training.eval_strategy,
        save_strategy=training.save_strategy,
        save_total_limit=training.save_total_limit,
        gradient_checkpointing=training.gradient_checkpointing,
        report_to=training.report_to,
        run_name=config.run.name,
        seed=config.run.seed,
        data_seed=config.run.seed,
        use_cpu=device.type == "cpu",
        dataloader_pin_memory=False,
        do_train=True,
        do_eval=True,
        use_cache=False,
    )


def write_json_report(path: Path, report: Mapping[str, object]) -> None:
    """Создать родительский каталог и атомарно записать JSON-отчёт."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: typing.Final = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(
        f"{json.dumps(report, ensure_ascii=False, indent=2)}\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def build_masking_report(
    train_dataset: AssistantResponseDataset,
    eval_dataset: AssistantResponseDataset,
) -> dict[str, object]:
    """Сформировать проверяемую сводку assistant-only masking."""
    summaries: typing.Final = train_dataset.masking_summaries + eval_dataset.masking_summaries
    assistant_counts: typing.Final = [summary.assistant_tokens for summary in summaries]
    return {
        "ignore_index": IGNORE_INDEX,
        "prompt_and_padding_excluded_from_loss": True,
        "assistant_tokens_in_loss": True,
        "examples_checked": len(summaries),
        "assistant_tokens": {"min": min(assistant_counts), "max": max(assistant_counts)},
        "first_example": {
            "id": summaries[0].example_id,
            "prompt_tokens": summaries[0].prompt_tokens,
            "assistant_tokens": summaries[0].assistant_tokens,
            "total_tokens": summaries[0].total_tokens,
        },
    }


def run_training(config_path: Path) -> dict[str, object]:
    """Выполнить baseline и LoRA-обучение, затем сохранить артефакты."""
    config: typing.Final = FineTuningConfigManager(config_path).load_and_validate()
    set_seed(config.run.seed)
    dataset_manager: typing.Final = FineTuningDatasetManager(
        train_path=config.paths.train_dataset,
        eval_path=config.paths.eval_dataset,
        model_path=config.paths.base_model,
        max_seq_length=config.model.max_seq_length,
    )
    splits: typing.Final = dataset_manager.load_and_validate()
    model, tokenizer, device = load_local_model_and_tokenizer(config)
    if tokenizer.pad_token_id is None:
        message = "pad_token_id необходим для collator"
        raise ValueError(message)
    train_dataset: typing.Final = AssistantResponseDataset(
        splits["train"], tokenizer, config.model.max_seq_length
    )
    eval_dataset: typing.Final = AssistantResponseDataset(
        splits["eval"], tokenizer, config.model.max_seq_length
    )
    collator: typing.Final = CausalLanguageModelCollator(tokenizer.pad_token_id)
    masking_report: typing.Final = build_masking_report(train_dataset, eval_dataset)

    baseline_eval_loss: typing.Final = evaluate_model_loss(
        model,
        eval_dataset,
        collator,
        config.training.per_device_eval_batch_size,
        device,
    )
    if not math.isfinite(baseline_eval_loss):
        message = "Baseline eval loss не является конечным числом"
        raise RuntimeError(message)
    baseline_examples: typing.Final = generate_answers(
        model, tokenizer, splits["eval"], config, device
    )
    baseline_report_path: typing.Final = config.paths.reports / "baseline_report.json"
    baseline_report: typing.Final = {
        "schema_version": 1,
        "stage": "baseline",
        "model": config.model.name,
        "revision": config.model.revision,
        "seed": config.run.seed,
        "device": device.type,
        "dtype": config.model.dtype,
        "eval_loss": baseline_eval_loss,
        "generation": {
            "do_sample": config.generation.do_sample,
            "max_new_tokens": config.generation.max_new_tokens,
        },
        "criteria": list(EVALUATION_CRITERIA),
        "examples": baseline_examples,
    }
    write_json_report(baseline_report_path, baseline_report)

    model, trainable_parameters, total_parameters = configure_lora_model(model, config)
    model.config.use_cache = False
    run_dir: typing.Final = config.paths.runs / config.run.name
    adapter_dir: typing.Final = config.paths.adapters / config.run.name
    run_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    trainer: typing.Final = Trainer(
        model=model,
        args=build_training_arguments(config, run_dir, device),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    started_at: typing.Final = time.perf_counter()
    train_result: typing.Final = trainer.train()
    training_seconds: typing.Final = time.perf_counter() - started_at
    if trainer.state.global_step <= 0:
        message = "Trainer не выполнил ни одного шага оптимизации"
        raise RuntimeError(message)
    tuned_eval_metrics: typing.Final = trainer.evaluate()
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    checkpoints: typing.Final = sorted(run_dir.glob("checkpoint-*"))
    if not checkpoints:
        message = f"Trainer не сохранил checkpoint в {run_dir}"
        raise RuntimeError(message)

    training_report: typing.Final = {
        "schema_version": 1,
        "stage": "training",
        "config": config.build_serializable_summary(),
        "device": device.type,
        "baseline_eval_loss": baseline_eval_loss,
        "tuned_eval_loss_before_reload": tuned_eval_metrics.get("eval_loss"),
        "train_metrics": train_result.metrics,
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "training_seconds": training_seconds,
        "trainable_parameters": trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_percentage": 100 * trainable_parameters / total_parameters,
        "masking": masking_report,
        "log_history": trainer.state.log_history,
        "checkpoints": [str(path) for path in checkpoints],
        "adapter_path": str(adapter_dir),
        "baseline_report_path": str(baseline_report_path),
    }
    write_json_report(config.paths.reports / "training_report.json", training_report)
    return training_report


def run_fine_tuning_cli() -> None:
    """Разобрать аргументы CLI, выполнить baseline и запустить LoRA."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    args: typing.Final = parser.parse_args()
    report: typing.Final = run_training(args.config)
    sys.stdout.write(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n")


if __name__ == "__main__":
    run_fine_tuning_cli()
