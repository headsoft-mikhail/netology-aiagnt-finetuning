"""Run baseline generation and train a LoRA adapter."""

# ruff: noqa: D105, D107, EM101, EM102, T201, TRY003

import argparse
import json
import math
import time
import typing
from collections.abc import Mapping
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

from fine_tuning.config import Config, get_path, get_section, load_config
from fine_tuning.dataset import Example, get_messages, load_datasets
from fine_tuning.environment import select_device

IGNORE_INDEX: typing.Final = -100
CRITERIA: typing.Final = [
    "Ответ на русском языке.",
    "Ответ краткий и по существу.",
    "Ответ соответствует правилам из эталона.",
    "Ответ не добавляет выдуманных условий.",
]


class ChatDataset(Dataset[dict[str, list[int]]]):
    """Store tokenized chats for Trainer."""

    def __init__(self, items: list[dict[str, list[int]]]) -> None:
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.items[index]


class ChatCollator:
    """Pad input IDs, masks and labels to the longest item in a batch."""

    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, items: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        """Pad one batch and convert it to tensors."""
        max_length: typing.Final = max(len(item["input_ids"]) for item in items)

        def pad(values: list[int], value: int) -> list[int]:
            return values + [value] * (max_length - len(values))

        return {
            "input_ids": torch.tensor(
                [pad(item["input_ids"], self.pad_token_id) for item in items]
            ),
            "attention_mask": torch.tensor([pad(item["attention_mask"], 0) for item in items]),
            "labels": torch.tensor([pad(item["labels"], IGNORE_INDEX) for item in items]),
        }


def token_ids(
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Apply the model chat template and return a flat list of token IDs."""
    result: typing.Final = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    ids: typing.Final = result.get("input_ids") if isinstance(result, Mapping) else result
    if not isinstance(ids, list):
        raise TypeError("Tokenizer вернул неожиданный формат input_ids")
    return typing.cast("list[int]", ids)


def tokenize_examples(
    examples: list[Example],
    tokenizer: PreTrainedTokenizerBase,
) -> ChatDataset:
    """Tokenize examples and exclude system and user tokens from loss."""
    items: typing.Final[list[dict[str, list[int]]]] = []
    for example in examples:
        messages = get_messages(example)
        prompt = token_ids(tokenizer, messages[:-1], add_generation_prompt=True)
        full = token_ids(tokenizer, messages, add_generation_prompt=False)
        if full[: len(prompt)] != prompt or len(full) == len(prompt):
            raise ValueError(f"Не удалось отделить ответ assistant у {example['id']}")
        items.append(
            {
                "input_ids": full,
                "attention_mask": [1] * len(full),
                "labels": [IGNORE_INDEX] * len(prompt) + full[len(prompt) :],
            }
        )
    return ChatDataset(items)


def load_model_and_tokenizer(
    config: Config,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    """Load the local base model and tokenizer."""
    model_config: typing.Final = get_section(config, "model")
    model_path: typing.Final = get_path(config, "base_model")
    device: typing.Final = select_device(str(model_config["device"]))
    tokenizer: typing.Final = typing.cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        ),
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    loaded_model: typing.Final = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=bool(model_config.get("trust_remote_code", False)),
        dtype=torch.float32,
        attn_implementation="eager",
    )
    model: typing.Final = typing.cast(
        "PreTrainedModel", typing.cast("typing.Any", loaded_model).to(device)
    )
    return model, tokenizer, device


def evaluate_loss(
    model: PreTrainedModel,
    dataset: ChatDataset,
    collator: ChatCollator,
    batch_size: int,
    device: torch.device,
) -> float:
    """Calculate mean loss on the eval split."""
    losses: typing.Final[list[float]] = []
    model.eval()
    with torch.inference_mode():
        for batch in DataLoader(dataset, batch_size=batch_size, collate_fn=collator):
            outputs = model(**{name: value.to(device) for name, value in batch.items()})
            losses.append(float(outputs.loss.item()))
    return sum(losses) / len(losses)


def generate_answers(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    examples: list[Example],
    generation: Config,
    device: torch.device,
) -> list[dict[str, object]]:
    """Generate answers without passing the reference assistant response."""
    results: typing.Final[list[dict[str, object]]] = []
    model.eval()
    for example in examples:
        messages = get_messages(example)
        inputs = typing.cast(
            "dict[str, torch.Tensor]",
            tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ),
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            generated = typing.cast("typing.Any", model).generate(
                **inputs,
                do_sample=bool(generation["do_sample"]),
                max_new_tokens=int(generation["max_new_tokens"]),
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
        results.append(
            {
                "id": example["id"],
                "system_prompt": messages[0]["content"],
                "user_prompt": messages[-2]["content"],
                "reference_answer": messages[-1]["content"],
                "generated_answer": answer,
            }
        )
    return results


def write_json(path: Path, value: object) -> None:
    """Write one UTF-8 JSON report."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def create_training_arguments(
    config: Config, output_dir: Path, device: torch.device
) -> TrainingArguments:
    """Build the small set of Trainer arguments used by this project."""
    training: typing.Final = get_section(config, "training")
    run: typing.Final = get_section(config, "run")
    return TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training["gradient_accumulation_steps"]),
        num_train_epochs=float(training["num_train_epochs"]),
        learning_rate=float(training["learning_rate"]),
        logging_steps=int(training["logging_steps"]),
        eval_strategy=str(training["eval_strategy"]),
        save_strategy=str(training["save_strategy"]),
        save_total_limit=int(training["save_total_limit"]),
        report_to="none",
        run_name=str(run["name"]),
        seed=int(run["seed"]),
        use_cpu=device.type == "cpu",
        dataloader_pin_memory=False,
    )


def run_training(config_path: Path) -> dict[str, object]:
    """Run baseline, LoRA training and artifact saving."""
    config: typing.Final = load_config(config_path)
    run: typing.Final = get_section(config, "run")
    training: typing.Final = get_section(config, "training")
    generation: typing.Final = get_section(config, "generation")
    set_seed(int(run["seed"]))

    model, tokenizer, device = load_model_and_tokenizer(config)
    train_examples, eval_examples = load_datasets(config, tokenizer)
    train_dataset: typing.Final = tokenize_examples(train_examples, tokenizer)
    eval_dataset: typing.Final = tokenize_examples(eval_examples, tokenizer)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer не содержит pad_token_id")
    collator: typing.Final = ChatCollator(tokenizer.pad_token_id)

    baseline_loss: typing.Final = evaluate_loss(
        model,
        eval_dataset,
        collator,
        int(training["per_device_eval_batch_size"]),
        device,
    )
    reports_dir: typing.Final = get_path(config, "reports")
    baseline_report: typing.Final = {
        "stage": "baseline",
        "model": get_section(config, "model")["name"],
        "eval_loss": baseline_loss,
        "perplexity": math.exp(baseline_loss),
        "generation": generation,
        "criteria": CRITERIA,
        "examples": generate_answers(model, tokenizer, eval_examples, generation, device),
    }
    write_json(reports_dir / "baseline_report.json", baseline_report)

    lora: typing.Final = get_section(config, "lora")
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora["r"]),
            lora_alpha=int(lora["lora_alpha"]),
            lora_dropout=float(lora["lora_dropout"]),
            bias=lora["bias"],
            target_modules=typing.cast("list[str]", lora["target_modules"]),
        ),
    )
    run_dir: typing.Final = get_path(config, "runs") / str(run["name"])
    adapter_dir: typing.Final = get_path(config, "adapters") / str(run["name"])
    trainer: typing.Final = Trainer(
        model=model,
        args=create_training_arguments(config, run_dir, device),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    started: typing.Final = time.perf_counter()
    train_result: typing.Final = trainer.train()
    training_seconds: typing.Final = time.perf_counter() - started
    tuned_metrics: typing.Final = trainer.evaluate()
    model.save_pretrained(str(adapter_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(adapter_dir))

    checkpoints: typing.Final = sorted(run_dir.glob("checkpoint-*"))
    if not checkpoints:
        raise RuntimeError(f"Trainer не сохранил checkpoint в {run_dir}")
    trainable: typing.Final = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total: typing.Final = sum(parameter.numel() for parameter in model.parameters())
    report: typing.Final = {
        "stage": "training",
        "run_name": run["name"],
        "device": device.type,
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "training_seconds": training_seconds,
        "train_loss": train_result.metrics["train_loss"],
        "eval_loss": tuned_metrics["eval_loss"],
        "learning_rate": training["learning_rate"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "log_history": trainer.state.log_history,
        "checkpoints": [str(path) for path in checkpoints],
        "adapter_path": str(adapter_dir),
    }
    write_json(reports_dir / "training_report.json", report)
    return report


def run_fine_tuning_cli() -> None:
    """Run training from the command line."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    args: typing.Final = parser.parse_args()
    print(json.dumps(run_training(args.config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run_fine_tuning_cli()
