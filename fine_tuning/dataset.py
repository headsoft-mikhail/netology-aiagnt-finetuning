"""Load and validate the train and eval JSONL files."""

# ruff: noqa: C901, EM101, EM102, T201, TRY003

import argparse
import json
import re
import typing
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from fine_tuning.config import Config, get_path, get_section, load_config

type Message = dict[str, str]
type Example = dict[str, typing.Any]

ALLOWED_ROLES: typing.Final = {"system", "user", "assistant"}
SENSITIVE_DATA: typing.Final = re.compile(
    r"(?i)(?:\b(?:sk|hf|ghp)_[A-Za-z0-9_-]{12,}\b|"
    r"\b(?:api[_ -]?key|access[_ -]?token|password|пароль)\s*[:=]\s*\S+|"
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b|"
    r"(?<!\d)(?:\+7|8)[\s()-]*\d{3}[\s()-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d))"
)


def get_messages(example: Example) -> list[Message]:
    """Return already validated chat messages."""
    messages: typing.Final = example["messages"]
    if not isinstance(messages, list):
        raise TypeError("messages должен быть списком")
    return messages


def read_jsonl(path: Path) -> list[Example]:
    """Read JSON objects from a non-empty JSONL file."""
    if not path.is_file():
        raise FileNotFoundError(f"Датасет не найден: {path}")

    examples: typing.Final[list[Example]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"{path}:{line_number}: пустая строка")
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: некорректный JSON") from error
        if not isinstance(value, dict):
            raise TypeError(f"{path}:{line_number}: ожидается JSON-объект")
        examples.append(value)
    if not examples:
        raise ValueError(f"Датасет пуст: {path}")
    return examples


def validate_examples(
    examples: list[Example],
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_seq_length: int,
) -> set[str]:
    """Validate IDs, messages, sensitive data and token length."""
    ids: typing.Final[set[str]] = set()
    dialogues: typing.Final[set[str]] = set()
    for line_number, example in enumerate(examples, start=1):
        example_id = example.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            raise ValueError(f"{split}:{line_number}: id должен быть непустой строкой")
        if example_id in ids:
            raise ValueError(f"{split}: повторяется id {example_id!r}")

        raw_messages = example.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError(f"{split}/{example_id}: messages должен быть непустым списком")
        messages: list[Message] = []
        for index, message in enumerate(raw_messages):
            if not isinstance(message, dict):
                raise TypeError(f"{split}/{example_id}/messages[{index}]: ожидается объект")
            role = message.get("role")
            content = message.get("content")
            if role not in ALLOWED_ROLES or not isinstance(role, str):
                raise ValueError(f"{split}/{example_id}: недопустимая роль {role!r}")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{split}/{example_id}: найдено пустое сообщение")
            if SENSITIVE_DATA.search(content) and "8-800-000-00-00" not in content:
                raise ValueError(f"{split}/{example_id}: найдены секрет или персональные данные")
            messages.append({"role": role, "content": content})

        roles = [message["role"] for message in messages]
        if "user" not in roles or roles[-1] != "assistant":
            raise ValueError(f"{split}/{example_id}: нужны запрос user и целевой ответ assistant")
        tokenized = tokenizer.apply_chat_template(messages, tokenize=True)
        token_ids = tokenized.get("input_ids") if isinstance(tokenized, Mapping) else tokenized
        if not isinstance(token_ids, list) or len(token_ids) > max_seq_length:
            raise ValueError(f"{split}/{example_id}: превышен max_seq_length={max_seq_length}")

        signature = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        if signature in dialogues:
            raise ValueError(f"{split}/{example_id}: повторяется диалог")
        ids.add(example_id)
        dialogues.add(signature)
        example["messages"] = messages
    return ids


def load_datasets(
    config: Config, tokenizer: PreTrainedTokenizerBase
) -> tuple[list[Example], list[Example]]:
    """Load both datasets and ensure that their IDs do not overlap."""
    train: typing.Final = read_jsonl(get_path(config, "train_dataset"))
    evaluation: typing.Final = read_jsonl(get_path(config, "eval_dataset"))
    max_length: typing.Final = int(get_section(config, "model")["max_seq_length"])
    train_ids: typing.Final = validate_examples(train, "train", tokenizer, max_length)
    eval_ids: typing.Final = validate_examples(evaluation, "eval", tokenizer, max_length)
    overlap: typing.Final = train_ids & eval_ids
    if overlap:
        raise ValueError(f"Train и eval пересекаются по id: {sorted(overlap)}")
    return train, evaluation


def run_dataset_validation_cli() -> None:
    """Validate the configured datasets from the command line."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    args: typing.Final = parser.parse_args()
    config: typing.Final = load_config(args.config)
    model: typing.Final = get_section(config, "model")
    tokenizer: typing.Final = typing.cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            get_path(config, "base_model"),
            local_files_only=True,
            trust_remote_code=bool(model.get("trust_remote_code", False)),
        ),
    )
    train, evaluation = load_datasets(config, tokenizer)
    print(json.dumps({"train_examples": len(train), "eval_examples": len(evaluation)}, indent=2))


if __name__ == "__main__":
    run_dataset_validation_cli()
