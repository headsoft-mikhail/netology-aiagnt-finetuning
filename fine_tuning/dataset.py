"""Управление train/eval-наборами и их проверка перед обучением."""

import argparse
import json
import re
import sys
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from transformers import AutoTokenizer

type JsonObject = dict[str, object]
type DialogueSignature = tuple[str, ...]

ALLOWED_ROLES: typing.Final = {"system", "user", "assistant"}
SECRET_PATTERNS: typing.Final = (
    re.compile(r"\b(?:sk|hf|ghp)_[A-Za-z0-9_-]{12,}\b"),
    re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|password|пароль)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}"),
)
EMAIL_PATTERN: typing.Final = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_PATTERN: typing.Final = re.compile(
    r"(?<!\d)(?:\+7|8)[\s()-]*\d{3}[\s()-]*\d{3}[\s-]*\d{2}[\s-]*\d{2}(?!\d)"
)
PLACEHOLDER_PHONES: typing.Final = {"88000000000"}


class ChatMessage(typing.TypedDict):
    """Одно сообщение диалога."""

    role: str
    content: str


class ChatTokenizer(typing.Protocol):
    """Часть интерфейса tokenizer, необходимая менеджеру данных."""

    chat_template: str | None

    def apply_chat_template(
        self,
        conversation: list[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> object:
        """Применить шаблон диалога и вернуть токены."""


class ValidationReport(typing.TypedDict):
    """Сводка успешной проверки train/eval."""

    valid: bool
    train_examples: int
    eval_examples: int
    system_prompt: str
    max_seq_length: int
    train_tokens: dict[str, int]
    eval_tokens: dict[str, int]
    id_overlap: int
    dialogue_overlap: int
    user_prompt_overlap: int


@dataclass(frozen=True, slots=True)
class ValidatedExample:
    """Проверенный пример с метаданными для поиска дубликатов."""

    example_id: str
    messages: list[ChatMessage]
    signature: DialogueSignature
    user_prompt: str
    system_prompt: str


class DatasetValidationError(ValueError):
    """Ошибка данных, при которой обучение запускать нельзя."""


class FineTuningDatasetManager:
    """Загружает, проверяет и описывает train/eval-наборы."""

    def __init__(
        self,
        train_path: Path,
        eval_path: Path,
        model_path: Path,
        max_seq_length: int,
    ) -> None:
        """Сохранить пути и максимальную длину для последующей проверки."""
        self.train_path = train_path
        self.eval_path = eval_path
        self.model_path = model_path
        self.max_seq_length = max_seq_length

    def validate_and_summarize(self) -> ValidationReport:
        """Загрузить оба набора, выполнить все проверки и вернуть сводку."""
        validated_splits: typing.Final = self.load_and_validate()
        tokenizer: typing.Final = self._load_tokenizer()
        lengths: typing.Final = {
            split: [self._count_message_tokens(tokenizer, example.messages) for example in examples]
            for split, examples in validated_splits.items()
        }
        return {
            "valid": True,
            "train_examples": len(validated_splits["train"]),
            "eval_examples": len(validated_splits["eval"]),
            "system_prompt": validated_splits["train"][0].system_prompt,
            "max_seq_length": self.max_seq_length,
            "train_tokens": {"min": min(lengths["train"]), "max": max(lengths["train"])},
            "eval_tokens": {"min": min(lengths["eval"]), "max": max(lengths["eval"])},
            "id_overlap": 0,
            "dialogue_overlap": 0,
            "user_prompt_overlap": 0,
        }

    def load_and_validate(self) -> dict[str, list[ValidatedExample]]:
        """Загрузить train/eval, выполнить проверки и вернуть готовые примеры."""
        self._validate_settings()
        raw_splits: typing.Final = {
            "train": self._load_jsonl(self.train_path),
            "eval": self._load_jsonl(self.eval_path),
        }
        validated_splits: typing.Final = {
            split: self._validate_split(split, examples) for split, examples in raw_splits.items()
        }
        self._validate_relationship_between_splits(validated_splits)
        tokenizer: typing.Final = self._load_tokenizer()
        lengths: typing.Final = {
            split: [self._count_message_tokens(tokenizer, example.messages) for example in examples]
            for split, examples in validated_splits.items()
        }
        self._reject_overlong_examples(validated_splits, lengths)
        return validated_splits

    def _validate_settings(self) -> None:
        """Проверить настройки, необходимые для валидации."""
        if self.max_seq_length <= 0:
            message = "max_seq_length должен быть положительным"
            raise DatasetValidationError(message)
        if not self.model_path.is_dir():
            message = f"Локальная модель не найдена: {self.model_path}"
            raise DatasetValidationError(message)

    def _load_jsonl(self, path: Path) -> list[JsonObject]:
        """Прочитать JSONL и указать номер строки при ошибке."""
        if not path.is_file():
            message = f"Файл не найден: {path}"
            raise DatasetValidationError(message)

        examples: typing.Final[list[JsonObject]] = []
        with path.open(encoding="utf-8") as source:
            for line_number, raw_line in enumerate(source, start=1):
                if not raw_line.strip():
                    message = f"{path}:{line_number}: пустая строка"
                    raise DatasetValidationError(message)
                try:
                    value: object = json.loads(raw_line)
                except json.JSONDecodeError as error:
                    message = f"{path}:{line_number}: некорректный JSON: {error.msg}"
                    raise DatasetValidationError(message) from error
                if not isinstance(value, dict):
                    message = f"{path}:{line_number}: ожидается JSON-объект"
                    raise DatasetValidationError(message)
                examples.append(value)
        if not examples:
            message = f"Файл пуст: {path}"
            raise DatasetValidationError(message)
        return examples

    def _validate_split(self, split: str, examples: list[JsonObject]) -> list[ValidatedExample]:
        """Проверить примеры одного набора и отклонить дубликаты."""
        validated_examples: typing.Final[list[ValidatedExample]] = []
        ids: typing.Final[set[str]] = set()
        signatures: typing.Final[set[DialogueSignature]] = set()
        user_prompts: typing.Final[set[str]] = set()
        for line_number, example in enumerate(examples, start=1):
            validated = self._validate_example(example, f"{split}:{line_number}")
            if validated.example_id in ids:
                message = f"{split}: повторяющийся id {validated.example_id!r}"
                raise DatasetValidationError(message)
            if validated.signature in signatures:
                message = f"{split}: дубликат диалога у {validated.example_id!r}"
                raise DatasetValidationError(message)
            if validated.user_prompt in user_prompts:
                message = f"{split}: повтор запроса у {validated.example_id!r}"
                raise DatasetValidationError(message)
            ids.add(validated.example_id)
            signatures.add(validated.signature)
            user_prompts.add(validated.user_prompt)
            validated_examples.append(validated)
        return validated_examples

    def _validate_example(self, example: JsonObject, source: str) -> ValidatedExample:
        """Проверить поля одного примера и привести сообщения к ожидаемому типу."""
        example_id: typing.Final = example.get("id")
        if not isinstance(example_id, str) or not example_id.strip():
            message = f"{source}: id должен быть непустой строкой"
            raise DatasetValidationError(message)

        raw_messages: typing.Final = example.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            message = f"{source}/{example_id}: messages должен быть непустым списком"
            raise DatasetValidationError(message)

        messages: typing.Final = [
            self._validate_message(raw_message, f"{source}/{example_id}/messages[{index}]")
            for index, raw_message in enumerate(raw_messages)
        ]
        roles: typing.Final = [message["role"] for message in messages]
        expected_roles: typing.Final = ["system", "user", "assistant"]
        if roles != expected_roles:
            message = f"{source}/{example_id}: ожидается порядок system → user → assistant"
            raise DatasetValidationError(message)

        signature: typing.Final = tuple(
            value for message in messages for value in (message["role"], message["content"])
        )
        return ValidatedExample(
            example_id=example_id,
            messages=messages,
            signature=signature,
            user_prompt=messages[1]["content"].casefold(),
            system_prompt=messages[0]["content"],
        )

    def _validate_message(self, raw_message: object, location: str) -> ChatMessage:
        """Проверить роль, текст и чувствительные данные одного сообщения."""
        if not isinstance(raw_message, dict):
            message = f"{location}: сообщение должно быть объектом"
            raise DatasetValidationError(message)
        role: typing.Final = raw_message.get("role")
        content: typing.Final = raw_message.get("content")
        if not isinstance(role, str) or role not in ALLOWED_ROLES:
            message = f"{location}: недопустимая роль {role!r}"
            raise DatasetValidationError(message)
        if not isinstance(content, str) or not content.strip():
            message = f"{location}: content должен быть непустой строкой"
            raise DatasetValidationError(message)
        sensitive_kind: typing.Final = self._detect_sensitive_data_kind(content)
        if sensitive_kind:
            message = f"{location}: найден {sensitive_kind}"
            raise DatasetValidationError(message)
        return {"role": role, "content": content.strip()}

    def _detect_sensitive_data_kind(self, text: str) -> str | None:
        """Найти секрет или персональный контакт и вернуть его вид."""
        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                return "похожее на секрет значение"
        if EMAIL_PATTERN.search(text):
            return "адрес электронной почты"
        for match in PHONE_PATTERN.finditer(text):
            digits = "".join(character for character in match.group() if character.isdigit())
            if digits not in PLACEHOLDER_PHONES:
                return "номер телефона"
        return None

    def _validate_relationship_between_splits(
        self,
        splits: dict[str, list[ValidatedExample]],
    ) -> str:
        """Отклонить пересечения train/eval и вернуть единый system prompt."""
        train: typing.Final = splits["train"]
        evaluation: typing.Final = splits["eval"]
        train_ids: typing.Final = {example.example_id for example in train}
        eval_ids: typing.Final = {example.example_id for example in evaluation}
        repeated_ids: typing.Final = train_ids & eval_ids
        if repeated_ids:
            message = f"train/eval пересекаются по id: {sorted(repeated_ids)}"
            raise DatasetValidationError(message)

        train_signatures: typing.Final = {example.signature for example in train}
        eval_signatures: typing.Final = {example.signature for example in evaluation}
        if train_signatures & eval_signatures:
            message = "train/eval содержат одинаковые диалоги"
            raise DatasetValidationError(message)

        train_prompts: typing.Final = {example.user_prompt for example in train}
        eval_prompts: typing.Final = {example.user_prompt for example in evaluation}
        if train_prompts & eval_prompts:
            message = "train/eval содержат одинаковые пользовательские запросы"
            raise DatasetValidationError(message)

        system_prompts: typing.Final = {
            example.system_prompt for examples in splits.values() for example in examples
        }
        if len(system_prompts) != 1:
            message = "во всех примерах должен использоваться один system prompt"
            raise DatasetValidationError(message)
        return next(iter(system_prompts))

    def _load_tokenizer(self) -> ChatTokenizer:
        """Загрузить tokenizer только из локального каталога модели."""
        tokenizer: typing.Final = typing.cast(
            "ChatTokenizer",
            AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=False,
            ),
        )
        if not tokenizer.chat_template:
            message: typing.Final = "у tokenizer отсутствует chat template"
            raise DatasetValidationError(message)
        return tokenizer

    def _count_message_tokens(self, tokenizer: ChatTokenizer, messages: list[ChatMessage]) -> int:
        """Применить chat template и посчитать токены полного диалога."""
        tokenized: typing.Final = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
        token_ids: typing.Final = (
            tokenized.get("input_ids") if isinstance(tokenized, Mapping) else tokenized
        )
        if not isinstance(token_ids, list) or not all(
            isinstance(token_id, int) for token_id in token_ids
        ):
            message: typing.Final = "tokenizer вернул неожиданный формат input_ids"
            raise DatasetValidationError(message)
        return len(token_ids)

    def _reject_overlong_examples(
        self,
        splits: dict[str, list[ValidatedExample]],
        lengths: dict[str, list[int]],
    ) -> None:
        """Отклонить примеры, превышающие настроенный лимит токенов."""
        for split, examples in splits.items():
            for example, length in zip(examples, lengths[split], strict=True):
                if length > self.max_seq_length:
                    message = (
                        f"{split}/{example.example_id}: {length} токенов превышают "
                        f"лимит {self.max_seq_length}"
                    )
                    raise DatasetValidationError(message)


def run_dataset_validation_cli() -> None:
    """Разобрать аргументы CLI, проверить данные и вывести JSON-сводку."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/fine_tuning/train.jsonl"))
    parser.add_argument("--eval", type=Path, default=Path("data/fine_tuning/eval.jsonl"))
    parser.add_argument(
        "--model-path", type=Path, default=Path("data/models/base/Qwen2.5-0.5B-Instruct")
    )
    parser.add_argument("--max-seq-length", type=int, default=512)
    args: typing.Final = parser.parse_args()
    manager: typing.Final = FineTuningDatasetManager(
        train_path=args.train,
        eval_path=args.eval,
        model_path=args.model_path,
        max_seq_length=args.max_seq_length,
    )
    report: typing.Final = manager.validate_and_summarize()
    sys.stdout.write(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n")


if __name__ == "__main__":
    run_dataset_validation_cli()
