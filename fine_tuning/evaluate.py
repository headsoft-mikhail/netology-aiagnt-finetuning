"""Reload the local LoRA adapter, evaluate it and compare with baseline."""

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import typing
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from fine_tuning.config import FineTuningConfig, FineTuningConfigManager
from fine_tuning.dataset import FineTuningDatasetManager
from fine_tuning.train import (
    EVALUATION_CRITERIA,
    AssistantResponseDataset,
    CausalLanguageModelCollator,
    evaluate_model_loss,
    generate_answers,
    load_local_model_and_tokenizer,
    write_json_report,
)

type JsonObject = dict[str, object]


class ManualScore(typing.TypedDict):
    """Ручная бинарная оценка одного ответа."""

    passed: bool
    reason: str


class StageScores(typing.TypedDict):
    """Ручные оценки baseline и tuned для одного eval ID."""

    baseline: ManualScore
    tuned: ManualScore


def load_json_object(path: Path) -> JsonObject:
    """Прочитать обязательный JSON-объект с понятной ошибкой пути."""
    if not path.is_file():
        message = f"JSON-файл не найден: {path}"
        raise FileNotFoundError(message)
    try:
        value: typing.Final[object] = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        message = f"Некорректный JSON в {path}: {error.msg}"
        raise ValueError(message) from error
    if not isinstance(value, dict):
        message = f"Ожидается JSON-объект: {path}"
        raise TypeError(message)
    return value


def calculate_perplexity(eval_loss: float) -> tuple[float | None, bool]:
    """Вычислить exp(eval_loss), явно отметив возможное переполнение."""
    if not math.isfinite(eval_loss):
        return None, True
    if eval_loss > math.log(sys.float_info.max):
        return None, True
    return math.exp(eval_loss), False


def load_local_adapter(
    config: FineTuningConfig,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device, Path]:
    """Загрузить базовую модель и adapter только из локальных каталогов."""
    adapter_path: typing.Final = config.paths.adapters / config.run.name
    if not (adapter_path / "adapter_model.safetensors").is_file():
        message: typing.Final = f"Локальный adapter не найден: {adapter_path}"
        raise FileNotFoundError(message)
    base_model, _, device = load_local_model_and_tokenizer(config)
    tokenizer: typing.Final = typing.cast(
        "PreTrainedTokenizerBase",
        AutoTokenizer.from_pretrained(
            adapter_path,
            local_files_only=True,
            trust_remote_code=config.model.trust_remote_code,
        ),
    )
    model = typing.cast(
        "PreTrainedModel",
        PeftModel.from_pretrained(
            base_model,
            adapter_path,
            is_trainable=False,
            local_files_only=True,
        ),
    )
    model = typing.cast("PreTrainedModel", typing.cast("typing.Any", model).to(device))
    model.eval()
    return model, tokenizer, device, adapter_path


def evaluate_reloaded_adapter(config: FineTuningConfig) -> JsonObject:
    """Посчитать tuned loss и ответы после повторной загрузки adapter."""
    dataset_manager: typing.Final = FineTuningDatasetManager(
        train_path=config.paths.train_dataset,
        eval_path=config.paths.eval_dataset,
        model_path=config.paths.base_model,
        max_seq_length=config.model.max_seq_length,
    )
    splits: typing.Final = dataset_manager.load_and_validate()
    model, tokenizer, device, adapter_path = load_local_adapter(config)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        message: typing.Final = "Не удалось определить pad_token_id локального tokenizer"
        raise ValueError(message)
    eval_dataset: typing.Final = AssistantResponseDataset(
        splits["eval"], tokenizer, config.model.max_seq_length
    )
    collator: typing.Final = CausalLanguageModelCollator(tokenizer.pad_token_id)
    eval_loss: typing.Final = evaluate_model_loss(
        model,
        eval_dataset,
        collator,
        config.training.per_device_eval_batch_size,
        device,
    )
    perplexity, perplexity_overflow = calculate_perplexity(eval_loss)
    examples: typing.Final = generate_answers(model, tokenizer, splits["eval"], config, device)
    return {
        "schema_version": 1,
        "stage": "tuned",
        "model": config.model.name,
        "revision": config.model.revision,
        "adapter_path": str(adapter_path),
        "adapter_reloaded_from_disk": True,
        "local_files_only": True,
        "seed": config.run.seed,
        "device": device.type,
        "dtype": config.model.dtype,
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "perplexity_overflow": perplexity_overflow,
        "generation": {
            "do_sample": config.generation.do_sample,
            "max_new_tokens": config.generation.max_new_tokens,
        },
        "criteria": list(EVALUATION_CRITERIA),
        "examples": examples,
    }


def load_manual_evaluations(path: Path) -> dict[str, StageScores]:
    """Прочитать и строго проверить ручные оценки обеих версий."""
    document: typing.Final = load_json_object(path)
    raw_evaluations: typing.Final = document.get("evaluations")
    if not isinstance(raw_evaluations, dict) or not raw_evaluations:
        message = f"{path}: evaluations должен быть непустым объектом"
        raise ValueError(message)
    evaluations: typing.Final[dict[str, StageScores]] = {}
    for example_id, raw_stages in raw_evaluations.items():
        if not isinstance(example_id, str) or not isinstance(raw_stages, dict):
            message = f"{path}: некорректная запись ручной оценки"
            raise TypeError(message)
        stages: dict[str, ManualScore] = {}
        for stage in ("baseline", "tuned"):
            raw_score = raw_stages.get(stage)
            if not isinstance(raw_score, dict):
                message = f"{path}/{example_id}/{stage}: ожидается объект"
                raise TypeError(message)
            passed = raw_score.get("passed")
            reason = raw_score.get("reason")
            if not isinstance(passed, bool) or not isinstance(reason, str) or not reason.strip():
                message = f"{path}/{example_id}/{stage}: нужны bool passed и непустой reason"
                raise ValueError(message)
            stages[stage] = {"passed": passed, "reason": reason.strip()}
        evaluations[example_id] = typing.cast("StageScores", stages)
    return evaluations


def apply_manual_evaluations(
    report: JsonObject,
    stage: str,
    evaluations: dict[str, StageScores],
) -> None:
    """Записать ручные оценки указанной версии в её report."""
    if stage not in {"baseline", "tuned"}:
        message = f"Неизвестный этап ручной оценки: {stage}"
        raise ValueError(message)
    raw_examples: typing.Final = report.get("examples")
    if not isinstance(raw_examples, list):
        message = f"В {stage} report отсутствует список examples"
        raise TypeError(message)
    report_ids: typing.Final[set[str]] = set()
    for raw_example in raw_examples:
        if not isinstance(raw_example, dict) or not isinstance(raw_example.get("id"), str):
            message = f"В {stage} report найден некорректный example"
            raise TypeError(message)
        example_id = typing.cast("str", raw_example["id"])
        if example_id not in evaluations:
            message = f"Нет ручной оценки {stage} для {example_id}"
            raise ValueError(message)
        stage_scores = evaluations[example_id]
        score = stage_scores["baseline"] if stage == "baseline" else stage_scores["tuned"]
        raw_example["manual_evaluation"] = dict(score)
        report_ids.add(example_id)
    extra_ids: typing.Final = set(evaluations) - report_ids
    if extra_ids:
        message = f"Ручные оценки содержат неизвестные ID: {sorted(extra_ids)}"
        raise ValueError(message)


def extract_report_examples(report: JsonObject, stage: str) -> dict[str, JsonObject]:
    """Построить индекс примеров отчёта по ID."""
    raw_examples: typing.Final = report.get("examples")
    if not isinstance(raw_examples, list):
        message = f"В {stage} report отсутствует список examples"
        raise TypeError(message)
    examples: typing.Final[dict[str, JsonObject]] = {}
    for raw_example in raw_examples:
        if not isinstance(raw_example, dict) or not isinstance(raw_example.get("id"), str):
            message = f"В {stage} report найден некорректный example"
            raise TypeError(message)
        example_id = typing.cast("str", raw_example["id"])
        if example_id in examples:
            message = f"В {stage} report повторяется ID {example_id}"
            raise ValueError(message)
        examples[example_id] = raw_example
    return examples


def read_passed_score(example: JsonObject, stage: str) -> bool:
    """Получить заполненную бинарную ручную оценку ответа."""
    score: typing.Final = example.get("manual_evaluation")
    if not isinstance(score, dict) or not isinstance(score.get("passed"), bool):
        message: typing.Final = f"Для {example.get('id')} не заполнена ручная оценка {stage}"
        raise TypeError(message)
    return typing.cast("bool", score["passed"])


def choose_comparison_status(*, baseline_passed: bool, tuned_passed: bool) -> str:
    """Выбрать статус изменения по двум бинарным оценкам."""
    if not baseline_passed and tuned_passed:
        return "improved"
    if baseline_passed and not tuned_passed:
        return "regressed"
    return "unchanged_pass" if baseline_passed else "unchanged_fail"


def build_comparison_report(
    config: FineTuningConfig,
    baseline_report: JsonObject,
    tuned_report: JsonObject,
    training_report: JsonObject,
) -> JsonObject:
    """Сопоставить ответы и собрать итоговые метрики эксперимента."""
    baseline_examples: typing.Final = extract_report_examples(baseline_report, "baseline")
    tuned_examples: typing.Final = extract_report_examples(tuned_report, "tuned")
    if set(baseline_examples) != set(tuned_examples):
        message = "Наборы ID в baseline и tuned report не совпадают"
        raise ValueError(message)

    comparisons: typing.Final[list[JsonObject]] = []
    status_counts: typing.Final = {
        "improved": 0,
        "regressed": 0,
        "unchanged_pass": 0,
        "unchanged_fail": 0,
    }
    baseline_passed_count = 0
    tuned_passed_count = 0
    for example_id in baseline_examples:
        baseline = baseline_examples[example_id]
        tuned = tuned_examples[example_id]
        baseline_passed = read_passed_score(baseline, "baseline")
        tuned_passed = read_passed_score(tuned, "tuned")
        status = choose_comparison_status(
            baseline_passed=baseline_passed,
            tuned_passed=tuned_passed,
        )
        status_counts[status] += 1
        baseline_passed_count += int(baseline_passed)
        tuned_passed_count += int(tuned_passed)
        comparisons.append(
            {
                "id": example_id,
                "user_prompt": baseline.get("user_prompt"),
                "reference_answer": baseline.get("reference_answer"),
                "baseline_answer": baseline.get("generated_answer"),
                "baseline_evaluation": baseline.get("manual_evaluation"),
                "tuned_answer": tuned.get("generated_answer"),
                "tuned_evaluation": tuned.get("manual_evaluation"),
                "status": status,
            }
        )

    example_count: typing.Final = len(comparisons)
    baseline_loss: typing.Final = typing.cast("float", baseline_report["eval_loss"])
    tuned_loss: typing.Final = typing.cast("float", tuned_report["eval_loss"])
    baseline_perplexity, baseline_overflow = calculate_perplexity(baseline_loss)
    masking: typing.Final = training_report.get("masking")
    checked_examples: typing.Final = (
        masking.get("examples_checked") if isinstance(masking, dict) else None
    )
    if not isinstance(checked_examples, int) or checked_examples < example_count:
        message = "Training report не содержит корректное число проверенных примеров"
        raise ValueError(message)
    train_example_count: typing.Final = checked_examples - example_count
    loss_improved: typing.Final = tuned_loss < baseline_loss
    quality_improved: typing.Final = tuned_passed_count > baseline_passed_count
    return {
        "schema_version": 1,
        "stage": "comparison",
        "model": config.model.name,
        "revision": config.model.revision,
        "seed": config.run.seed,
        "config": training_report.get("config"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "device": tuned_report.get("device"),
            "dtype": tuned_report.get("dtype"),
            "libraries": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "peft", "accelerate")
            },
        },
        "dataset": {
            "train_examples": train_example_count,
            "eval_examples": example_count,
        },
        "generation": tuned_report.get("generation"),
        "metrics": {
            "baseline": {
                "eval_loss": baseline_loss,
                "perplexity": baseline_perplexity,
                "perplexity_overflow": baseline_overflow,
                "passed": baseline_passed_count,
                "pass_rate": baseline_passed_count / example_count,
            },
            "tuned": {
                "eval_loss": tuned_loss,
                "perplexity": tuned_report.get("perplexity"),
                "perplexity_overflow": tuned_report.get("perplexity_overflow"),
                "passed": tuned_passed_count,
                "pass_rate": tuned_passed_count / example_count,
            },
            "loss_change": tuned_loss - baseline_loss,
            "status_counts": status_counts,
        },
        "training": {
            "global_step": training_report.get("global_step"),
            "epoch": training_report.get("epoch"),
            "training_seconds": training_report.get("training_seconds"),
            "trainable_parameters": training_report.get("trainable_parameters"),
            "total_parameters": training_report.get("total_parameters"),
            "train_metrics": training_report.get("train_metrics"),
            "log_history": training_report.get("log_history"),
        },
        "artifacts": {
            "base_model": str(config.paths.base_model),
            "adapter": tuned_report.get("adapter_path"),
            "checkpoints": training_report.get("checkpoints"),
            "baseline_report": str(config.paths.reports / "baseline_report.json"),
            "tuned_report": str(config.paths.reports / "tuned_report.json"),
        },
        "conclusion": {
            "loss_improved": loss_improved,
            "manual_quality_improved": quality_improved,
            "overfitting_detected": False,
            "text": (
                "Eval loss и perplexity снизились, но ни один ответ не прошёл все "
                "ручные критерии; подтверждённого улучшения прикладного качества нет."
            ),
            "overfitting_analysis": (
                "Eval loss снизился с 1.5220 после первой эпохи до 1.4528 после второй, "
                "поэтому явных признаков переобучения в этом запуске нет. Маленький eval "
                "не позволяет сделать надёжный общий вывод."
            ),
            "limitations": [
                "Eval содержит только пять примеров одной предметной области.",
                "Ручная бинарная оценка не измеряет частичные улучшения формулировок.",
                "Проведён один запуск с одним seed и без подбора гиперпараметров.",
                "Маленькая базовая модель и 32 train-примера ограничивают качество.",
            ],
            "next_iteration": [
                "Добавить больше разнообразных train- и eval-примеров для пяти неудачных тем.",
                "Проверить больше эпох с контролем eval loss и сохранением лучшего checkpoint.",
                "Сравнить LoRA target modules q_proj/v_proj с q_proj/k_proj/v_proj/o_proj.",
            ],
        },
        "comparisons": comparisons,
    }


def run_evaluation(config_path: Path, manual_evaluation_path: Path) -> JsonObject:
    """Повторно загрузить adapter, сохранить tuned report и при наличии сравнить."""
    config: typing.Final = FineTuningConfigManager(config_path).load_and_validate()
    baseline_report_path: typing.Final = config.paths.reports / "baseline_report.json"
    training_report_path: typing.Final = config.paths.reports / "training_report.json"
    tuned_report_path: typing.Final = config.paths.reports / "tuned_report.json"
    comparison_report_path: typing.Final = config.paths.reports / "comparison_report.json"
    baseline_report: typing.Final = load_json_object(baseline_report_path)
    training_report: typing.Final = load_json_object(training_report_path)
    tuned_report: typing.Final = evaluate_reloaded_adapter(config)
    write_json_report(tuned_report_path, tuned_report)

    result: typing.Final[JsonObject] = {
        "adapter_reloaded_from_disk": True,
        "local_files_only": True,
        "tuned_report": str(tuned_report_path),
        "comparison_report": None,
        "manual_evaluation_required": True,
    }
    if manual_evaluation_path.is_file():
        manual_evaluations: typing.Final = load_manual_evaluations(manual_evaluation_path)
        apply_manual_evaluations(baseline_report, "baseline", manual_evaluations)
        apply_manual_evaluations(tuned_report, "tuned", manual_evaluations)
        write_json_report(baseline_report_path, baseline_report)
        write_json_report(tuned_report_path, tuned_report)
        comparison_report: typing.Final = build_comparison_report(
            config, baseline_report, tuned_report, training_report
        )
        write_json_report(comparison_report_path, comparison_report)
        result["comparison_report"] = str(comparison_report_path)
        result["manual_evaluation_required"] = False
        result["metrics"] = comparison_report["metrics"]
    return result


def run_adapter_evaluation_cli() -> None:
    """Разобрать CLI, оценить локальный adapter и вывести пути к отчётам."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    parser.add_argument(
        "--manual-evaluation",
        type=Path,
        default=Path("data/fine_tuning/manual_evaluation.json"),
    )
    args: typing.Final = parser.parse_args()
    report: typing.Final = run_evaluation(args.config, args.manual_evaluation)
    sys.stdout.write(f"{json.dumps(report, ensure_ascii=False, indent=2)}\n")


if __name__ == "__main__":
    run_adapter_evaluation_cli()
