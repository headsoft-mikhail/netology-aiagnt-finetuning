"""Reload the LoRA adapter and compare it with the saved baseline."""

# ruff: noqa: EM101, EM102, FBT001, T201, TRY003

import argparse
import json
import math
import typing
from pathlib import Path

from peft import PeftModel

from fine_tuning.config import Config, get_path, get_section, load_config
from fine_tuning.dataset import load_datasets
from fine_tuning.train import (
    CRITERIA,
    ChatCollator,
    evaluate_loss,
    generate_answers,
    load_model_and_tokenizer,
    tokenize_examples,
    write_json,
)

type JsonObject = dict[str, object]


def read_json(path: Path) -> JsonObject:
    """Read a JSON object from disk."""
    value: typing.Final = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Ожидается JSON-объект: {path}")
    return value


def apply_manual_scores(report: JsonObject, manual: JsonObject, stage: str) -> None:
    """Attach the saved manual score to each generated answer."""
    evaluations: typing.Final = manual["evaluations"]
    examples: typing.Final = report["examples"]
    if not isinstance(evaluations, dict) or not isinstance(examples, list):
        raise TypeError("Некорректный формат ручной оценки или отчёта")
    for example in examples:
        if not isinstance(example, dict):
            raise TypeError("Некорректный пример в отчёте")
        score = evaluations[str(example["id"])][stage]
        example["manual_evaluation"] = score


def comparison_status(baseline_passed: bool, tuned_passed: bool) -> str:
    """Return the required status for one pair of answers."""
    if not baseline_passed and tuned_passed:
        return "improved"
    if baseline_passed and not tuned_passed:
        return "regressed"
    return "unchanged_pass" if baseline_passed else "unchanged_fail"


def build_comparison(
    config: Config,
    baseline: JsonObject,
    tuned: JsonObject,
    training: JsonObject,
) -> JsonObject:
    """Build the final before/after report."""
    baseline_examples: typing.Final = baseline["examples"]
    tuned_examples: typing.Final = tuned["examples"]
    if not isinstance(baseline_examples, list) or not isinstance(tuned_examples, list):
        raise TypeError("В отчётах отсутствует список examples")

    comparisons: typing.Final[list[dict[str, object]]] = []
    counts: typing.Final = {"improved": 0, "regressed": 0, "unchanged_pass": 0, "unchanged_fail": 0}
    baseline_passed_count = 0
    tuned_passed_count = 0
    for baseline_item, tuned_item in zip(baseline_examples, tuned_examples, strict=True):
        if not isinstance(baseline_item, dict) or not isinstance(tuned_item, dict):
            raise TypeError("Некорректный пример в отчёте")
        baseline_score = baseline_item["manual_evaluation"]
        tuned_score = tuned_item["manual_evaluation"]
        if not isinstance(baseline_score, dict) or not isinstance(tuned_score, dict):
            raise TypeError("Отсутствует ручная оценка")
        baseline_passed = bool(baseline_score["passed"])
        tuned_passed = bool(tuned_score["passed"])
        status = comparison_status(baseline_passed, tuned_passed)
        counts[status] += 1
        baseline_passed_count += int(baseline_passed)
        tuned_passed_count += int(tuned_passed)
        comparisons.append(
            {
                "id": baseline_item["id"],
                "baseline_answer": baseline_item["generated_answer"],
                "tuned_answer": tuned_item["generated_answer"],
                "baseline_evaluation": baseline_score,
                "tuned_evaluation": tuned_score,
                "status": status,
            }
        )

    example_count: typing.Final = len(comparisons)
    return {
        "stage": "comparison",
        "model": get_section(config, "model")["name"],
        "method": "LoRA",
        "dataset": {
            "train_examples": training["train_examples"],
            "eval_examples": example_count,
        },
        "metrics": {
            "train_loss": training["train_loss"],
            "baseline_eval_loss": baseline["eval_loss"],
            "tuned_eval_loss": tuned["eval_loss"],
            "baseline_perplexity": baseline["perplexity"],
            "tuned_perplexity": tuned["perplexity"],
            "baseline_pass_rate": baseline_passed_count / example_count,
            "tuned_pass_rate": tuned_passed_count / example_count,
            "improved_count": counts["improved"],
            "regressed_count": counts["regressed"],
            "unchanged_count": counts["unchanged_pass"] + counts["unchanged_fail"],
            "status_counts": counts,
        },
        "artifacts": {
            "adapter": training["adapter_path"],
            "checkpoints": training["checkpoints"],
        },
        "conclusion": {
            "quality_improved": tuned_passed_count > baseline_passed_count,
            "text": (
                "Eval loss снизился, но ручной pass rate остался 0%. "
                "Эксперимент не показал улучшения прикладного качества."
            ),
            "reasons": [
                "В train только 32 примера, многие из них близки по смыслу.",
                "За две эпохи выполнено только 16 шагов оптимизации.",
                "Модель 0.5B слабо следует инструкциям уже до обучения.",
                "Eval содержит пять примеров, а бинарная оценка не учитывает частичные улучшения.",
            ],
            "next_iteration": (
                "Расширить train и eval, добавить разнообразные примеры по проваленным темам "
                "и проверить больше эпох или модель 1–1.5B."
            ),
        },
        "comparisons": comparisons,
    }


def run_evaluation(config_path: Path, manual_path: Path) -> JsonObject:
    """Reload the adapter, evaluate it and save tuned/comparison reports."""
    config: typing.Final = load_config(config_path)
    run_name: typing.Final = str(get_section(config, "run")["name"])
    reports_dir: typing.Final = get_path(config, "reports")
    adapter_dir: typing.Final = get_path(config, "adapters") / run_name
    baseline: typing.Final = read_json(reports_dir / "baseline_report.json")
    training: typing.Final = read_json(reports_dir / "training_report.json")
    manual: typing.Final = read_json(manual_path)

    base_model, tokenizer, device = load_model_and_tokenizer(config)
    _, eval_examples = load_datasets(config, tokenizer)
    model: typing.Final = PeftModel.from_pretrained(
        base_model,
        adapter_dir,
        local_files_only=True,
        is_trainable=False,
    ).to(device)
    if tokenizer.pad_token_id is None:
        raise ValueError("Tokenizer не содержит pad_token_id")
    eval_dataset: typing.Final = tokenize_examples(eval_examples, tokenizer)
    eval_loss: typing.Final = evaluate_loss(
        model,
        eval_dataset,
        ChatCollator(tokenizer.pad_token_id),
        int(get_section(config, "training")["per_device_eval_batch_size"]),
        device,
    )
    tuned: typing.Final[JsonObject] = {
        "stage": "tuned",
        "model": get_section(config, "model")["name"],
        "adapter_path": str(adapter_dir),
        "adapter_reloaded_from_disk": True,
        "eval_loss": eval_loss,
        "perplexity": math.exp(eval_loss),
        "generation": get_section(config, "generation"),
        "criteria": CRITERIA,
        "examples": generate_answers(
            model,
            tokenizer,
            eval_examples,
            get_section(config, "generation"),
            device,
        ),
    }
    apply_manual_scores(baseline, manual, "baseline")
    apply_manual_scores(tuned, manual, "tuned")
    comparison: typing.Final = build_comparison(config, baseline, tuned, training)
    write_json(reports_dir / "baseline_report.json", baseline)
    write_json(reports_dir / "tuned_report.json", tuned)
    write_json(reports_dir / "comparison_report.json", comparison)
    return comparison


def run_adapter_evaluation_cli() -> None:
    """Run adapter evaluation from the command line."""
    parser: typing.Final = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/fine_tuning.yaml"))
    parser.add_argument(
        "--manual-evaluation",
        type=Path,
        default=Path("data/fine_tuning/manual_evaluation.json"),
    )
    args: typing.Final = parser.parse_args()
    print(
        json.dumps(
            run_evaluation(args.config, args.manual_evaluation), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    run_adapter_evaluation_cli()
