# netology-aiagnt-finetuning

Минимальное решение задания Нетологии [«Практика fine-tuning»](https://netology.ru/profile/program/aiagnt-1/lessons/651858/lesson_items/3501310).
План реализации — [PLAN.md](PLAN.md).

На первом шаге подготовлены окружение и каркас проекта. Подготовка train/eval,
обучение LoRA и сравнение качества выполняются на следующих шагах.

## Установка и зависимости

Все команды выполнять из корня репозитория:

```bash
cd /Users/mikhail_afanasiev/Dev/netology/netology-aiagnt-finetuning
just setup
just python 3.12.14
just lock
just install
just lint
```

`just setup` обновляет Homebrew, устанавливает/обновляет pyenv и uv.
`just python 3.12.14` устанавливает Python, записывает `.python-version` и
**пересоздаёт `.venv`**. Эту команду достаточно выполнить при первоначальной
настройке или смене Python.

Библиотеки в `pyproject.toml` перечислены **без ограничений версий**.
`just lock` вызывает `uv lock --upgrade` и выбирает последние совместимые версии.
`uv.lock` сохраняет фактически выбранные версии для воспроизводимости.
После обновления зависимостей выполнять:

```bash
just lock
just install
just lint
just inspect_env
```

Для установки уже согласованного окружения достаточно `just install`:
эта команда использует `uv.lock` без пересчёта. `just lint` запускает
`auto-typing-final`, Ruff (форматирование и исправления) и ty.
Рецепты `setup`, `python`, `lock`, `install`, `lint` сохранены без изменений.
Команда `just` по умолчанию выполняет `install` и `lint`.

Файлом зависимостей служит `pyproject.toml` вместе с `uv.lock`; отдельный
`requirements.txt` не нужен. Обучение планируется на PyTorch/Transformers/PEFT
с Accelerate; YAML будет читаться через PyYAML. Утилиты линтинга находятся
в группе `dev`.

## Базовая модель

Используется [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct):
causal instruct-модель примерно на 0.49B параметров, с русским языком и chat template.
Зафиксирован revision `7ae557604adf67be50417f59c2c2f167def9a775`.
Ограничения маленькой модели: ошибки в фактах, формулировках и следовании инструкции;
наличие улучшений после обучения предстоит измерить.

Скачать модель один раз:

```bash
just download_model
```

Ожидаемый каталог: `data/models/base/Qwen2.5-0.5B-Instruct/`.
Команда загружает tokenizer, конфигурацию, веса Safetensors и сопутствующие файлы
с указанного revision. Базовые веса и локальный кеш загрузки исключены из Git.
Для публичной модели HF token не требуется, `.env.example` не создаётся.
Ключ Groq не используется.

## Проверка окружения

```bash
just inspect_env
# Явный запуск на CPU:
just inspect_env cpu
```

Проверка загружает модель и tokenizer **только с диска**, проверяет chat template,
наличие модулей `q_proj` и `v_proj`, генерирует короткий ответ и проверяет autograd
на небольшом отдельном тензоре. Веса модели не обучаются и не меняются.
Это проверка работоспособности, а не baseline или оценка качества.

На macOS выбирается MPS, если он доступен, иначе CPU. Для первого небольшого
эксперимента используется float32 без mixed precision; автоматический выбор
CUDA/XPU не входит в этот минимальный вариант. Их доступность выводится для
диагностики. При ошибке операции на MPS можно явно выбрать CPU — ошибки
не скрываются автоматическим повтором.

В консоль выводится JSON с версиями Python и библиотек, устройством, dtype,
памятью, статусами загрузки модели и tokenizer, числом параметров и проверочным
ответом. Полный LoRA backward/optimizer step будет проверен на шаге обучения.

Фактическая проверка первого шага выполнена 16 сентября 2026 года:

- Python 3.12.14, PyTorch 2.14.0, Transformers 5.17.0, PEFT 0.21.0;
- macOS arm64, 16 GiB unified memory, MPS доступен и выбран;
- модель загружена на MPS в float32, число параметров — 494 032 768;
- tokenizer и chat template работают, проверочный ответ — `Привет!`;
- модули LoRA `q_proj` и `v_proj` найдены, простая проверка autograd пройдена;
- CUDA и XPU недоступны, CPU fallback доступен.

## Структура первого шага

```text
Justfile
pyproject.toml
uv.lock
.python-version
fine_tuning/
  __init__.py
  environment.py
config/
data/
  fine_tuning/
    reports/
    runs/
  models/
    base/
    adapters/
```

Пустые каталоги отмечены `.gitkeep`. Места для конфигурации, данных и результатов
подготовлены; фиктивных train/eval, отчётов обучения и адаптеров нет.
Секреты, `.venv`, кеши, базовая модель, будущие checkpoints и adapters исключены
из Git. Способ передачи checkpoint и adapter проверяющему будет описан при сдаче.
