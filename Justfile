default: lock install lint

setup:
    brew update
    brew install pyenv uv
    brew upgrade pyenv uv

python version:
    if ! pyenv versions --bare | grep -qx "{{version}}"; then \
        pyenv install {{version}}; \
    fi
    pyenv local {{version}}
    pyenv rehash

    uv python pin {{version}}
    uv venv --python {{version}} --clear

lock:
    uv lock --upgrade

install:
    uv sync --frozen --all-extras --no-install-project --all-groups
    . ./.venv/bin/activate

lint:
    uv run auto-typing-final .
    uv run ruff format
    uv run ruff check --fix
    uv run ty check


download_model:
    HF_HUB_DISABLE_XET=1 uv run hf download Qwen/Qwen2.5-0.5B-Instruct --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-dir data/models/base/Qwen2.5-0.5B-Instruct

# device: auto, cpu или gpu (MPS на macOS)
inspect_env device="auto":
    uv run python -m fine_tuning.environment --device {{device}}

validate_data:
    uv run python -m fine_tuning.dataset
