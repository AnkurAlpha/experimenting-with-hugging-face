# Transformers Experiments

A small repo for learning and experimenting with the Transformers ecosystem.

## Setup (`uv` only)

1. Install `uv` (Arch):
```bash
sudo pacman -S uv
```
2. Pin Python:
```bash
uv python pin 3.14
```
3. Create virtual environment:
```bash
uv venv
```
4. Install dependencies from lockfile:
```bash
uv sync
```

## Verify installation

```bash
uv run python --version
uv run python -c "import torch; print(torch.cuda.is_available())"
uv run python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
```

## Daily workflow

```bash
uv run python your_script.py
uv run python -m pipeline_funcitons.some_module
```

## Common `uv` commands

```bash
uv add <package>
uv remove <package>
uv sync
uv lock
uv tree
```

## Helpful maintenance

```bash
uv cache clean
uv cache clean <package-name>
uv sync --reinstall
```

Clean rebuild:

```bash
rm -rf .venv
uv venv
uv sync
```

## Notes

- Keep work inside `.venv`.
- Use `uv add <package>` while experimenting.
- Run `uv sync` to restore the exact locked environment.
