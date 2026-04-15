# Transformers Experiments

A small repo for learning and experimenting with the Transformers ecosystem.

## Setup (`uv` + `.venv`)

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
4. Activate environment:
```bash
source .venv/bin/activate.fish
```
For bash/zsh:
```bash
source .venv/bin/activate
```
5. Install dependencies:
```bash
uv pip install -r requirements.txt
```

## Verify installation

```bash
which python
python --version
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__)"
```

## Daily workflow

```bash
source .venv/bin/activate.fish
python your_script.py
python -m pipeline_funcitons.some_module
deactivate
```

## Common `uv pip` commands

```bash
uv pip install <package>
uv pip install --upgrade <package>
uv pip uninstall <package>
uv pip list
uv pip freeze
uv pip show <package>
uv pip tree
uv pip check
uv pip sync requirements.txt
```

## Helpful maintenance

```bash
uv cache clean
uv cache clean <package-name>
uv pip install --reinstall <package-name>
```

Clean rebuild:

```bash
rm -rf .venv
uv venv
source .venv/bin/activate.fish
uv pip sync requirements.txt
```

## Notes

- Keep work inside `.venv`.
- Use `uv pip install -r requirements.txt` while experimenting.
- Use `uv pip sync requirements.txt` when you want an exact environment.
