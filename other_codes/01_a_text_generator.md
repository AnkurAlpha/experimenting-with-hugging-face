# 01_a_text_generator.py

## What this file does
Creates a text-generation pipeline with a chat-style prompt and returns one generated response.

## Model
- `Qwen/Qwen2.5-Coder-0.5B-Instruct`

## Code flow
1. `create_generator()` builds a `text-generation` pipeline.
2. `generate_text(...)` builds `system` + `user` messages.
3. Calls generator with sampling settings (`temperature`, `top_p`).
4. Handles output format differences (`str`, `dict`, or list of messages).
5. In `__main__`, sends a coding request and prints model output.

## How to run
```bash
uv run python other_codes/01_a_text_generator.py
```

## Notes
- `trust_remote_code=True` allows model-specific code execution from model repo.
- Output parsing is defensive because pipeline output formats can vary across versions.
