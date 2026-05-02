# 02_a_text_generator_more_userlike.py

## What this file does
This is a streaming version of text generation. Instead of waiting for full output, it prints tokens as they are generated.

## Main difference from `01_a_text_generator.py`
Uses `TextStreamer` so response appears live in terminal.

## Code flow
1. Builds text-generation pipeline for Qwen.
2. Creates chat messages.
3. Creates `TextStreamer` with:
- `skip_prompt=True`
- `skip_special_tokens=True`
4. Calls pipeline with `streamer=...` and sampling settings.
5. In `__main__`, reads user input and appends it to a starter prompt.

## How to run
```bash
uv run python other_codes/02_a_text_generator_more_userlike.py
```

## Notes
- This script focuses on UX (interactive feel), not structured return values.
