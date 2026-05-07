# 12_using_textStreamer.py

## What this file does
Demonstrates streaming generated tokens directly to terminal using `TextStreamer`.

## Goal
Improve interaction UX by showing output as it is produced instead of waiting for full completion.

## Code flow
1. Loads local model/tokenizer.
2. Tokenizes prompt.
3. Creates `TextStreamer` with:
- `skip_prompt=True`
- `skip_special_tokens=True`
4. Calls `model.generate(...)` with `streamer=...`.

## How to run
```bash
uv run python pipeline_funcitons/12_using_textStreamer.py
```

## Notes
- This script prints output live and does not store structured final generation metadata.
