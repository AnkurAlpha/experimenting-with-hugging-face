# 11_text_generation_using_local_weights.py

## What this file does
Runs text generation using the locally saved Qwen model weights.

## Goal
Generate text without depending on online model download at runtime.

## Code flow
1. Loads model/tokenizer from `saved_models/My-Qwen-model`.
2. Defines a user prompt.
3. Tokenizes prompt into PyTorch tensors.
4. Calls `model.generate(...)` with:
- `num_beams=4`
- `max_new_tokens=100`
5. Decodes generated token IDs and prints final text.

## How to run
```bash
uv run python pipeline_funcitons/11_text_generation_using_local_weights.py
```

## Notes
- Beam search is used (`num_beams=4`), so output is more deterministic than sampling-based generation.
