# 10_uploading_it_into_hugging_face.py

## What this file does
Loads a local model/tokenizer and uploads both to a Hugging Face Hub repository.

## Goal
Publish your locally saved model artifacts to your own Hub repo.

## Code flow
1. Reads local path: `saved_models/My-Qwen-model`.
2. Sets target Hub repo ID (`AnkurAlpha/My-first-Qwen-model`).
3. Loads model and tokenizer from local files.
4. Calls `push_to_hub(...)` for model and tokenizer.

## How to run
```bash
uv run python pipeline_funcitons/10_uploading_it_into_hugging_face.py
```

## Requirements
- You must be authenticated with Hugging Face CLI/token.
- You need permission to push to the target repo.
