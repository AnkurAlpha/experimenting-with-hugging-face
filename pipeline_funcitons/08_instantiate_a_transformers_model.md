# 08_instantiate_a_transformers_model.py

## What this file does
Downloads a pretrained Qwen model and tokenizer, then saves both locally to a project folder.

## Goal
Create a local copy of model artifacts so later scripts can load from disk instead of downloading each time.

## Code flow
1. Imports `AutoModel`, `AutoTokenizer`, and `Path`.
2. Sets model ID: `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
3. Loads model and tokenizer from Hugging Face Hub.
4. Creates local directory: `saved_models/My-Qwen-model`.
5. Saves tokenizer and model with `save_pretrained(...)`.

## How to run
```bash
uv run python pipeline_funcitons/08_instantiate_a_transformers_model.py
```

## Output
A local folder containing config, tokenizer files, and model weight files.
