# 05_AutoTokenizer.py

## What this file does
Shows how to use `AutoTokenizer` to tokenize a batch of sentences and return PyTorch tensors.

## Code flow
1. Loads tokenizer from a checkpoint:
- `distilbert-base-uncased-finetuned-sst-2-english`
2. Defines two input sentences.
3. Tokenizes with:
- `padding=True`
- `truncation=True`
- `return_tensors="pt"`
4. Prints tokenized outputs (`input_ids`, `attention_mask`, etc.).

## How to run
```bash
uv run python pipeline_funcitons/05_AutoTokenizer.py
```

## Why this matters
Tokenization is the first step before passing text into Transformer models.
