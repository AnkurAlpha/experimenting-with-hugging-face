# 01_basics.py

## What this file does
This is the simplest Hugging Face `pipeline` example in the repo. It performs sentiment analysis on one sentence.

## Code flow
1. Imports `pipeline` as `p`.
2. Creates a sentiment-analysis pipeline using a DistilBERT SST-2 model.
3. Runs inference on one input string.
4. Prints the result list with `label` and `score`.

## How to run
```bash
uv run python pipeline_funcitons/01_basics.py
```

## Expected output shape
```python
[{'label': 'POSITIVE' or 'NEGATIVE', 'score': float}]
```
