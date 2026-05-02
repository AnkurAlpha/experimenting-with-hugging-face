# 02_reuse_weights1.py

## What this file does
This file experiments with reusing a prebuilt pipeline object by serializing it with `pickle` to disk.

## Goal
Reduce repeated model setup time by loading a cached object (`cachefile/distilbert_classifier.pkl`) if present.

## Code flow
1. Defines cache path in `cachefile/`.
2. Creates the directory if needed.
3. If pickle exists:
- loads classifier from disk.
4. Else:
- builds a new sentiment pipeline.
- pickles it to disk.
5. Runs inference and prints result.

## How to run
```bash
uv run python pipeline_funcitons/02_reuse_weights1.py
```

## Important caution
The source code itself already warns this is an experiment. Pickling full pipeline/model objects is often fragile across library/version/platform changes. For robust workflows, reload model/tokenizer from identifiers and rely on Hugging Face local cache.
