# 03_reuse_weights2.py

## What this file does
This script shows a better reuse pattern than repeatedly rebuilding the pipeline: create it once in-process, then call it for many inputs.

## Code flow
1. Creates one sentiment-analysis pipeline.
2. Defines a list of sentences.
3. Loops through sentences, runs inference, and prints compact formatted output.

## How to run
```bash
uv run python pipeline_funcitons/03_reuse_weights2.py
```

## Why this pattern matters
If you process multiple inputs in one run, reusing the same pipeline avoids repeated initialization overhead.
