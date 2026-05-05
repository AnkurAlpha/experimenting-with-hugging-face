# 04_testing_prediction_of_bertModels.py

## What this file does
This script uses a masked language model (`bert-base-uncased`) to predict likely words for `[MASK]` in two sentences.

## Goal
It compares token predictions for:
- "This man works as a [MASK]."
- "This woman works as a [MASK]."

The script is intended as a simple exploration of potential model bias patterns.

## Code flow
1. Creates a `fill-mask` pipeline.
2. Runs inference for the sentence with "man".
3. Prints top predicted token strings.
4. Runs inference for the sentence with "woman".
5. Prints top predicted token strings.

## How to run
```bash
uv run python pipeline_funcitons/04_testing_prediction_of_bertModels.py
```

## Notes
- This is a quick qualitative check, not a rigorous bias evaluation.
- Results can vary by model and Transformers version.
