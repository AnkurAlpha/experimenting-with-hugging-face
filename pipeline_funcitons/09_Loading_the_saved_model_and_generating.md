# 09_Loading_the_saved_model_and_generating.py

## What this file does
Loads a locally saved causal language model and tokenizer, runs a forward pass on sample inputs, and prints logits plus softmax probabilities.

## Goal
Verify local model loading works and inspect raw model prediction tensors.

## Code flow
1. Loads model/tokenizer from `saved_models/My-Qwen-model`.
2. Tokenizes two sample input strings.
3. Runs `model(**inputs)` under `torch.no_grad()`.
4. Prints full output object, logits, logits shape, and softmax probabilities.

## How to run
```bash
uv run python pipeline_funcitons/09_Loading_the_saved_model_and_generating.py
```

## Notes
- This script uses direct forward pass, not `generate(...)`.
- For token generation, see script `11_...`.
