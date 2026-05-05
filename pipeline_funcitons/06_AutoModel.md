# 06_AutoModel.py

## What this file does
Demonstrates running a base Transformer model (`AutoModel`) on tokenized text and printing raw model outputs.

## Code flow
1. Loads:
- `AutoTokenizer`
- `AutoModel`
from the same DistilBERT checkpoint.
2. Tokenizes two input sentences into PyTorch tensors.
3. Runs forward pass with `model(**inputs)`.
4. Prints the full model output object.

## How to run
```bash
uv run python pipeline_funcitons/06_AutoModel.py
```

## Notes
- `AutoModel` returns hidden states style outputs, not task-specific labels.
- For classification tasks, use a task head model (as shown in `07_...`).
