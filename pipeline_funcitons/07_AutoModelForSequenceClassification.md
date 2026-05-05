# 07_AutoModelForSequenceClassification.py

## What this file does
Runs sentiment classification logits with `AutoModelForSequenceClassification`, then converts logits to probabilities using softmax.

## Code flow
1. Loads sequence classification model and tokenizer from DistilBERT SST-2 checkpoint.
2. Tokenizes two sentences into tensors.
3. Runs model forward pass.
4. Prints:
- full output object
- `output.logits`
- `output.logits.shape`
5. Applies `torch.nn.functional.softmax(..., dim=-1)` to get probabilities.

## How to run
```bash
uv run python pipeline_funcitons/07_AutoModelForSequenceClassification.py
```

## Output interpretation
- Each row in logits/probabilities corresponds to one input sentence.
- For SST-2, there are two classes (negative/positive).
