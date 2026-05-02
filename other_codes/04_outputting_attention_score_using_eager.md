# 04_outputting_attention_score_using_eager.py

## What this file does
Loads a causal language model with eager attention implementation and prints attention tensor metadata (number of layers and shape of layer 0).

## Why eager attention is used
Some optimized attention backends may not expose attention matrices in the same way. Setting `attn_implementation="eager"` helps make attention outputs accessible for inspection.

## Code flow
1. Loads tokenizer and model (`Qwen/Qwen2.5-Coder-0.5B-Instruct`).
2. Reads `example.py` content.
3. Builds chat-style prompt using `apply_chat_template(...)`.
4. Tokenizes prompt.
5. Runs forward pass with:
- `output_attentions=True`
- `return_dict=True`
6. Prints:
- number of attention layers
- shape of first layer's attention tensor

## How to run
From inside `other_codes/`:
```bash
uv run python 04_outputting_attention_score_using_eager.py
```
Or from repo root (if path adjusted).

## Notes
- This script is for inspection/debugging, not normal text generation output.
- Attention tensors can consume significant memory for long inputs.
