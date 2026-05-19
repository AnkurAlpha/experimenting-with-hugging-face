# 09_using_4bit_limit_the_weight_laod.py

## What this file does
Loads `google/gemma-2-2b-it` in 4-bit quantized mode and manually splits model layers between GPU and CPU to control VRAM usage, then performs streamed generation.

## Main goal
Test a more aggressive memory-saving setup than 8-bit by using:
- 4-bit NF4 quantization
- double quantization
- manual per-layer device mapping

## Code flow
1. Loads tokenizer and ensures `pad_token_id` exists.
2. Builds 4-bit `BitsAndBytesConfig` with:
- `load_in_4bit=True`
- `bnb_4bit_quant_type="nf4"`
- `bnb_4bit_compute_dtype=torch.float16`
- `bnb_4bit_use_double_quant=True`
3. Defines layer placement strategy:
- first `GPU_LAYERS` on GPU
- remaining layers on CPU
- keeps normalization and rotary embedding on CPU
4. Loads model with explicit `device_map`.
5. Prints device map and GPU memory usage stats.
6. Gets user prompt and builds chat input.
7. Sends inputs to the embedding device.
8. Streams generated text with `TextIteratorStreamer` + background thread.
9. Prints GPU memory stats again after generation.

## How to run
From repo root:
```bash
uv run python other_codes/09_using_4bit_limit_the_weight_laod.py
```

## Notes
- This script is tuned for constrained hardware and layer-placement experimentation.
- `GPU_LAYERS` can be adjusted to find a stable speed/memory tradeoff.
