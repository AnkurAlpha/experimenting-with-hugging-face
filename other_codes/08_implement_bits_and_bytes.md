# 08_implement_bits_and_bytes.py

## What this file does
Loads `google/gemma-2-2b-it` using 8-bit quantization (`bitsandbytes`) with automatic device placement, then generates a streamed response from a user prompt.

## Main goal
Run a larger instruction model on limited VRAM by combining:
- 8-bit model weights
- CPU offload support
- explicit memory limits

## Code flow
1. Loads tokenizer and sets a fallback `pad_token_id` if missing.
2. Builds `BitsAndBytesConfig` with:
- `load_in_8bit=True`
- `llm_int8_enable_fp32_cpu_offload=True`
3. Loads model with:
- `device_map="auto"`
- `max_memory` for GPU and CPU
- `low_cpu_mem_usage=True`
4. Prints the resolved `hf_device_map`.
5. Stops execution if any module would be offloaded to disk.
6. Reads user prompt.
7. Formats chat input via `apply_chat_template(...)`.
8. Streams generation output using `TextIteratorStreamer` and a background thread.

## How to run
From repo root:
```bash
uv run python other_codes/08_implement_bits_and_bytes.py
```

## Notes
- This script is useful for memory-constrained local inference experiments.
- Disk offload is explicitly blocked to avoid very slow generation.
