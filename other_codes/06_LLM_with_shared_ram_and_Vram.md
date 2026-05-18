# 06_LLM_with_shared_ram_and_Vram.py

## What this file does
Loads `google/gemma-2-2b-it` using mixed GPU/CPU placement so inference can run on constrained VRAM systems, then runs a single prompt-response chat generation.

## Main components
1. DType selection
- Chooses `bfloat16` when CUDA BF16 is supported.
- Falls back to `float16` on CUDA, otherwise `float32` on CPU.
2. Model loading with memory caps
- Uses `device_map="auto"` and `max_memory` (`3GiB` GPU, `12GiB` CPU).
- Uses `low_cpu_mem_usage=True` and `offload_state_dict=False` to avoid temporary disk offload.
3. Disk-offload safety check
- Prints `model.hf_device_map`.
- Stops execution if any module is mapped to `disk`.
4. Prompt + generation
- Builds a chat prompt via `tokenizer.apply_chat_template(...)`.
- Generates sampled output with `max_new_tokens=1024`, `temperature=0.7`, `top_p=0.9`.

## Code flow
1. Resolve runtime dtype.
2. Load tokenizer and model with RAM/VRAM sharing.
3. Validate that no disk offload was used.
4. Read user prompt from terminal.
5. Generate and print decoded response.

## How to run
```bash
uv run python other_codes/06_LLM_with_shared_ram_and_Vram.py
```

## Notes
- This script prints full decoded output after generation completes (non-streaming).
- If memory is insufficient and HF needs disk offload, it exits intentionally.
