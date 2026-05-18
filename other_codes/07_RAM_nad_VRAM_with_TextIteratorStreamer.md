# 07_RAM_nad_VRAM_with_TextIteratorStreamer.py

## What this file does
Extends the RAM/VRAM shared-loading approach with `TextIteratorStreamer` so generated text is streamed token-by-token while generation is running.

## Main components
1. DType + memory-aware loading
- Same adaptive dtype logic as file 06.
- Uses `device_map="auto"` and `max_memory` for hybrid GPU/CPU execution.
- Fails fast if any layer is placed on `disk`.
2. Tokenizer safety
- Sets `pad_token_id` to `eos_token_id` if missing.
3. Streaming setup
- Creates `TextIteratorStreamer(skip_prompt=True, skip_special_tokens=True)`.
- Runs `model.generate(...)` inside a background `Thread`.
4. Incremental output
- Iterates over streamer chunks, prints live text, and accumulates into `final_text`.

## Code flow
1. Load tokenizer/model with constrained memory settings.
2. Build chat prompt with `apply_chat_template(...)`.
3. Start background generation thread.
4. Consume streamer output in the main thread.
5. Join thread when generation ends.

## How to run
```bash
uv run python other_codes/07_RAM_nad_VRAM_with_TextIteratorStreamer.py
```

## Notes
- Useful for terminal apps or APIs that need progressive output.
- `final_text` is collected for reuse, even though only streamed output is printed by default.
