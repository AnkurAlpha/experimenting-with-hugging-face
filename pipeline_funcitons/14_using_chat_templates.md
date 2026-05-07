# 14_using_chat_templates.py

## What this file does
Builds a chat-formatted prompt using tokenizer chat templates, then streams response text with `TextIteratorStreamer`.

## Goal
Use instruct/chat models in the intended message format instead of plain prompt strings.

## Code flow
1. Loads local model/tokenizer.
2. Creates chat messages list (`role` + `content`).
3. Converts messages into model prompt using `apply_chat_template(..., add_generation_prompt=True)`.
4. Tokenizes formatted text.
5. Starts generation in a background thread with iterator streaming.
6. Prints live chunks and final combined text.

## How to run
```bash
uv run python pipeline_funcitons/14_using_chat_templates.py
```

## Notes
- Chat templates help align prompt formatting with model training/inference expectations.
