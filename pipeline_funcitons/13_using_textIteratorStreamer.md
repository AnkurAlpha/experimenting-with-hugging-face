# 13_using_textIteratorStreamer.py

## What this file does
Uses `TextIteratorStreamer` with a background thread so you can iterate over generated text chunks in Python while generation is running.

## Goal
Support programmatic streaming workflows where you want both live output and collected final text.

## Code flow
1. Loads local model/tokenizer.
2. Tokenizes prompt.
3. Creates `TextIteratorStreamer`.
4. Starts `model.generate(...)` in a background `Thread`.
5. Iterates over streamer chunks, printing and appending to `final_text`.
6. Joins thread and prints full collected text.

## How to run
```bash
uv run python pipeline_funcitons/13_using_textIteratorStreamer.py
```

## Why use this pattern
It separates generation execution from token consumption, which is useful for custom UIs and APIs.
