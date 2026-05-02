# 03_scan_the_file_find_the_bugs.py

## What this file does
Reads a local Python file (`example.py`) and asks a language model to analyze that code, streaming the response in terminal.

## Code flow
1. `create_generator()` creates a text-generation pipeline.
2. `generate_text(...)`:
- wraps prompt in chat messages
- defines `GenerationConfig`
- streams output with `TextStreamer`
3. In `__main__`:
- reads `./example.py`
- appends instruction asking for analysis
- runs generation

## How to run
From inside `other_codes/`:
```bash
uv run python 03_scan_the_file_find_the_bugs.py
```
Or from repo root (if path adjusted).

## Notes
- Current file path is relative (`Path("./example.py")`), so working directory affects whether it finds the file.
