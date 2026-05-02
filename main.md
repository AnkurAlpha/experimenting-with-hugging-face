# main.py

## What this file does
This is a minimal environment-check script. It prints the exact Python interpreter path being used to run the script.

## Why this is useful
When working with virtual environments, this helps confirm you are using the expected Python binary.

## Code flow
1. Imports Python's built-in `sys` module.
2. Defines `main()`.
3. Prints `sys.executable`.
4. Runs `main()` only when executed directly.

## How to run
```bash
uv run python main.py
```

## Expected output
A path such as:
```text
/home/ankur/git_work/experimenting_with_huggingface/.venv/bin/python
```
