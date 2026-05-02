# 01_turboquant.py

## What this file does
This script compares two generation runs using the same model:
1. Baseline generation (normal KV cache behavior).
2. Generation with `TurboQuantCache(bits=4)`.

It prints timing and (if on CUDA) peak GPU memory usage for each run.

## Main idea
The file is an experiment to check whether quantized KV cache (TurboQuant) might improve memory usage or speed for repeated generation tasks.

## Dependencies
- `torch`
- `transformers`
- `turboquant`
- `numpy`

## Code flow
1. Applies a NumPy compatibility patch (`np.trapz`) for newer NumPy versions.
2. Loads tokenizer and model (`Qwen/Qwen2.5-Coder-0.5B-Instruct`) in 4-bit weights via `BitsAndBytesConfig`.
3. Builds a long prompt by repeating a code snippet.
4. Defines `run_once()` which:
- tokenizes input
- runs `model.generate(...)`
- measures elapsed time
- reports generated text and memory usage
5. Runs baseline.
6. Creates `TurboQuantCache(bits=4)` and runs again.
7. Prints a note that this is a small benchmark.

## How to run
```bash
uv run python Different_methods/01_turboquant.py
```

## Notes for new readers
- This is a quick benchmark, not a production benchmark.
- Results can vary by GPU, driver, model size, and prompt length.
- `device_map="auto"` lets Transformers place model parts automatically.
