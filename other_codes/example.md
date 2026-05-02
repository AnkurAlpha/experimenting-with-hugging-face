# example.py

## What this file does
Implements a prime-check function and prints prime numbers from 1 to 100.

## Code flow
1. Defines `is_prime(n)` using standard optimizations:
- reject `n <= 1`
- accept `2` and `3`
- reject multiples of `2` and `3`
- test factors in `6k ± 1` pattern
2. Loops from 1 to 100.
3. Prints values where `is_prime(num)` is `True`.

## How to run
```bash
uv run python other_codes/example.py
```

## Why this file exists here
It is used as sample input for model-analysis scripts in `other_codes/`.
