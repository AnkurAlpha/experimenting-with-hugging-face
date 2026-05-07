# 04_outputting_LLM_structure.py

## What this file does
Implements a small object-oriented chat generation wrapper around a local Hugging Face causal model (`Qwen/Qwen2.5-Coder-0.5B-Instruct`).

## Main components
1. `LLM_GenerationConfigs`
- Provides a static `NormalConfig(tokenizer)` method returning a `GenerationConfig` with sampling settings (`max_new_tokens`, `temperature`, `top_p`, and EOS/PAD IDs).
2. `LLM_Model`
- Loads tokenizer and model.
- Moves model to available device (`cuda` or `cpu`).
- Encodes chat messages via `apply_chat_template(...)`.
- Generates output with `model.generate(...)`.
- Returns only newly generated text (prompt portion removed).
3. `Messages`
- Simple helper class to build message dicts (`{"role": ..., "content": ...}`).

## Code flow
1. Instantiate `LLM_Model`.
2. Build system and user messages.
3. Convert messages to chat-template prompt and tokenize.
4. Generate text using configured generation parameters.
5. Decode and print only the generated continuation.

## How to run
From inside `other_codes/`:
```bash
uv run python 04_outputting_LLM_structure.py
```

## Notes
- This script now performs text generation, not attention-matrix inspection.
- User prompt is interactive (`input("Enter your prompt: ")`).
