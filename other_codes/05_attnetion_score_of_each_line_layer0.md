# 05_attnetion_score_of_each_line_layer0.py

## What this file does
Creates an interactive chat-style generation utility using `AutoModelForCausalLM` and `GenerationConfig`.

## Main components
1. `LLM_GenerationConfigs`
- Defines `NormalConfig(tokenizer)` for sampling defaults:
`max_new_tokens=1024`, `do_sample=True`, `temperature=0.7`, `top_p=0.9`, and EOS/PAD IDs.
2. `LLM_Model`
- Loads tokenizer/model from `Qwen/Qwen2.5-Coder-0.5B-Instruct`.
- Chooses `cuda` when available.
- Builds prompt with `apply_chat_template(...)`.
- Runs `model.generate(...)` and decodes only newly generated tokens.
3. `Messages`
- Helper class for creating message dictionaries.

## Code flow
1. Initialize model wrapper.
2. Create a system instruction message.
3. Read user input from terminal.
4. Generate and print assistant response.

## How to run
From inside `other_codes/`:
```bash
uv run python 05_attnetion_score_of_each_line_layer0.py
```

## Notes
- Despite the filename mentioning attention scores, current implementation is a chat generation prototype.
- The script does not compute or print attention matrices yet.
