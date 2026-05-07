from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

saved_models = Path("./saved_models/") / "My-Qwen-model"
repo_id = "AnkurAlpha/My-first-Qwen-model"

model = AutoModelForCausalLM.from_pretrained(saved_models)
tokenizer = AutoTokenizer.from_pretrained(saved_models)

model.push_to_hub(repo_id, commit_message="My first commit")
tokenizer.push_to_hub(repo_id, commit_message="My first commit")
