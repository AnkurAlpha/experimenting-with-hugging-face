from transformers import AutoModel, AutoTokenizer
from pathlib import Path

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
saved_models = Path("./saved_models/") / "My-Qwen-model"
saved_models.mkdir(parents=True, exist_ok=True)

tokenizer.save_pretrained(saved_models)
model.save_pretrained(saved_models)
