from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

saved_models = Path("./saved_models/") / "My-Qwen-model"

model = AutoModelForCausalLM.from_pretrained(saved_models)
tokenizer = AutoTokenizer.from_pretrained(saved_models)

prompt = "I am learning AI, explain transformers simply"

inputs = tokenizer(
    prompt,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
generated_ids = model.generate(
    **inputs,
    num_beams=4,
    max_new_tokens=100,
)
text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
