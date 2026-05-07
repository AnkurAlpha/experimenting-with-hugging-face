from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import torch

saved_models = Path("./saved_models/") / "My-Qwen-model"
# saved_models.mkdir(parents=True, exist_ok=True)

model = AutoModelForCausalLM.from_pretrained(saved_models)
tokenizer = AutoTokenizer.from_pretrained(saved_models)


# tokenizer.save_pretrained(saved_models)
# model.save_pretrained(saved_models)

raw_inputs = [
    "I am a linux user learning IA.",
    "I love this so much!",
]

inputs = tokenizer(
    raw_inputs,
    padding=True,
    truncation=True,
    return_tensors="pt"
)
with torch.no_grad():
    output = model(**inputs)

print(output)
print(output.logits)
print(output.logits.shape)
print("probabilites : ")
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(predictions)
