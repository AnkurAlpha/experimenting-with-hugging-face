from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from pathlib import Path

saved_models = Path("./saved_models/") / "My-Qwen-model"

model = AutoModelForCausalLM.from_pretrained(saved_models)
tokenizer = AutoTokenizer.from_pretrained(saved_models)

prompt = "I am learning AI, explain transformers simply"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
)
streamer = TextStreamer(
    tokenizer,
    skip_special_tokens=True,
    skip_prompt=True)

_ = model.generate(
    **inputs,
    streamer=streamer,
    max_new_tokens=1024
)
