from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from pathlib import Path

saved_models = Path("./saved_models") / "My-Qwen-model"

model = AutoModelForCausalLM.from_pretrained(saved_models)
tokenizer = AutoTokenizer.from_pretrained(saved_models)

prompt = "Explain transformers simply."
inputs = tokenizer(prompt, return_tensors="pt")

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

generation_kwargs = {
    **inputs,
    "streamer": streamer,
    "max_new_tokens": 80
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

final_text = ""

for new_text in streamer:
    print(new_text, end="", flush=True)
    final_text += new_text

thread.join()

print("\n\nFinal saved text:")
print(final_text)
