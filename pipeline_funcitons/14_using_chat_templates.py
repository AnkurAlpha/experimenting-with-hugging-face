from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from pathlib import Path
from threading import Thread

saved_path = Path("./saved_models") / "My-Qwen-model"

model = AutoModelForCausalLM.from_pretrained(saved_path)
tokenizer = AutoTokenizer.from_pretrained(saved_path)

messages = [
    {"role": "user", "content": "Explain transformers simply."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(text, return_tensors="pt")

streamer = TextIteratorStreamer(
    tokenizer,
    skip_prompt=True,
    skip_special_tokens=True
)

generation_kwargs = {
    **inputs,
    "streamer": streamer,
    "max_new_tokens": 200,
    "pad_token_id": tokenizer.eos_token_id,
}

thread = Thread(target=model.generate, kwargs=generation_kwargs)
thread.start()

final_text = ""

for new_text in streamer:
    print(new_text, end="", flush=True)
    final_text += new_text

thread.join()

print("\n\nFinal text:")
print(final_text)
