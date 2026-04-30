from pathlib import Path
from transformers import pipeline, TextStreamer
import torch

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def create_generator():
    generator = pipeline(
        task="text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available()
        else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    generator.model.generation_config.max_length = None
    return generator


def generate_text(generator, user_prompt: str):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Give clear and simple answers."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    stream = TextStreamer(
        tokenizer=generator.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generator(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        streamer=stream
    )


if __name__ == "__main__":
    file_path = Path("./example.py")
    content = file_path.read_text()
    prompt = content + "\n\n Read the following code and analyze it"

    generator = create_generator()
    generate_text(generator, prompt)
