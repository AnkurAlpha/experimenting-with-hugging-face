from pathlib import Path
from transformers import pipeline, TextStreamer, GenerationConfig
import torch

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def create_generator():
    text_generator = pipeline(
        task="text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available()
        else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    # text_generator.model.generation_config.max_length = None
    return text_generator


def generate_text(text_generator, user_prompt: str):
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
        tokenizer=text_generator.tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    generation_config = GenerationConfig(
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=text_generator.tokenizer.eos_token_id
    )
    text_generator(
        messages,
        return_full_text=False,
        generation_config=generation_config,
        streamer=stream
    )


if __name__ == "__main__":
    file_path = Path("./example.py")
    content = file_path.read_text()
    prompt = content + "\n\n Read the following code and analyze it"

    text_generator = create_generator()
    generate_text(text_generator, prompt)
