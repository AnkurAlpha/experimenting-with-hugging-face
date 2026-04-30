# from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from transformers import pipeline

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def create_generator():
    generator = pipeline(
        task="text-generation",
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        dtype=torch.bfloat16 if torch.cuda.is_available()
        else torch.float32,
        device_map="auto",
        trust_remote_code=True,
        # token = True, # only needed for private/gated models
    )
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
    output = generator(
        messages,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
    )
    generated_text = output[0]["generated_text"]

    # For chat models , sometimes generated_text may be a
    # message dict/list depending on version
    if isinstance(generated_text, list):
        return generated_text[-1]["content"]
    if isinstance(generated_text, dict):
        return generated_text["content"]
    return generated_text


if __name__ == "__main__":
    generator = create_generator()
    user_prompt = "Write a simple python program to check for whether the input given by user is primer number"
    generated_text = generate_text(generator, user_prompt)
    print(generated_text)
