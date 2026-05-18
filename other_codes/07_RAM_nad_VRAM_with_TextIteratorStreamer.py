import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from threading import Thread

MODEL_ID = "google/gemma-2-2b-it"

if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

print("torch_dtype:", torch_dtype)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch_dtype,
    device_map="auto",
    max_memory={
        0: "3GiB",
        "cpu": "12GiB",
    },
    low_cpu_mem_usage=True,
    offload_state_dict=False,
)

print("Device map:")
print(model.hf_device_map)

if any(device == "disk" for device in model.hf_device_map.values()):
    raise RuntimeError(
        "Disk offload required. Stopping because disk offload is disabled."
    )


if __name__ == "__main__":
    prompt = input("Enter your prompt: ")

    messages = [
        {
            "role": "user",
            "content": (
                "You are a helpful assistant. Give clear and simple answers.\n\n"
                f"User question: {prompt}"
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = inputs.to(model.device)

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )

    generation_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    final_text = ""

    for new_text in streamer:
        print(new_text, end="", flush=True)
        final_text += new_text  # for future use

    thread.join()
