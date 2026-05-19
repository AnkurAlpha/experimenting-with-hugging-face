import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from threading import Thread

MODEL_ID = "google/gemma-2-2b-it"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,

    # Allows CPU offload if GPU VRAM is not enough.
    # Important: CPU-offloaded parts are usually kept in FP32,
    # so they may use more normal RAM.
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map="auto",
    max_memory={
        0: "1GiB",
        "cpu": "12GiB",
    },
    low_cpu_mem_usage=True,
    offload_state_dict=False,
)

device_map = getattr(model, "hf_device_map", None)

if device_map is not None:
    print("Device map:")
    print(device_map)

    if any(device == "disk" for device in device_map.values()):
        raise RuntimeError(
            "Disk offload required. Stopping because disk offload is disabled."
        )
else:
    print("No hf_device_map found. Model loaded, but Transformers did not attach a device map.")

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
        final_text += new_text

    thread.join()

    print("\n")
