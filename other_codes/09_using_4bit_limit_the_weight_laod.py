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

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,

    # Needed when some modules are intentionally kept on CPU
    llm_int8_enable_fp32_cpu_offload=True,
)

# Gemma 2 2B has 26 decoder layers: model.layers.0 to model.layers.25
# Start with 18 GPU layers. If it works, try 20, 22, 24, 26.
GPU_LAYERS = 18
TOTAL_LAYERS = 26

device_map = {
    "model.embed_tokens": 0,
    "lm_head": 0,
}

for i in range(TOTAL_LAYERS):
    if i < GPU_LAYERS:
        device_map[f"model.layers.{i}"] = 0
    else:
        device_map[f"model.layers.{i}"] = "cpu"

device_map["model.norm"] = "cpu"
device_map["model.rotary_emb"] = "cpu"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    dtype=torch.float16,
    device_map=device_map,
    low_cpu_mem_usage=True,
    offload_state_dict=False,
)

print("Device map:")
print(model.hf_device_map if hasattr(model, "hf_device_map") else device_map)

if any(device == "disk" for device in device_map.values()):
    raise RuntimeError(
        "Disk offload required. Stopping because disk offload is disabled."
    )

if torch.cuda.is_available():
    print(f"Allocated VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
    print(f"Reserved VRAM:  {torch.cuda.memory_reserved() / 1024**3:.2f} GiB")


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

    # Send inputs to the device where embeddings are placed.
    input_device = next(model.model.embed_tokens.parameters()).device
    inputs = inputs.to(input_device)

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

    if torch.cuda.is_available():
        print(f"Allocated VRAM after generation: {
              torch.cuda.memory_allocated() / 1024**3:.2f} GiB")
        print(f"Reserved VRAM after generation:  {
              torch.cuda.memory_reserved() / 1024**3:.2f} GiB")

# 4-bit needs manual layer split
