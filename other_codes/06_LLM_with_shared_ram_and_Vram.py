import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "google/gemma-2-2b-it"

# Use BF16 only if the GPU properly supports it.
# Otherwise use FP16 on CUDA, FP32 on CPU.
if torch.cuda.is_available():
    if torch.cuda.is_bf16_supported():
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float16
else:
    torch_dtype = torch.float32

print("torch_dtype:", torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    device_map="auto",
    max_memory={
        0: "3GiB",
        "cpu": "12GiB",
    },
    low_cpu_mem_usage=True,
    offload_state_dict=False,  # important: avoid temporary disk offload
)

print("Device map:")
print(model.hf_device_map)

# Stop if Hugging Face tried to use disk.
if any(device == "disk" for device in model.hf_device_map.values()):
    raise RuntimeError(
        "Disk offload required. Stopping because disk offload is disabled.")


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

    # Move input tensors to the model's main device.
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,   # start smaller first
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            use_cache=True,
        )

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(response)
