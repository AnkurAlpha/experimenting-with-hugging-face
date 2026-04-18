import time
import numpy as np

# NumPy compatibility patch for TurboQuant on newer NumPy versions
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from turboquant import TurboQuantCache

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

print(f"Loading tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map="auto",
)
model.eval()

prompt = """
Read this code and explain what it does briefly.

def add(a, b):
    return a + b

def mul(a, b):
    return a * b

def solve(x):
    y = add(x, 3)
    z = mul(y, 10)
    return z
""" * 5


def run_once(label: str, cache=None, max_new_tokens: int = 16):
    inputs = tokenizer(prompt, return_tensors="pt")

    if DEVICE == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    input_len = inputs["input_ids"].shape[1]

    start = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            past_key_values=cache,
        )
    elapsed = time.time() - start

    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text = tokenizer.decode(
        outputs[0][input_len:], skip_special_tokens=True)

    print(f"\n--- {label} ---")
    print(f"Time: {elapsed:.2f}s")

    if DEVICE == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        print(f"Peak CUDA memory: {peak_mb:.2f} MB")

    print("\nGenerated text only:")
    print(generated_text if generated_text.strip()
          else "[No visible new text generated]")

    print("\nFull output preview:")
    print(full_text[:700])
    print("-" * 80)


run_once("Baseline (no TurboQuant)")

tq_cache = TurboQuantCache(bits=4)
run_once("TurboQuant cache (4-bit)", cache=tq_cache)

print()
print()
print("Please note that this benchmark is too small to decide whether turboquant is useful or not")
