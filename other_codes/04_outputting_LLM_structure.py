# the code is under refactoration for the next aim
from pathlib import Path
# from transformers import pipeline, TextStreamer, GenerationConfig
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"  # important for getting attention matrix
)

file_path = Path("./example.py")
content = file_path.read_text(encoding="utf-8")

messages = [
    {
        "role": "system",
        "content": "You are a helpful assistant.\
                    Give clear and simple answers."
    },
    {
        "role": "user",
        "content": f"""
        Read the following Python code and analyze it.
        Code:
        ```python
        {content}
        ```
        """
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output = model(
        **inputs,
        output_attentions=True,
        return_dict=True,
        # other extra params
    )

attention_scores = output.attentions
print("Number of layers:", len(attention_scores))
print("Shape of layer 0 attention scores:", attention_scores[0].shape)
