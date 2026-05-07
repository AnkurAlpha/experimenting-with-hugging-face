import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"


class LLM_GenerationConfigs:
    @staticmethod
    def NormalConfig(tokenizer):
        return GenerationConfig(
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )


class LLM_Model:
    def __init__(self, model_id):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
        self.model.to(self.device)
        self.model.eval()
        self.generation_config = LLM_GenerationConfigs.NormalConfig(
            self.tokenizer)

    def encode(self, messages):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(
            prompt, return_tensors="pt").to(self.model.device)
        return inputs

    def generate(self, messages):
        inputs = self.encode(messages)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                generation_config=self.generation_config
            )
        input_length = inputs["input_ids"].shape[-1]
        new_tokens = generated_ids[0][input_length:]
        return self.tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        )


class Messages:
    def __init__(self, role, content):
        self.role = role
        self.content = content

    def message(self):
        return {
            "role": self.role,
            "content": self.content
        }


if __name__ == "__main__":
    model = LLM_Model(MODEL_ID)
    system_messagee = Messages(
        "system",
        "You are a helpful assistant. Give clear and simple answers."
    )
    x = input("Enter your prompt: ")
    user_message = Messages("user", x)
    messages = [system_messagee.message(), user_message.message()]
    print(model.generate(messages))
