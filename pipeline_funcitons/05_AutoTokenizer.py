from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

raw_inputs = [
    "I am a linux user learning IA.",
    "I love this so much!",
]
inputs = tokenizer(raw_inputs, padding=True,
                   truncation=True, return_tensors="pt")

print(inputs)
