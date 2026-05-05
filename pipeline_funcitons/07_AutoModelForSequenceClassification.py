from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
texts = [
    "I am a linux user learning IA.",
    "I love this so much!"
]

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    return_tensors="pt",
)
output = model(**inputs)
print(output)
print(output.logits)
print(output.logits.shape)

print("probabilites : ")
predictions = torch.nn.functional.softmax(output.logits, dim=-1)
print(predictions)
