# ---------------------------------------------------------------------------------
# The next best way is to create a pipeline object
# once and reuse it again and again
# ---------------------------------------------------------------------------------
from transformers import pipeline

# Create pipeline once
classifier = pipeline("sentiment-analysis",
                      model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")

# Test multiple sentences
sentences = [
    "I love this movie!",
    "This is terrible",
    "Hey there! I have been soo much exited for you to open this repo!"
]

for sentence in sentences:
    result = classifier(sentence)
    print(f"'{sentence}' → {result[0]['label']} ({result[0]['score']:.3f})")
