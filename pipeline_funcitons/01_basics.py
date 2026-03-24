from transformers import pipeline as p

classifier = p("sentiment-analysis",
               model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
result = classifier(
    "Hey there! I have been soo much exited for you to open this repo!")
print(result)

generator = p("text-generation",
              model="distilbert/distilgpt2")
result = generator("The weather today is", max_length=20)
print(result)
