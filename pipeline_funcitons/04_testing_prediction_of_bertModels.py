from transformers import pipeline

unmasker = pipeline("fill-mask", model="bert-base-uncased")

result = unmasker("This man works as a [MASK].")
print("This code is for understanding biases")
print("For man:")
print([r["token_str"] for r in result])

result = unmasker("This woman works as a [MASK].")
print("For woman:")
print([r["token_str"] for r in result])
