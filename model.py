from transformers import pipeline


generator = pipeline(
    "text-generation",
    model="distilgpt2"
)


prompt = "Artificial intelligence is"

result = generator(
    prompt,
    max_new_tokens=50,
    num_return_sequences=1
)

print(result[0]["generated_text"])
