from transformers import pipeline


class Chatbot:

    def __init__(self):
        print("Loading language model...")

        self.generator = pipeline(
            "text-generation",
            model="distilgpt2"
        )

        print("Model loaded successfully.")

    def generate_response(self, prompt):
        result = self.generator(
            prompt,
            max_new_tokens=80,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7
        )

        return result[0]["generated_text"]


def main():

    chatbot = Chatbot()

    print("=" * 60)
    print("Simple LLM Chatbot")
    print("Powered by HuggingFace Transformers")
    print("Type 'exit' to quit.")
    print("=" * 60)

    while True:

        user_input = input("\nYou: ")

        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        if not user_input.strip():
            print("Please enter a question.")
            continue

        response = chatbot.generate_response(user_input)

        print("\nAI:", response)


if __name__ == "__main__":
    main()