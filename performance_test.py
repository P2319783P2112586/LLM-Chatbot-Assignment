import os
import time

import psutil
import torch

from src.chatbot import Chatbot


def main():

    process = psutil.Process(os.getpid())

    print("=" * 60)
    print("LLM Performance Test")
    print("=" * 60)

    print("MPS available:",
          torch.backends.mps.is_available())

    print()

    print("Loading model...")
    chatbot = Chatbot()

    prompt = "Explain artificial intelligence in simple terms."

    memory_before = process.memory_info().rss / (1024 * 1024)

    start_time = time.perf_counter()

    response = chatbot.generate_response(prompt)

    end_time = time.perf_counter()

    memory_after = process.memory_info().rss / (1024 * 1024)

    latency = end_time - start_time
    memory_change = memory_after - memory_before

    print("\nPrompt:")
    print(prompt)

    print("\nResponse:")
    print(response)

    print("\nPerformance Results")
    print("-" * 40)
    print(f"Latency: {latency:.2f} seconds")
    print(f"Memory before generation: {memory_before:.2f} MB")
    print(f"Memory after generation: {memory_after:.2f} MB")
    print(f"Memory change: {memory_change:.2f} MB")


if __name__ == "__main__":
    main()