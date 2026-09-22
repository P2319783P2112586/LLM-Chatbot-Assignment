import pytest

from src.chatbot import Chatbot


@pytest.fixture(scope="module")
def chatbot():
    return Chatbot()


def test_chatbot_initialization(chatbot):
    assert chatbot is not None
    assert chatbot.generator is not None


def test_chatbot_response(chatbot):
    response = chatbot.generate_response(
        "What is Python?"
    )

    assert response is not None
    assert isinstance(response, str)
    assert len(response.strip()) > 0


def test_multiple_prompts(chatbot):

    prompts = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is computer science?"
    ]

    for prompt in prompts:

        response = chatbot.generate_response(prompt)

        assert response is not None
        assert isinstance(response, str)
        assert len(response.strip()) > 0