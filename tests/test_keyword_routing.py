import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

import app


@pytest.mark.parametrize("message", ["hi", "Hello", "hey", "heyy"])
def test_greetings_get_the_greeting(message):
    assert "Welcome to Zevoir" in app.check_chatbot_response(message)


def test_misspelling_is_fixed_before_matching():
    assert app.check_chatbot_response("thnks") == app.check_chatbot_response("thanks")


def test_short_services_message_gets_the_services_reply():
    assert "AI & Automation" in app.check_chatbot_response("services")


def test_email_address_is_captured():
    assert "team member will connect" in app.check_chatbot_response("reach me at jo@example.com")


@pytest.mark.parametrize(
    "question",
    [
        "How do I build a machine learning model from scratch?",
        "How does ChatGPT work?",
        "How can I contact Zevoir Technologies?",
        "What services does Zevoir Technologies offer?",
        "Who won the 2024 FIFA World Cup?",
    ],
)
def test_long_questions_go_to_retrieval_not_keywords(question):
    assert app.check_chatbot_response(question) is None
