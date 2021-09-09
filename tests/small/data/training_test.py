from convmodel.data import ConversationExample
from convmodel.data import ConversationExampleError
import pytest


def test_conversation():
    ex = ConversationExample(conversation=["Hi", "Hello"])
    assert ex.conversation == ["Hi", "Hello"]


def test_conversation_empty():
    with pytest.raises(ConversationExampleError):
        ConversationExample(conversation=[])
