import pydantic
from typing import List


class ConversationExampleError(Exception):
    pass


class ConversationExample(pydantic.BaseModel):
    conversation: List[str]

    @pydantic.validator("conversation")
    def check_conversation_not_empty(cls, v):
        if not v:
            raise ConversationExampleError("conversation should not be empty")
        return v
