import transformers
from typing import List


class ConversationTokenizer:
    def __init__(self, tokenizer, sep_token="<sep>"):
        if tokenizer.sep_token_id is None:
            tokenizer.sep_token = sep_token

        assert tokenizer.sep_token_id is not None
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name_or_path, **argv):
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name_or_path,
            **argv
        )
        return cls(tokenizer=tokenizer)

    def save_pretrained(self, *args, **argv):
        return self._tokenizer.save_pretrained(*args, **argv)

    def encode(self, text: str, **argv):
        assert type(text) == str, f"{text} should be a type of string"
        return self._tokenizer.encode(text, **argv)

    def decode(self, ids: List[int]):
        return self._tokenizer.decode(ids)

    def __call__(self, texts: List[str]):
        assert type(texts) == list
        assert texts

        # Add sep token id.
        # If tokens for each text are t0, t1, t2, then
        # - sep_token_id + t0
        # - sep_token_id + t1
        # - sep_token_id + t2
        # - sep_token_id
        input_ids_list = [
            [self._tokenizer.sep_token_id] + self._tokenizer.encode(t)
            for t in texts
        ]
        input_ids_list.append([self._tokenizer.sep_token_id])

        token_type_ids_list = []
        cur_token_type_id = 0
        for input_ids in input_ids_list:
            token_type_ids_list.append([cur_token_type_id] * len(input_ids))
            cur_token_type_id = (cur_token_type_id + 1) % 2

        input_ids = sum(input_ids_list, [])
        token_type_ids = sum(token_type_ids_list, [])
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
