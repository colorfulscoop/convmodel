from convmodel.tokenizer import ConversationTokenizer


class TokenizerMock:
    @property
    def sep_token_id(self):
        return 5

    def encode(self, text):
        encode_map = {
            "こんにちは": [10272, 15, 679, 9],
            "私は誰誰です": [5598, 5885, 5885, 2282],
            "おはようございます": [25373, 939, 13092, 2633]
        }
        return encode_map[text]


def test_tokenizer_encode():
    text = "こんにちは"
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    got = tokenizer.encode(text)
    want = [10272, 15, 679, 9]

    assert got == want


def test_tokenizer_call():
    texts = ["こんにちは", "私は誰誰です", "おはようございます"]
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    got = tokenizer(texts=texts)
    want = {
        'input_ids': [
            5, 10272, 15, 679, 9,
            5, 5598, 5885, 5885, 2282,
            5, 25373, 939, 13092, 2633,
            5],
        'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
        'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}

    assert got == want
