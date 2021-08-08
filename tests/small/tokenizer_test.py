from convmodel.tokenizer import ConversationTokenizer


class TokenizerMock:
    def __init__(self, init_sep_token=True):
        if init_sep_token:
            # Emulate sep_token is defined in tokenizer
            self._sep_token = "<sep>"
            self._sep_token_id = 5
        else:
            # Emulate sep_token is not defined
            self._sep_token = None
            self._sep_token_id = None

    @property
    def sep_token_id(self):
        return self._sep_token_id

    @property
    def sep_token(self):
        return self._sep_token

    @sep_token.setter
    def sep_token(self, val):
        self._sep_token = val
        self._sep_token_id = 5

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


def test_tokenizer_init_sep_token_id():
    hf_tokenizer = TokenizerMock(init_sep_token=False)
    assert hf_tokenizer.sep_token_id is None

    tokenizer = ConversationTokenizer(
        tokenizer=TokenizerMock(init_sep_token=False)
    )

    assert tokenizer.sep_token == "<sep>"
    assert type(tokenizer.sep_token_id) == int
