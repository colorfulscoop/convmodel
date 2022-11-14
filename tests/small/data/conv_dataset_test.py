from convmodel.models.gpt2lm.dataset import LMDataset as Dataset
from convmodel.tokenizer import ConversationTokenizer
import datasets


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


def test_iter():
    corpus = datasets.IterableDataset.from_generator(lambda: [
        {"turns": ["こんにちは"]},
        {"turns": ["こんにちは", "私は誰誰です"]},
        {"turns": ["こんにちは", "私は誰誰です", "おはようございます"]},
    ])
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    dataset = Dataset(
        iterator=corpus,
        tokenizer=tokenizer,
    )
    got = list(dataset.build_dataset())
    want = [
        {
            'input_ids': [5, 10272, 15, 679, 9, 5],
            'token_type_ids': [0, 0, 0, 0, 0, 1],
            'attention_mask': [1, 1, 1, 1, 1, 1],
            "labels": [5, 10272, 15, 679, 9, 5]
        },
        {
            'input_ids': [5, 10272, 15, 679, 9, 5, 5598, 5885, 5885, 2282, 5],
            'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "labels": [5, 10272, 15, 679, 9, 5, 5598, 5885, 5885, 2282, 5]
        },
        {
            'input_ids': [
                5, 10272, 15, 679, 9,
                5, 5598, 5885, 5885, 2282,
                5, 25373, 939, 13092, 2633,
                5
            ],
            'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "labels": [
                5, 10272, 15, 679, 9,
                5, 5598, 5885, 5885, 2282,
                5, 25373, 939, 13092, 2633,
                5
            ]
        }
    ]

    assert got == want


def test_iter_max_len():
    corpus = datasets.IterableDataset.from_generator(lambda: [
        {"turns": ["こんにちは"]},
        {"turns": ["こんにちは", "私は誰誰です"]},
    ])
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    dataset = Dataset(
        iterator=corpus,
        tokenizer=tokenizer,
        max_len=7,
    )
    got = list(dataset.build_dataset())
    want = [
        {
            'input_ids': [5, 10272, 15, 679, 9, 5],
            'token_type_ids': [0, 0, 0, 0, 0, 1],
            'attention_mask': [1, 1, 1, 1, 1, 1],
            "labels": [5, 10272, 15, 679, 9, 5]
        },
        {
            'input_ids': [5, 10272, 15, 679, 9, 5, 5598],
            'token_type_ids': [0, 0, 0, 0, 0, 1, 1],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1],
            "labels": [5, 10272, 15, 679, 9, 5, 5598]
        },
    ]
    assert got == want


def test_build_data_loader():
    corpus = datasets.IterableDataset.from_generator(lambda: [
        {"turns": ["こんにちは"]},
        {"turns": ["こんにちは", "私は誰誰です"]},
        {"turns": ["こんにちは", "私は誰誰です", "おはようございます"]},
    ])
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    dataset = Dataset(
        iterator=corpus,
        tokenizer=tokenizer,
    )
    loader = dataset.build_data_loader(batch_size=2)

    assert [item["input_ids"].shape[0] for item in loader] == [2, 1]
