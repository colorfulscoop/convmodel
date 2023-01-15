from convmodel.models.gpt2lmcl.dataset import LMWithClassificationDataset as Dataset
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
        {"turns": ["こんにちは", "私は誰誰です"]},
        {"turns": ["こんにちは", "私は誰誰です", "おはようございます"]},
    ])
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    dataset = Dataset(
        iterator=corpus,
        tokenizer=tokenizer,
    )
    got = list(dataset.build_dataset(batch_size=2, shuffle_func=lambda x: x.reverse()))
    want = [
        {
            'input_ids': [5, 10272, 15, 679, 9, 5, 5598, 5885, 5885, 2282, 5],
            'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "lm_labels": [5, 10272, 15, 679, 9, 5, 5598, 5885, 5885, 2282, 5],
            "cl_labels": 1,
        },
        {
            # negative sample: ["こんにちは", "おはようございます"]
            'input_ids': [5, 10272, 15, 679, 9, 5, 25373, 939, 13092, 2633, 5],
            'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "lm_labels": [-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, -100],
            "cl_labels": 0,
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
            "lm_labels": [
                5, 10272, 15, 679, 9,
                5, 5598, 5885, 5885, 2282,
                5, 25373, 939, 13092, 2633,
                5
            ],
            "cl_labels": 1,
        },
        {
            # negative sample: ["こんにちは", "私は誰々です", "私は誰々です"]
            'input_ids': [
                5, 10272, 15, 679, 9,
                5, 5598, 5885, 5885, 2282,
                5, 5598, 5885, 5885, 2282,
                5
            ],
            'token_type_ids': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1],
            'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            "lm_labels": [
                -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100,
                -100, -100, -100, -100, -100,
                -100
            ],
            "cl_labels": 0,
        }
    ]

    assert got == want


def test_build_data_loader():
    corpus = datasets.IterableDataset.from_generator(lambda: [
        {"turns": ["こんにちは", "私は誰誰です"]},
        {"turns": ["こんにちは", "私は誰誰です", "おはようございます"]},
    ])
    tokenizer = ConversationTokenizer(tokenizer=TokenizerMock())
    dataset = Dataset(
        iterator=corpus,
        tokenizer=tokenizer,
    )
    loader = dataset.build_data_loader(batch_size=2)

    assert [item["input_ids"].shape[0] for item in loader] == [2, 2]
