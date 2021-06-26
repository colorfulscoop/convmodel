from convmodel.data import BertForPreTrainingDataset, BertSample
import random

random.seed(1)


def encode(text):
    """
    Args:
        text (str):
    Returns:
        List[int]: 
    """
    return list(range(len(text)))


def test_BlockDataset():
    a = "AAAAAAAAAA"
    b = "BBBBBBBBBBBBBBB"
    c = "CCCCCCCCCCCCCCCCCCCC"
    dataset = BertForPreTrainingDataset(
        generator=lambda: [
            BertSample(sentence=a, next_sentence=c, next_sentence_label=1),
            BertSample(sentence=b, next_sentence=c, next_sentence_label=0),
        ],
        encode_fn=encode,
        sep_token_id=10000,
        cls_token_id=10001,
        mask_token_id=10002,
        random_token_ids=[100, 101, 102],
        max_seq_len=40,
    )
    got = list(iter(dataset))

    want = [{
        'input_ids': [
            10001, 0, 1, 2, 3, 10002, 5, 6, 7, 8, 9, 10000,
            0, 1, 2, 3, 4, 5, 6, 7, 10002, 9, 10, 11, 12, 13, 100, 10002, 16, 10002, 18, 19, 10000,
            0, 0, 0, 0, 0, 0, 0],
        'labels': [
            -100, -100, -100, -100, -100, 4, -100, -100, -100, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100, -100, 8, -100, -100, -100, -100, -100, 14, 15, -100, 17, -100, -100, -100,
            -100, -100, -100, -100, -100, -100, -100],
        'attention_mask': [
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0],
        'token_type_ids': [
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            0, 0, 0, 0, 0, 0, 0],
        'next_sentence_label': 1
        },
        {
            'input_ids': [
                10001, 0, 1, 2, 3, 4, 5, 6, 10002, 8, 9, 10, 11, 12, 13, 14, 10000,
                10002, 10002, 2, 3, 4, 101, 6, 7, 8, 9, 10, 11, 12, 10002, 14, 15, 10002, 17, 18, 19, 10000,
                0, 0],
            'labels': [
                -100, -100, -100, -100, -100, -100, -100, -100, 7, -100, -100, -100, -100, -100, -100, -100, -100,
                0, 1, -100, -100, -100, 5, -100, -100, -100, -100, -100, -100, -100, 13, -100, -100, 16, -100, -100, -100, -100,
                -100, -100],
            'attention_mask': [
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0],
            'token_type_ids': [
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                0, 0],
            'next_sentence_label': 0
        }
    ]
    assert got == want

