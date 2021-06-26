from convmodel.data import BlockDataset


def encode(text):
    """
    Args:
        text (str):
    Returns:
        List[int]: 
    """
    return list(range(len(text)))


def test_BlockDataset():
    dataset = BlockDataset(
        generator=lambda: ["Hello", "World"],
        encode_fn=encode,
        block_size=4,
        drop_last=True
    )
    got = list(iter(dataset))
    expected = [
        {'input_ids': [0, 1, 2, 3], 'labels': [1, 2, 3, 4]},
        {'input_ids': [4, 0, 1, 2], 'labels': [0, 1, 2, 3]},
        ]

    assert got == expected


def test_BlockDataset_keep_last():
    dataset = BlockDataset(
        generator=lambda: ["Hello", "World"],
        encode_fn=encode,
        block_size=4,
        drop_last=False
    )
    got = list(iter(dataset))
    expected = [
        {'input_ids': [0, 1, 2, 3], 'labels': [1, 2, 3, 4]},
        {'input_ids': [4, 0, 1, 2], 'labels': [0, 1, 2, 3]},
        {'input_ids': [3], 'labels': [4]},
        ]

    assert got == expected
