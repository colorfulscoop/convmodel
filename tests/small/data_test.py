from convmodel.data import BlockDataset


class MockTokenizer:
    def encode(self, text):
        """
        Args:
            text (str):
        Returns:
            List[int]: 
        """
        return list(range(len(text)))


def test_BlockDataset():
    tokenizer = MockTokenizer()
    dataset = BlockDataset(
        generator=lambda: ["Hello", "World"],
        tokenizer=tokenizer,
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
    tokenizer = MockTokenizer()
    dataset = BlockDataset(
        generator=lambda: ["Hello", "World"],
        tokenizer=tokenizer,
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
