import torch


class BlockDataset(torch.utils.data.IterableDataset):
    def __init__(self, tokenizer, generator, block_size, drop_last=True):
        super().__init__()
        self._block_size = block_size
        self._tokenizer = tokenizer
        self._generator = generator
        self._drop_last = drop_last

    @classmethod
    def from_texts(cls, tokenizer, texts, block_size):
        """
        Args:
            tokenizer (transformers.AutoTokenizer)
            texts (List[str])
            block_size (int)
        """
        return cls(
            tokenizer=tokenizer,
            generator=lambda: texts,
            block_size=block_size
        )

    @classmethod
    def from_file(cls, tokenizer, filepath, block_size):
        return cls(
            tokenizer=tokenizer,
            generator=lambda: (line.strip("\n") for line in open(filepath)),
            block_size=block_size
        )

    def __iter__(self):
        """
            Yields (List[int])
        """
        ids = []
        for text in self._generator():
            ids.extend(self._tokenizer.encode(text))
            while len(ids) >= self._block_size+1:
                yield {"input_ids": ids[:self._block_size],
                       "labels": ids[1:self._block_size+1]}
                ids = ids[self._block_size:]
        if not self._drop_last:
            yield {"input_ids": ids[:-1],
                   "labels": ids[1:]}

    @classmethod
    def collate_fn(cls, item):
        """Collate function for DataLoader
        Args:
            item (List[dict[str, List[int]]]): BlockDataset のイテレータが返す辞書のリスト
        Returns:
            (dict[str, torch.Tensor]):
        """
        keys = item[0].keys()
        dic = {
            key: torch.tensor([x[key] for x in item])
            for key in keys
        }
        return dic
