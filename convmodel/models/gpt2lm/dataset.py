import torch
from convmodel.tokenizer import ConversationTokenizer


class LMDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        iterator,
        tokenizer: ConversationTokenizer,
        max_len: int=None
    ):
        """
        Args:
            max_len: max length of a tensor input to a model
        """
        super().__init__()
        self._iterator = iterator
        self._tokenizer = tokenizer
        self._max_len = max_len

    def _tokenize(self, sample):
        model_input = self._tokenizer(sample["turns"])
        model_input["labels"] = model_input["input_ids"]
        if self._max_len is not None:
            for key, val in model_input.items():
                model_input[key] = val[:self._max_len]

        return model_input

    @classmethod
    def collate_fn(cls, item):
        """Collate function for DataLoader
        Args:
            item (List[dict[str, List[int]]]): BlockDataset のイテレータが返す辞書のリスト
        Returns:
            (dict[str, torch.Tensor]):
        """
        keys = item[0].keys()
        max_len = max(len(x["labels"]) for x in item)
        dic = {
            key: torch.tensor([x[key] + [-100 if key == "labels" else 0] * (max_len - len(x[key])) for x in item])
            for key in keys
        }
        return dic

    def build_dataset(self):
        dataset = self._iterator
        # Prepare model input
        dataset = dataset.map(self._tokenize, batched=False)
        dataset = dataset.remove_columns("turns")
        return dataset

    def build_data_loader(self, shuffle_buffer_size=None, batch_size=1,
                          num_workers=0, prefetch_factor=2):
        """build_data_loader builds DataLoader based on Dataset"""
        dataset = self.build_dataset()

        # Shuffle dataset
        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # Convert to PyTorch format
        dataset = dataset.with_format("torch")

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.__class__.collate_fn,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
        )

        return loader
