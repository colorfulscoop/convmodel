import torch
from convmodel.tokenizer import ConversationTokenizer
from convmodel.data import BufferedShuffleDataset


class ConversationDataset(torch.utils.data.IterableDataset):
    def __init__(self, iterator, tokenizer: ConversationTokenizer):
        super().__init__()
        self._iterator = iterator
        self._tokenizer = tokenizer

    def __iter__(self):
        """
            Yields (List[int])
        """
        for example in self._iterator:
            model_input = self._tokenizer(example.conversation)
            model_input["labels"] = model_input["input_ids"]
            yield model_input

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

    def build_data_loader(self, shuffle_buffer_size=None, batch_size=1,
                          num_workers=0, prefetch_factor=2):
        """build_data_loader builds DataLoader based on Dataset"""
        dataset = self
        if shuffle_buffer_size:
            dataset = BufferedShuffleDataset(dataset, buffer_size=shuffle_buffer_size)

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.__class__.collate_fn,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
        )

        return loader
