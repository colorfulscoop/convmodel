import torch
from convmodel.tokenizer import ConversationTokenizer


class ConversationDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator, tokenizer: ConversationTokenizer):
        super().__init__()
        self._generator = generator
        self._tokenizer = tokenizer

    def __iter__(self):
        """
            Yields (List[int])
        """
        for context in self._generator():
            model_input = self._tokenizer(context)
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
