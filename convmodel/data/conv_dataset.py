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
            tokenizer_result = self._tokenizer(context)
            model_input = {
                key: val[:-1] for key, val in tokenizer_result.items()
            }
            model_input["labels"] = tokenizer_result["input_ids"][1:]
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
        dic = {
            key: torch.tensor([x[key] for x in item])
            for key in keys
        }
        return dic
