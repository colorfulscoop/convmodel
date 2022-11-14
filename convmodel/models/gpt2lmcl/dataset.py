import torch
from convmodel.tokenizer import ConversationTokenizer
import random
from datasets import interleave_datasets


class LMWithClassificationDataset:
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
        self._iterator = iterator
        self._tokenizer = tokenizer
        self._max_len = max_len

    def _tokenize(self, sample):
        model_input = self._tokenizer(sample["turns"])
        cl_labels = sample["cl_labels"]

        if cl_labels == 1:
            lm_labels = model_input["input_ids"]
        elif cl_labels == 0:
            lm_labels = [-100] * len(model_input["input_ids"])
        else:
            raise Exception()

        # Add labels field for LM head
        model_input["lm_labels"] = lm_labels

        # ConversationTokenizer currently does not support truncate option while Hugging Face tokenizers suport
        # https://huggingface.co/docs/transformers/pad_truncation
        # So the truncate needs to be manually implemented.
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
        max_len = max(len(x["input_ids"]) for x in item)

        # 0 must be equal to tokenizer.pad_token_id
        dic = dict()
        for key in keys:
            if key == "cl_labels":
                dic[key] = torch.tensor([x["cl_labels"] for x in item])
            else:
                dic[key] = torch.tensor(
                    [x[key] + ([-100 if key == "lm_labels" else 0] * (max_len - len(x[key])))
                     for x in item])
        return dic

    def _shuffle_last_utterance(self, batch, shuffle_func):
        turns = batch["turns"]
        last_turns = [turn[-1] for turn in turns]
        shuffle_func(last_turns)

        return {
            "turns": [
                turn[:-1] + [last_turns[i]]
                for i, turn in enumerate(batch["turns"])
            ],
            "cl_labels": [0] * len(turns)
        }

    def build_dataset(self, batch_size=1000, shuffle_func=random.shuffle):
        pos = self._iterator.map(lambda item: {"cl_labels": 1}, batched=False)
        # Need to parametrize batch_size
        neg = self._iterator.map(
            lambda batch: self._shuffle_last_utterance(batch, shuffle_func=shuffle_func),
            batched=True,
            batch_size=batch_size
        )
        dataset = interleave_datasets([pos, neg])
        dataset = dataset.map(self._tokenize, batched=False)
        dataset = dataset.remove_columns("turns")
        return dataset

    def build_data_loader(self, shuffle_buffer_size=None, batch_size=1,
                          num_workers=0, prefetch_factor=2,
                          neg_shuffle_batch_size=1000,
                          neg_shuffle_func=random.shuffle,
                          ):
        """build_data_loader builds DataLoader based on Dataset"""

        dataset = self.build_dataset(batch_size=neg_shuffle_batch_size,
                                     shuffle_func=neg_shuffle_func)

        if shuffle_buffer_size:
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

        # Need to call with_format to avoid error `IterableDataset’ has no len()
        # https://discuss.huggingface.co/t/using-iterabledataset-with-trainer-iterabledataset-has-no-len/15790
        #
        # Note:
        #   When dataset is IterableDataset (like this function), with_fomrat method does not convert list to torch.tensor.it keeps it as is.
        #   However, when dataset is Dataset (not iterable), with_format convert list to torch.tensor.
        #   This may be a bug of datasets
        dataset = dataset.with_format("torch")

        loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            collate_fn=self.__class__.collate_fn,
            prefetch_factor=prefetch_factor,
            num_workers=num_workers,
        )

        return loader
