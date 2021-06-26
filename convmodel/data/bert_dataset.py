import math
import random
import torch
from convmodel.data import BufferedShuffleDataset


class BertForPreTrainingDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        generator,
        encode_fn,
        sep_token_id,
        cls_token_id,
        mask_token_id,
        random_token_ids,
        max_seq_len,
        ignore_label=-100,
        buffer_size=1000,
    ):
        """
        Args:
            generator: `generator` yields a sentence.
        """
        super().__init__()
        self._generator = generator
        self._encode_fn = encode_fn
        self._sep_token_id = sep_token_id
        self._cls_token_id = cls_token_id
        self._mask_token_id = mask_token_id
        self._random_token_ids = random_token_ids
        self._max_seq_len = max_seq_len
        self._ignore_label = ignore_label
        self._buffer_size = buffer_size

    @classmethod
    def _get_random_token_ids_from_tokenizer(self, tokenizer):
        return [i for i in range(tokenizer.vocab_size) if i not in tokenizer.all_special_ids]

    @classmethod
    def from_texts(cls, texts, tokenizer, max_seq_len, **params):
        """
        Args:
            tokenizer (transformers.AutoTokenizer)
            texts (List[str])
            block_size (int)
        """
        return cls(
            generator=lambda: texts,
            encode_fn=tokenizer.encode,
            sep_token_id=tokenizer.sep_token_id,
            cls_token_id=tokenizer.cls_token_id,
            mask_token_id=tokenizer.mask_token_id,
            random_token_ids=self._get_random_token_ids_from_tokenizer(tokenizer=tokenizer),
            max_seq_len=max_seq_len,
            **params,
        )

    @classmethod
    def from_file(cls, filename, tokenizer, max_seq_len, **params):
        """
        Args:
            filename: the file needs to contain one sentence per each line.
            tokenizer (transformers.AutoTokenizer):
        """
        return cls(
            generator=lambda: (line.strip("\n") for line in open(filename)),
            encode_fn=tokenizer.encode,
            sep_token_id=tokenizer.sep_token_id,
            cls_token_id=tokenizer.cls_token_id,
            mask_token_id=tokenizer.mask_token_id,
            random_token_ids=cls._get_random_token_ids_from_tokenizer(tokenizer=tokenizer),
            max_seq_len=max_seq_len,
            **params,
        )

    def _convert_to_sample(self, s1, s2, next_sentence_label):
        s1_ids = self._encode_fn(s1)
        s2_ids = self._encode_fn(s2)

        ids = [self._cls_token_id] + s1_ids + [self._sep_token_id] + s2_ids + [self._sep_token_id]
        ids = ids[:self._max_seq_len]

        special_token_index = [0] + [1 + len(s1_ids)] + [len(ids) - 1]
        normal_token_index = [i for i in range(len(ids)) if i not in special_token_index]

        # Identify which sample is replaced with MASKs
        # Rule of replacement
        # - Select 15% of tokens out of all tokens from ids except for [CLS] and [SEP]
        # - Repalce the selected token with
        #   - 80% -> [MASK]
        #   - 10% -> Random token
        #   - 10% -> Keep the same token

        # Random sort
        shuffled_index = random.sample(normal_token_index, k=math.ceil(len(normal_token_index)*0.15))
        ilen = len(shuffled_index)

        mask_index = set(shuffled_index[:math.ceil(ilen*0.8)])
        random_token_index = set(shuffled_index[math.ceil(ilen*0.8):math.ceil(ilen*0.9)])
        same_token_index = set(shuffled_index[math.ceil(ilen*0.9):])

        input_ids = []
        labels = []
        for i in range(len(ids)):
            if i in mask_index:
                input_ids.append(self._mask_token_id)
                labels.append(ids[i])
            elif i in random_token_index:
                input_ids.append(random.choice(self._random_token_ids))
                labels.append(ids[i])
            elif i in same_token_index:
                input_ids.append(ids[i])
                labels.append(ids[i])
            else:
                input_ids.append(ids[i])
                labels.append(self._ignore_label)

        input_ids = input_ids + [0] * (self._max_seq_len - len(input_ids))
        labels = labels + [-100] * (self._max_seq_len - len(labels))
        attention_mask = [1] * len(ids) + [0] * (self._max_seq_len - len(ids))
        token_type_ids = [0] * (len(s1_ids) + 2) + [1] * (len(s2_ids) + 1) + [0] * (self._max_seq_len - len(ids))

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "next_sentence_label": next_sentence_label,
        }

    def __iter__(self):
        original_generator = self._generator()
        shuffled_generator = BufferedShuffleDataset(self._generator(), buffer_size=self._buffer_size)

        prev_text = None
        for text, random_text in zip(original_generator, shuffled_generator):
            if not prev_text:
                prev_text = text
                continue
            next_sentence_label = random.choice([0, 1])
            # 0 means the next sentence is continued, while 1 means the next sentence is randomly picked up
            yield self._convert_to_sample(s1=prev_text, s2=text if next_sentence_label == 0 else random_text, next_sentence_label=next_sentence_label)
            prev_text = text

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
