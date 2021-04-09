# torchlang

![](https://github.com/colorfulscoop/tfdlg/workflows/torchlang/badge.svg)

**torchlang** provides PyTorch features to support your language modeling 😉.

## Install

First, install Python >= 3.6 first.

Then install PyTorch >= 1.8. Please refer to [official document](https://pytorch.org/get-started/locally/)
to find out correct installation for your environment.

Finally, install torchlang.

```sh
pip install git+https://github.com/colorfulscoop/torchlang
```

## Tutorial

This tutorial uses a tokenizer from Hugging Face's [transformers](https://github.com/huggingface/transformers) package.
Therefore install the package before running codes.

```sh
$ pip install transformers
```

Please prepare your text data in advance. This tutorial uses this markdown file as a text data.

### Dataset

`BlockDataset` enables you to create PyTorch Dataset and DataLoader for modeling languages.


```py
>>> import torch
>>> from torchlang.data import BlockDataset
>>> import transformers
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
>>> dataset = BlockDataset.from_file(filename="README.md", tokenizer=tokenizer, block_size=8)
>>> next(iter(dataset))
{'input_ids': [2, 28034, 17204, 0, 58, 16151, 5450, 1378], 'labels': [28034, 17204, 0, 58, 16151, 5450, 1378, 12567]}
```

DataLoader can be initialized with `BlockDataset.collate_fn`.

```
>>> data_loader = torch.utils.data.DataLoader(dataset, collate_fn=BlockDataset.collate_fn)
>>> next(iter(data_loader))
{'input_ids': tensor([[    2, 28034, 17204,     0,    58, 16151,  5450,  1378]]), 'labels': tensor([[28034, 17204,     0,    58, 16151,  5450,  1378, 12567]])}
```

`BlockDataset` implements [iterable-style dataset](https://pytorch.org/docs/stable/data.html#iterable-style-datasets).
Therefore, to shuffle it in a training step, please combine it with `torch.utils.data.BufferedShuffleDataset`

```py
# Shuffle dataset
>>> shuffled_dataset = torch.utils.data.BufferedShuffleDataset(dataset, buffer_size=100)
>>> shuffled_data_loader = torch.utils.data.DataLoader(shuffled_dataset, collate_fn=BlockDataset.collate_fn)
>>> next(iter(shuffled_data_loader))
{'input_ids': tensor([[ 2682,    11,  1596, 18638,    11,   657,    11,  7618]]), 'labels': tensor([[   11,  1596, 18638,    11,   657,    11,  7618,    11]])}
```
