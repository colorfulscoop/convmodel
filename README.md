# convmodel

![](https://github.com/colorfulscoop/convmodel/workflows/unittest/badge.svg)

**convmodel** provides features to support your Conversational AI models 😉.

## Install

First, install Python >= 3.8 first.

### Install PyTorch

Then install PyTorch >= 1.8,<1.9. Please refer to [official document](https://pytorch.org/get-started/locally/)
to find out correct installation for your environment.

Some examples of installtion are as follows.

<details>
<summary>Example to install in Docker container without GPU</summary>


```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
$ pip install torch==1.8.1
```
</details>

<details>
<summary>Example to install in Docker container with GPU</summary>

Assume that CUDA 11.1 is installed in your environment.

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
```

**Note:** `--ipc` option is required because share memory would not be enough because DataLoader multiprocess requires them. Refer to the URL for more details. https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990

```sh
$ apt update && apt install -y python3 python3-pip git
```

Install PyTorch which corresponds to your environment by following [the installation guide](https://pytorch.org/get-started/locally/).

For example, in CUDA 11.1 environment, PyTorch can be installed as follows.

 ```sh
$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
</details>

### Install convmodel

Finally, install convmodel:

```sh
$ pip install git+https://github.com/colorfulscoop/convmodel
```

## PyTorch Lightning modules

convmodel provides [Lightning Modules](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html) to train with [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

Current available modules are under `trainer` directory.

| Name | Path | Description |
| --- | --- | --- |
| PLGPT2LMHeadModel | [trainer/gpt2_lm/](trainer/gpt2_lm) | Train language model with [transformers' GPT2LMHeadModel](https://huggingface.co/transformers/model_doc/gpt2.html#gpt2lmheadmodel) |


## Tutorial as Library

This tutorial uses a tokenizer from Hugging Face's [transformers](https://github.com/huggingface/transformers) package.
Therefore install the package before running codes.

```sh
$ pip install transformers
```

Please prepare your text data in advance. This tutorial uses this markdown file as a text data.

### Dataset

`BlockDataset` enables you to create PyTorch Dataset and DataLoader for modeling languages.

To create `BlockDataset`, prepare a tokenizer first.

```py
>>> import transformers
>>> tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
```

Then initialize `BlockDataset` with the tokenizer and your text data.

```py
>>> from convmodel.data import BlockDataset
>>> dataset = BlockDataset.from_file(filename="README.md", tokenizer=tokenizer, block_size=8)
>>> next(iter(dataset))
{'input_ids': [2, 28034, 17204, 0, 58, 16151, 5450, 1378], 'labels': [28034, 17204, 0, 58, 16151, 5450, 1378, 12567]}
```

DataLoader can be initialized with `BlockDataset.collate_fn`.

```py
>>> import torch
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
