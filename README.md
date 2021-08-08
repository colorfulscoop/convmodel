# convmodel

![](https://github.com/colorfulscoop/convmodel/workflows/unittest/badge.svg)

**convmodel** provides a simple wrapper of [transformers](https://github.com/huggingface/transformers) to train and use a conversation model based on GPT2 :wink:.

## Install

First, install Python >= 3.8 first.

### Install PyTorch

Then install PyTorch >= 1.8,<=1.9. Please refer to [official document](https://pytorch.org/get-started/locally/)
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

### Model Architecture

A conversation model architecture implemented in convmodel is the same as transformers [GPT2LMHeadModel](https://huggingface.co/transformers/model_doc/gpt2.html?highlight=gpt2lmheadmodel#transformers.GPT2LMHeadModel).

When this model gets a context `["Hello", "How are you"]' , then it is encoded as follow.

```py
>>> from convmodel import ConversationTokenizer
>>> tokenizer = ConversationTokenizer.from_pretrained("gpt2")
>>> tokenizer(["Hello", "How are you"])
{'input_ids': [50256, 15496, 50256, 2437, 389, 345, 50256], 'token_type_ids': [0, 0, 1, 1, 1, 1, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
```

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| word | \<sep\> | Hello | \<sep\> | How | are | you | \<sep\> |
| input_ids | 50256 | 15496 | 50256 | 2437 | 389 | 345 | 50256 |
| token_type_ids | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

**Note:** if a tokenizer does not assign a value to `sep_token_id`, it is automatically set with `sep_token` of `<sep>`.

Then ConversationModel generates words until a `<sep>` token appears.

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| word | \<sep\> | Hello | \<sep\> | How | are | you | \<sep\> | Good | thank | you |
| input_ids | 50256 | 15496 | 50256 | 2437 | 389 | 345 | 50256 | 10248 | 5875 | 345 |
| token_type_ids | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| | | | | | | | ↓ | ↓ | ↓ | ↓ |
| generated word | - | - | - | - | - | - | Good | thank | you | \<sep\> |

## Usage

Once a ConversationModel is trained, the ConversationModel class can load it via `from_pretrained` and generate a response by `generate` method.

All the options defined in a transformers [generate method](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate) can be also available in the ConversationModel `generate` method.
A genration example as below uses `top_p` and `top_k` options with `do_sample`.

```py
>>> from convmodel import ConversationModel
>>> model = ConversationModel.from_pretrained("model")
>>> model.generate(context=["こんにちは"], do_sample=True, top_p=0.95, top_k=50)
```

## Model Training

To train your model, install convmodel with `train` option.

```sh
$ pip install git+https://github.com/colorfulscoop/convmodel[train]
```

convmodel provides training script of yoru conversation model utilizing [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html) by [PyTorch Lightning](https://www.pytorchlightning.ai/).

The instruction to train your conversation model is under [trainer/conversation/README.md]().
Please take a look at it and enjoy your own conversaiton :wink: .
