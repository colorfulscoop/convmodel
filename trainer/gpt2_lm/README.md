# Training GPT2 Language Model

This directory contains a training script for language models.
This script utilizes [Hugging Face transformers](https://github.com/huggingface/transformers) for language models and [PyTorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training models.

## Preparation

By folllowing [README](https://github.com/colorfulscoop/convmodel), install PyTorch and convmodel first.

Then install other dependent packages from requirements.txt:

```sh
$ pip3 install git+https://github.com/colorfulscoop/convmodel
$ pip3 install -r requirements.txt
```

## Usage

If you do not have tokenizer, use `train_tokenizer.py` before training model to prepare your tokenizer model.

### Training

Training script uses [PyTorch Lightning CLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html).

First, create your config file.

```sh
$ python trainer.py --print_config > default_config.yaml
```

Second, modify the default_config.yaml file to set up your parameters for training.
Following parameters are some of them to recommend to set up.

**For trainer parameters:**

| params | what to set | example |
| --- | --- | --- |
| trainer.seed_everything | Set an int value for reproducibility | 1000 |
| trainer.max_epochs | Set the number of epochs | 10 |
| trainer.deterministic | Set true to ensure reproducibility while training on GPU | true |
| [trainer.precision](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#accumulate-gradients) | Set 16 for 16-bit training if while training on GPU | 16 |
| [trainer.accumulate_grad_batches](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#accumulate-gradients) | Set the number of batches to calculate gradient for updating parameters | 16 |
| [trainer.gradient_clip_val](https://pytorch-lightning.readthedocs.io/en/stable/advanced/training_tricks.html#gradient-clipping) | Set a value to clip gradient | 1 |

Following setting might be useful when you need to monitor values in callbacks:

```yaml
trainer:
  callbacks:
    - class_path: pytorch_lightning.callbacks.LearningRateMonitor
    - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
```

**For model parameters:**

The default parameters for GPT2 is for small model. You can specify `model.block_size`, `model.n_layer`, `model.n_head`, `model.n_embd` parameters to change the network size.

| model.tokenizer_model | Set GPT2 tokenizer model on [Hugging Face Model Hub](https://huggingface.co/models) | colorfulscoop/gpt2-small-ja |
| --- | --- | --- |
| model.train_file | Set text file for train your language model | data/train.txt |
| model.valid_file | Set text file for validate your language model | data/valid.txt |
| model.test_file | Set text file for test your language model | data/test.txt |
| model.block_size | Set context size of GPT2 model | 1024 |
| model.n_layer | Set the number of layers of GPT2 model | 12 |
| model.n_head | Set the number of attention head of GPT2 model | 12 |
| model.n_embd | Set the embedding dimension of GPT2 model | 768 |

Assume that the modified config file is saved as `config.yaml`

Then run training with the config file:

```sh
$ python trainer.py --config config.yaml
```

While training, you can check log via TensorBoard

```sh
docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
```

### Testing

Once your model is trained, use `test.py` script to measure loss and PPL metrics.
You can specify a config file and checkpoint which PyTorch Lightning automatically saves.

```sh
$ python test.py --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch\=2-step\=8.ckpt
```
