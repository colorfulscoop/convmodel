# Model Training

convmodel provides training script of yoru conversation model utilizing [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html) by [PyTorch Lightning](https://www.pytorchlightning.ai/).

## Prepare data

convmodel requires [JSON Lines](https://jsonlines.org/) format data for train/valid/test data.
Each line needs to contain lists of utterances of conversation. convmodel can handle multi-turn conversation, therefore you can provide any turns in one conversation.

Here is an example of train dataset.

```sh
$ head -n3 train.jsonl
["Hello", "Hi, how are you?", "Good, thank you, how about you?", "Good, thanks!"]
["I am hungry", "How about eating pizza?"]
["Tired...", "Let's have a break!", "Nice idea!"]
```

## Config training options

convmodel utilizes [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html) for training.
You first need to generate config file for training.

```sh
$ python -m convmodel.trainer --print_config >default_config.yaml
```

Then copy the config file to modify the setting.

```sh
$ cp default_config.yaml config.yaml
```

In config.yaml model-specific parameters are defined under the `model` key, 

```yaml
model:
  pretrained_model_or_path: null
  train_file: null
  valid_file: null
  test_file: null
  batch_size: 2
  prefetch_factor: 10
  num_workers: 1
  shuffle_buffer_size: 10000
  lr: 0.0001
  num_warmup_steps: 10000
  num_training_steps: null
```

You need to set these parameters to train your model. The following table shows each parameter definition.

| Parameter | Non null value required | Description | Example of value |
| --- | --- | --- | --- |
| pretrained_model_or_path | Required | pretrained model or path to use as pretrained model of conversation model | `gpt2`, `colorfulscoop/gpt2-small-ja` |
| train_file | Required | File path of train data | `data/train.jsonl` |
| valid_file | Required | File path of train data | `data/valid.jsonl` |
| test_file | Required | File path of train data | `data/test.jsonl` |
| batch_size | Required | `batch_size` parameter used in DataLoader | `2` |
| prefetch_factor | Required | `prefetch_factor` parameter used in DataLoader | `10` |
| shuffle_buffer_size | Required | Shuffle buffer size used when shuffling train data | `10000` |
| lr | Required | learning rate of Adam optimizer | `0.0001` |
| num_warmup_steps | Required | The number of warmup steps of learning rate after starting training | `10000` |
| num_training_steps | Optional | Learning rate linearly decreases to `0` while training_steps reaches to `num_training_steps`. If set to `null`, learning rate does not decrease. | `null`, `1000000` |

One example of configuration is as follows.

```sh
$ diff default_config.yaml config.yaml
1c1
< seed_everything: null
---
> seed_everything: 1000
7c7
<   gradient_clip_val: 0.0
---
>   gradient_clip_val: 1.0
13c13
<   gpus: null
---
>   gpus: 1
24c24
<   max_epochs: null
---
>   max_epochs: 10
33c33
<   val_check_interval: 1.0
---
>   val_check_interval: 10000
38c38
<   precision: 32
---
>   precision: 16
46c46
<   deterministic: false
---
>   deterministic: true
60a61,69
>   callbacks:
>     - class_path: pytorch_lightning.callbacks.LearningRateMonitor
>     - class_path: pytorch_lightning.callbacks.GPUStatsMonitor
>     - class_path: pytorch_lightning.callbacks.ModelCheckpoint
>       init_args:
>         monitor: val_loss
>         mode: min
>         every_n_train_steps: 10000
>         save_top_k: 3
62,66c71,75
<   pretrained_model_or_path: null
<   train_file: null
<   valid_file: null
<   test_file: null
<   batch_size: 2
---
>   pretrained_model_or_path: colorfulscoop/gpt2-small-ja
>   train_file: data/train.jsonl
>   valid_file: data/valid.jsonl
>   test_file:  data/test.jsonl
>   batch_size: 16
69,71c78,80
<   shuffle_buffer_size: 10000
<   lr: 0.0001
<   num_warmup_steps: 10000
---
>   shuffle_buffer_size: 1000000
>   lr: 1.0e-04
>   num_warmup_steps: 100000
```

## Train model

After configure `config.yaml` , you can start training as follows.

```sh
$ python -m convmodel.trainer --config config.yaml
```

You can check training logs via Tensorboard.

```sh
$ docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
```

## Export model

After training, `export_model` allows to export Transformers model from a Lightning checkpoint.

```sh
$ python3 -m convmodel export_model --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch\=9-step\=2409999.ckpt --output_dir model
```

You can find your model under `model` directory.

```sh
$ ls -1 model
config.json
pytorch_model.bin
special_tokens_map.json
spiece.model
tokenizer_config.json
```
