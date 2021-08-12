# Usage

## Training

convmodel provides training script of yoru conversation model utilizing [LightningCLI](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_cli.html) by [PyTorch Lightning](https://www.pytorchlightning.ai/).

### Prepare data

```sh
```

### Prepare config file

```sh
$ python -m convmodel.trainer --print_config >>config.yaml
```

Then modify config.yaml

### Train model

```sh
$ python -m convmodel.trainer --config config.yaml
```

You can check training logs via Tensorboard.

```sh
$ docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
```

After training, `export_model` allows you to export ConversationModel from a Lightning checkpoint.

```sh
$ python3 -m convmodel export_model --config lightning_logs/version_0/config.yaml --ckpt_path lightning_logs/version_0/checkpoints/epoch\=0-step\=49999.ckpt  --output_dir model
```

### Test model

```sh
```

## Response generation

Once a ConversationModel is trained, the ConversationModel class can load it via `from_pretrained` and generate a response by `generate` method.

All the options defined in a transformers [generate method](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate) can be also available in the ConversationModel `generate` method.
A genration example as below uses `top_p` and `top_k` options with `do_sample`.

```py
>>> from convmodel import ConversationModel
>>> model = ConversationModel.from_pretrained("model")
>>> model.generate(context=["こんにちは"], do_sample=True, top_p=0.95, top_k=50)
```
