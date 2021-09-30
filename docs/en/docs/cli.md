# CLI (Experimental)

!!! warning
    Currently convmodel CLI is an experimental feature.

CLI provides commands to continue "fit - eval - try" loop to improve your conversation model.


To use convmodel CLI, install convmodel with `cli` option.

```sh
$ pip install convmodel[cli]
```

## fit - train your model

This is a simple wrapper interface of `ConversationModel.fit` method.
You can simply run training based on a json config file.

All you need to do is preparing json config file.
You can generate a template file as follows.

```sh
$ python -m convmodel fit --print_config >config.json
$ cat config.json
{
  "pretrained_model_or_path": "",
  "output_path": "",
  "train_file": "",
  "valid_file": "",
  "eval_file": null,
  "save_best_model": false,
  "device": null,
  "lr": 0.0001,
  "warmup_steps": 10000,
  "use_amp": false,
  "epochs": 1,
  "accumulation_steps": 1,
  "show_progress_bar": true,
  "log_steps": 100,
  "shuffle_buffer_size": null,
  "batch_size": 1,
  "num_workers": 0,
  "prefetch_factor": 2,
  "seed": null,
  "deterministic": false
}
```

At least you need to edit 4 parameters.

| Parameter | Description | Example value |
| --- | --- | --- |
| pretrained_model_or_path | Pretrained model path to use | `gpt2` |
| output_path | Path to save your trained model | `model` |
| train_file | Path for training data file. The format should be Json Lines. Each line needs to contain a list of string, which are one example of conversation | `input/train.jsonl` |
| valid_file | Path for validation data file. Format is the same as `train_file`. | `input/valid.jsonl` |

The format of train and valid files should be [JSON Lines](https://jsonlines.org/).
Each line should be a list of utterances of each conversation.

One example of the files is as follows.

```sh
$ head -n3 input/train.jsonl
["Hello", "Hi, how are you?", "Good, thank you, how about you?", "Good, thanks!"]
["I am hungry", "How about eating pizza?"]
["Tired...", "Let's have a break!", "Nice idea!"]
```

After preparing config json file, you can start training by `fit` CLI command.

```sh
$ python -m convmodel fit --config config.json
```

Once training completes, you can load the trained model from `output_path` for `ConversationModel`.

```py
>>> from convmodel import ConversationModel
>>> model = ConversationModel.from_pretrained("model")
```

## eval - evaluate your model

You can evaluate your model towards `eval_file` defined in config file.

```sh
$ python -m convmodel eval --config config.json
```

## try - try your model

convmodel CLI provides streamlit interface to test conversation of your model.

```sh
# Default server address and port will be used
$ python -m convmodel try

# You can set server port via --server.port option
$ python -m convmodel try --server.port 8080

# You can set server address and port via --server.address
$ python -m convmodel try --server.port 8080 --server.address 0.0.0.0

# You can check all options by --help
$ python -m convmodel try --help
```

As default, you can access UI via http://localhost:8501/ .

![convmodel_streamlit](img/convmodel_streamlit.jpg)
