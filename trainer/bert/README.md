# Train BERT model

## Dataset

Prepare text data and split it into train, valid and test.

## Train tokenizer

```sh
python train_tokenizer.py --train_file data/train.txt
```

## Train model

### Prepare dataset

 Then convert them to jsonl format for training.

```sh
python prepare_train_data.py --filename data/train.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw False >data/train.jsonl
python prepare_train_data.py --filename data/valid.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw True >data/valid.jsonl
python prepare_train_data.py --filename data/test.txt  --buffer_size 10000 --tokenizer_model output/model --max_seq_len 512 --seed 1000 --get_raw True >data/test.jsonl
```

#### Train model

```sh
python trainer.py --print_config >config.yaml
```

modify config.yaml

```sh
python trainer.py --config config.yaml
```

### Export transformers model

```sh
python export_model.py --config lightning_logs/version_20/config.yaml --ckpt_path lightning_logs/version_20/checkpoints/epoch\=18-step\=949.ckpt --output_dir model
```
