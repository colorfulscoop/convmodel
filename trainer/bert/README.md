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
python prepare_train_data.py --filename data/train.txt  --buffer_size 10000 >data/train.jsonl
python prepare_train_data.py --filename data/valid.txt  --buffer_size 10000 >data/valid.jsonl
python prepare_train_data.py --filename data/test.txt  --buffer_size 10000 >data/test.jsonl
```

#### Train model

```sh
python trainer.py --print_config >config.yaml
```

modify config.yaml

```sh
python trainer.py --config config.yaml
```
