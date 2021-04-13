# Language Modeling

This directory contains a training script for language models.
This script utilizes [Hugging Face transformers](https://github.com/huggingface/transformers) for language models and [PyTorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training models.

## Preparation

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
```

Note: `--ipc` option is required because share memory would not be enough because DataLoader multiprocess requires them. Refer to the URL for more details. https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990

```sh
$ apt update && apt install -y python3 python3-pip git
```

Install PyTorch which corresponds to your environment by following [the installation guide](https://pytorch.org/get-started/locally/).

For example, in CUDA 11.1 environment, PyTorch can be installed as follows.

 ```sh
$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

Then install packages.

```sh
$ pip3 install transformers==4.4.2 sentencepiece==0.1.95 pytorch-lightning==1.2.7 fire==0.4.0 git+https://github.com/colorfulscoop/torchlang
```

### DeepSpeed

Install following packages as well.

```sh
$ apt install -y libopenmpi-dev llvm cmake
$ pip3 install deepspeed mpi4py
```

Finally, check your DeepSpeed status.

```sh
$ ds_report
```

## Examples of Training

Train 4 epochs with batch size 2

```sh
$ python3 train.py --tokenizer_model=colorfulscoop/gpt2-small-ja --save_model_dir=model --train_file=data/train.txt --valid_file=data/valid.txt --gpus=1 --precision=16 --lr=1e-4 --seed=1000 --max_epochs=4 --batch_size 2
```

Train 4 epochs with updating parameter in every 32 samples (Each batch has 2 samples. Therefore parameters need to be updated after forwarding 16 times)

```sh
$ python3 train.py --tokenizer_model=colorfulscoop/gpt2-small-ja --save_model_dir=model --train_file=data/train.txt --valid_file=data/valid.txt --gpus=1 --precision=16 --lr=1e-4 --seed=1000 --max_epochs=4 --batch_size 2 --accumulate_grad_batches 16
```

Scheduler can be enabled with with `--num_training_steps` option

```sh
$ python3 train.py --tokenizer_model=colorfulscoop/gpt2-small-ja --save_model_dir=model --train_file=data/train.txt --valid_file=data/valid.txt --gpus=1 --precision=16 --lr=1e-4 --seed=1000 --max_epochs=4 --batch_size 2 --accumulate_grad_batches 16 --num_training_steps 1000000
```

## Check log

docker container run -p 6006:6006 -v $(pwd):/work -w /work --rm -it tensorflow/tensorflow:2.4.1-gpu tensorboard --logdir lightning_logs --host 0.0.0.0
