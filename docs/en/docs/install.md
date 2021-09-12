# Install

First, install Python >= 3.8.

## Install PyTorch

Then install PyTorch >= 1.8,<=1.9. Please refer to [official document](https://pytorch.org/get-started/locally/) to find out correct installation for your environment.

Some examples of installtion are as follows.

### Install in Docker container without GPU

```sh
$ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
$ pip install torch==1.8.1
```

### Install in Docker container enabling GPU and CUDA 11.1

Assume that CUDA 11.1 is installed in your environment.

```sh
$ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
```

**Note:** `--ipc` option is required because share memory would not be enough because DataLoader multiprocess requires them. Refer to the [pytorch discussion](https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990) for more details.

```sh
$ apt update && apt install -y python3 python3-pip git
```

Install PyTorch which corresponds to your environment by following [the installation guide](https://pytorch.org/get-started/locally/).

For example, in CUDA 11.1 environment, PyTorch can be installed as follows.

```sh
$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

## Install convmodel

Finally, install convmodel:

```sh
$ pip install convmodel
```
