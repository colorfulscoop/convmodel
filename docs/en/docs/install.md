# Install

Following steps are required to complete installing convmodel.

1. Prepare Python3.8+ environment
2. Install PyTorch
3. Install convmodel

## Prepare Python 3.8+

First, prepare Python 3.8+ environment.

## Install PyTorch

Then install PyTorch >= 1.8,<=1.9. Please refer to [official document](https://pytorch.org/get-started/locally/) to find out correct installation for your environment.

Some examples of installtion are as follows.

!!! example "Install in Docker container without GPU"

    ```sh
    $ docker container run -w /work -v $(pwd):/work --rm -it python:3.8.6-slim-buster bash
    (container)$ pip install torch==1.8.1
    ```

!!! example "Install in Docker container enabling GPU and CUDA 11.1"

    Assume that CUDA 11.1 is installed in your environment.

    ```sh
    $ docker container run --gpus all --ipc=host --rm -it -v $(pwd):/work -w /work nvidia/cuda:11.1-devel-ubuntu20.04 bash
    ```

    `--ipc` option is required because share memory would not be enough because DataLoader multiprocess requires them. Refer to the [pytorch discussion](https://discuss.pytorch.org/t/unable-to-write-to-file-torch-18692-1954506624/9990) for more details.

    Then install Python3.

    ```sh
    (container)$ apt update && apt install -y python3 python3-pip git
    ```

    Install PyTorch which corresponds to your environment by following [the installation guide](https://pytorch.org/get-started/locally/).
    For example, in CUDA 11.1 environment, PyTorch can be installed as follows.

    ```sh
    (container)$ pip3 install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
    ```

## Install convmodel

Finally, install convmodel from PyPI.

```sh
$ pip install convmodel
```

If you want to run tests, specify `[test]` option to install dependencies.

```sh
$ pip install convmodel[test]
```
