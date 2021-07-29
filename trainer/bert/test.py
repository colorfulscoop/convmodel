"""
This script runs test with trained model.
Official PyTorch Lightning documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/test_set.html
"""

import pytorch_lightning as pl
from model import PLBertForPreTraining
# Import yaml and loader
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def main(config, ckpt_path):
    config_yaml = yaml.load(open(config), Loader=Loader)
    model = PLBertForPreTraining(**config_yaml["model"])
    trainer = pl.Trainer(**config_yaml["trainer"])
    trainer.test(model=model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
