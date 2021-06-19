import pytorch_lightning as pl
from convmodel.data import BlockDataset
from model import PLGPT2LMHeadModel
# Import yaml and loader
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def main(config, ckpt_path, output_dir):
    config_yaml = yaml.load(open(config), Loader=Loader)
    model = PLGPT2LMHeadModel.load_from_checkpoint(ckpt_path, **config_yaml["model"])

    model.model.save_pretrained(output_dir)
    model._tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
