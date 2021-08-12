from .lightning_model import LightningConversationModel
# Import yaml and loader
import yaml
import pytorch_lightning as pl
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


def load_model(config, ckpt_path):
    config_yaml = yaml.load(open(config), Loader=Loader)
    lightning_model = LightningConversationModel.load_from_checkpoint(
        ckpt_path,
        **config_yaml["model"]
    )
    return config_yaml, lightning_model


class Main:
    def export_model(self, config, ckpt_path, output_dir):
        _, lightning_model = load_model(config, ckpt_path)
        lightning_model.model.save_pretrained(output_dir)

    def test_model(self, config, ckpt_path):
        config_yaml, lightning_model = load_model(config, ckpt_path)
        trainer = pl.Trainer(**config_yaml["trainer"])
        trainer.test(model=lightning_model)


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
