from .lightning_model import LightningConversationModel
# Import yaml and loader
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
   

class Main:
    def export_model(self, config, ckpt_path, output_dir):
        config_yaml = yaml.load(open(config), Loader=Loader)
        lightning_model = LightningConversationModel.load_from_checkpoint(
            ckpt_path,
            **config_yaml["model"]
        )
        lightning_model.model.save_pretrained(output_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(Main)
