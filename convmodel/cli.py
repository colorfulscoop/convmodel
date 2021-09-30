from convmodel import ConversationModel
from convmodel import ConversationExample
from typing import Optional
from pkg_resources import resource_filename
from pydantic import BaseModel
import subprocess
import json


class JsonLinesIterator:
    """Json Lines data loader used in fit command"""
    def __init__(self, filename: str):
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as fd:
            for line in fd:
                yield ConversationExample(conversation=json.loads(line))


class FitConfig(BaseModel):
    pretrained_model_or_path: str
    output_path: str
    train_file: str
    valid_file: str
    eval_file: Optional[str] = None
    save_best_model: bool = False
    device: Optional[str] = None
    lr: float = 1e-4
    warmup_steps: int = 10000
    use_amp: bool = False
    epochs: int = 1
    accumulation_steps: int = 1
    show_progress_bar: bool = True
    log_steps: int = 100
    shuffle_buffer_size: int = None
    batch_size: int = 1
    num_workers: int = 0
    prefetch_factor: int = 2
    seed: Optional[int] = None
    deterministic: bool = False


class CliEntrypoint:
    def __init__(self):
        # Use setattr to avoid invalid syntac of `self.try = ...`
        setattr(self, "try", self._run_streamlit)

    def _run_streamlit(self, **kwargs):
        options = []
        for key, val in kwargs.items():
            options.extend([f"--{key}", f"{val}"])

        path_to_app = resource_filename("convmodel", "app.py")
        cmd = ["streamlit", "run", path_to_app, *options]
        print(f"Command to execute in subprocess:\n")
        print(f"\t$", " ".join(cmd))

        subprocess.run(cmd)

    def fit(self, config: Optional[str]=None, print_config: bool=False):
        if print_config:
            config = FitConfig(pretrained_model_or_path="", output_path="", train_file="", valid_file="")
            print(config.json(indent=2))
        elif config:
            config = FitConfig.parse_file(config)
            # Prepare model
            model = ConversationModel.from_pretrained(config.pretrained_model_or_path, device=config.device)

            # Prepare data
            train_data = JsonLinesIterator(config.train_file)
            valid_data = JsonLinesIterator(config.valid_file)

            # Fit model
            model.fit(
                train_iterator=train_data,
                valid_iterator=valid_data,
                save_best_model=config.save_best_model,
                output_path=config.output_path,
                use_amp=config.use_amp,
                epochs=config.epochs,
                accumulation_steps=config.accumulation_steps,
                show_progress_bar=config.show_progress_bar,
                log_steps=config.log_steps,
                shuffle_buffer_size=config.shuffle_buffer_size,
                batch_size=config.batch_size,
                num_workers=config.num_workers,
                prefetch_factor=config.prefetch_factor,
                seed=config.seed,
                deterministic=config.deterministic,
            )
        else:
            raise Exception("Specify one of the options from --config or --print_config")

    def eval(self, config):
        config = FitConfig.parse_file(config)
        # Prepare model
        model = ConversationModel.from_pretrained(config.pretrained_model_or_path, device=config.device)

        # Prepare data
        eval_data = JsonLinesIterator(config.eval_file)

        # Fit model
        model.eval(
            eval_iterator=eval_data,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            prefetch_factor=config.prefetch_factor
        )
