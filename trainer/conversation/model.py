import pytorch_lightning as pl
import torch
import transformers
from convmodel.tokenizer import ConversationTokenizer
from convmodel.data import ConversationDataset
from convmodel.data import BufferedShuffleDataset
import json


class PLConversationModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_model_or_path,
        train_file,
        valid_file,
        test_file,
        batch_size=2,
        prefetch_factor=10,
        num_workers=1,
        shuffle_buffer_size=1000,
        lr=5e-5,
        num_warmup_steps=10000,
        num_training_steps=None,
    ):
        super().__init__()

        # Load model
        tokenizer = ConversationTokenizer.from_pretrained(pretrained_model_or_path)
        model = transformers.AutoModelForCausalLM.from_pretrained(pretrained_model_or_path)

        self.model = model
        self._tokenizer = tokenizer

        self._train_file = train_file
        self._valid_file = valid_file
        self._test_file = test_file
        self._batch_size = batch_size
        self._prefetch_factor = prefetch_factor
        self._num_workers = num_workers
        self._shuffle_buffer_size = shuffle_buffer_size
        self._lr = lr
        self._num_warmup_steps = num_warmup_steps
        self._num_training_steps = num_training_steps

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        if self._num_training_steps:
            scheduler = transformers.get_linear_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self._num_warmup_steps,
                num_training_steps=self._num_training_steps
            )
        else:
            scheduler = transformers.get_constant_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self._num_warmup_steps,
            )
        optimizers = [optimizer]
        schedulers = [{"scheduler": scheduler, "interval": "step"}]
        return optimizers, schedulers

    def forward(self, batch):
        output = self.model(**batch)
        return output.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch=batch)
        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-loop
        """
        loss = self.forward(batch=batch)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = sum(validation_step_outputs) / len(validation_step_outputs)
        self.log("val_loss", val_loss.item())

    def test_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/test_set.html?highlight=test#test-after-fit
        """
        loss = self.forward(batch=batch)
        return loss

    def test_epoch_end(self, test_step_outputs):
        test_loss = sum(test_step_outputs) / len(test_step_outputs)
        self.log("test_loss", test_loss.item())

    def _load_dataset(self, filename):
        return (json.loads(line) for line in open(filename))

    def train_dataloader(self):
        # Load data
        train_dataset = ConversationDataset(
            generator=lambda: self._load_dataset(self._train_file),
            tokenizer=self._tokenizer,
        )
        shuffled_train_dataset = BufferedShuffleDataset(
            train_dataset,
            buffer_size=self._shuffle_buffer_size,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=shuffled_train_dataset,
            batch_size=self._batch_size,
            collate_fn=ConversationDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ConversationDataset(
            generator=lambda: self._load_dataset(self._valid_file),
            tokenizer=self._tokenizer,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=self._batch_size,
            collate_fn=ConversationDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_dataset = ConversationDataset(
            generator=lambda: self._load_dataset(self._test_file),
            tokenizer=self._tokenizer,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self._batch_size,
            collate_fn=ConversationDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return test_loader
