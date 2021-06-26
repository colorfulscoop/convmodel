import pytorch_lightning as pl
import torch
import transformers
from convmodel.data import BertForPreTrainingDataset
from convmodel.data import BufferedShuffleDataset


class PLBertForPreTraining(pl.LightningModule):
    def __init__(
        self,
        tokenizer_model,
        train_file,
        valid_file,
        test_file,
        from_pretrained=None,
        block_size=1024,
        # [Model config]
        # base size
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        # large size
        #hidden_size=1024, num_hidden_layers=24, num_attention_heads=16,
        # [DataLoader options]
        batch_size=2,
        prefetch_factor=10,
        num_workers=1,
        shuffle_buffer_size=1000,
        lr=1e-4, num_warmup_steps=0, num_training_steps=None,
    ):
        super().__init__()

        # Load tokenzier
        tokenizer = transformers.AlbertTokenizer.from_pretrained(tokenizer_model)
        self._tokenizer = tokenizer

        # Load or initialize model
        if from_pretrained:
            config = transformers.BertConfig.from_pretrained(from_pretrained)
            model = transformers.BertForPreTraining.from_pretrained(from_pretrained)
        else:
            # Prepare model
            config = transformers.BertConfig(
                vocab_size=len(tokenizer),
                tokenizer_class=tokenizer.__class__.__name__,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=tokenizer.sep_token_id,
                cls_token_id=tokenizer.cls_token_id,
                unk_token_id=tokenizer.unk_token_id,
                mask_token_id=tokenizer.mask_token_id,
                #
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads,
            )
            model = transformers.BertForPreTraining(config)

        self.model = model
        self._config = config

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
        output = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["labels"],
            next_sentence_label=batch["next_sentence_label"]
        )
        return output.loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(batch=batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-loop
        """
        loss = self.forward(batch=batch)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = sum(validation_step_outputs) / len(validation_step_outputs)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/test_set.html?highlight=test#test-after-fit
        """
        loss = self.forward(batch=batch)
        return loss

    def test_epoch_end(self, test_step_outputs):
        test_loss = sum(test_step_outputs) / len(test_step_outputs)
        self.log("test_loss", test_loss)

    def train_dataloader(self):
        # Load data
        train_dataset = BertForPreTrainingDataset.from_jsonl(
            filename=self._train_file,
            tokenizer=self._tokenizer,
            max_seq_len=self._config.max_position_embeddings,
        )
        shuffled_train_dataset = BufferedShuffleDataset(
            train_dataset,
            buffer_size=self._shuffle_buffer_size,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=shuffled_train_dataset,
            batch_size=self._batch_size,
            collate_fn=BertForPreTrainingDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return train_loader

    def valid_dataloader(self):
        valid_dataset = BertForPreTrainingDataset.from_jsonl(
            filename=self._valid_file,
            tokenizer=self._tokenizer,
            max_seq_len=self._config.max_position_embeddings,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=self._batch_size,
            collate_fn=BertForPreTrainingDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_dataset = BertForPreTrainingDataset.from_jsonl(
            filename=self._test_file,
            tokenizer=self._tokenizer,
            max_seq_len=self._config.max_position_embeddings,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self._batch_size,
            collate_fn=BertForPreTrainingDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return test_loader
