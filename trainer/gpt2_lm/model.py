import pytorch_lightning as pl
import torch
import transformers
from convmodel.data import BlockDataset


class PLGPT2LMHeadModel(pl.LightningModule):
    def __init__(
        self,
        tokenizer_model,
        train_file,
        valid_file,
        test_file,
        from_pretrained=None,
        block_size=1024,
        # [Model config]
        # for small
        n_layer=12, n_head=12, n_embd=768,
        # for medium -> n_layer=24, n_head=16, n_embd=1024
        # for large  -> n_layer=36, n_head=20, n_embd=5120
        # for XL     -> n_layer=48, n_head=24, n_embd=6400
        # [DataLoader options]
        batch_size=2,
        prefetch_factor=10,
        num_workers=1,
        shuffle_buffer_size=1000,
        lr=1e-4, num_warmup_steps=0, num_training_steps=None,
    ):
        super().__init__()

        # Load tokenzier
        tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model)
        self._tokenizer = tokenizer

        # Load or initialize model
        if from_pretrained:
            config = transformers.GPT2Config.from_pretrained(from_pretrained)
            model = transformers.GPT2LMHeadModel.from_pretrained(from_pretrained)
        else:
            # Prepare model
            config = transformers.GPT2Config(
                vocab_size=len(tokenizer),
                tokenizer_class=tokenizer.__class__.__name__,
                bos_token_id=tokenizer.bos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                sep_token_id=tokenizer.sep_token_id,
                cls_token_id=tokenizer.cls_token_id,
                unk_token_id=tokenizer.unk_token_id,
                #
                n_layer=n_layer, n_head=n_head, n_embd=n_embd
            )
            model = transformers.GPT2LMHeadModel(config)

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

    def forward(self, model, pad_token_id, src, tgt):
        output = model(input_ids=src)
        logits = output.logits  # shape: (batch_size, input_len, vocab_size)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
        loss = loss_fn(
            input=logits.view(-1, logits.shape[-1]),
            target=tgt.view(-1)
        )
        return loss

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

    def training_step(self, batch, batch_idx):
        src, tgt = batch["input_ids"], batch["labels"]
        loss = self(self.model, self._config.pad_token_id, src, tgt)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#validation-loop
        """
        src, tgt = batch["input_ids"], batch["labels"]
        loss = self(self.model, self._config.pad_token_id, src, tgt)
        return loss

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = sum(validation_step_outputs) / len(validation_step_outputs)
        self.log("val_loss", val_loss)

    def test_step(self, batch, batch_idx):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/test_set.html?highlight=test#test-after-fit
        """
        src, tgt = batch["input_ids"], batch["labels"]
        loss = self(self.model, self._config.pad_token_id, src, tgt)
        return loss

    def test_epoch_end(self, test_step_outputs):
        test_loss = sum(test_step_outputs) / len(test_step_outputs)
        self.log("test_loss", test_loss)

        test_ppl = torch.exp(test_loss)
        self.log("test_ppl", test_ppl)

    def train_dataloader(self):
        # Load data
        train_dataset = BlockDataset.from_file(
            block_size=self._config.n_ctx,
            tokenizer=self._tokenizer,
            filename=self._train_file,
        )
        shuffled_train_dataset = torch.utils.data.BufferedShuffleDataset(
            train_dataset,
            buffer_size=self._shuffle_buffer_size,
        )
        train_loader = torch.utils.data.DataLoader(
            dataset=shuffled_train_dataset,
            batch_size=self._batch_size,
            collate_fn=BlockDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return train_loader

    def valid_dataloader(self):
        valid_dataset = BlockDataset.from_file(
            block_size=self._config.n_ctx,
            tokenizer=self._tokenizer,
            filename=self._valid_file,
        )
        valid_loader = torch.utils.data.DataLoader(
            dataset=valid_dataset,
            batch_size=self._batch_size,
            collate_fn=BlockDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return valid_loader

    def test_dataloader(self):
        test_dataset = BlockDataset.from_file(
            block_size=self._config.n_ctx,
            tokenizer=self._tokenizer,
            filename=self._test_file,
        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=self._batch_size,
            collate_fn=BlockDataset.collate_fn,
            prefetch_factor=self._prefetch_factor,
            num_workers=self._num_workers,
        )
        return test_loader
