import pytorch_lightning as pl
import torch
import transformers
from train import PLModel
from convmodel.data import BlockDataset


def main(
    checkpoint, model, test_file,
    batch_size=2, prefetch_factor=10, num_workers=1,
    **train_options,
):
    config = transformers.AutoConfig.from_pretrained(model)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model)
    model = transformers.AutoModelForCausalLM.from_pretrained(model)

    test_dataset = BlockDataset.from_file(
        block_size=config.n_ctx,
        tokenizer=tokenizer,
        filename=test_file,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        collate_fn=BlockDataset.collate_fn,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
    )

    # pl_model = PLModel(model=model, config=config, lr=0, num_warmup_steps=0, num_training_steps=0)
    pl_model = PLModel.load_from_checkpoint(checkpoint, config=config, lr=0, num_warmup_steps=0, num_training_steps=0)
    trainer = pl.Trainer(**train_options)
    trainer.test(model=pl_model, test_dataloaders=test_loader)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
