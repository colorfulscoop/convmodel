from pathlib import Path
import sentencepiece as spm
import transformers
from transformers.tokenization_utils import AddedToken


def main(
    train_file,
    spm_model_dir="output/spm",
    tf_model_dir="output/model",
    vocab_size=32000,
    input_sentence_size=1000000,
    add_dummy_prefix=False,
    do_lower_case=False,  # Important to recognize [MASK] as it is
    keep_accents=True,  # Important to recognize 濁点 in Japanese
    # Token surfaces are aligned with AlbertTokenizer
    # https://huggingface.co/transformers/model_doc/albert.html#transformers.AlbertTokenizer
    sep_token="[SEP]",
    cls_token="[CLS]",
    pad_token="<pad>",
    unk_token="<unk>",
    bos_token="[CLS]",
    eos_token="[SEP]",
    mask_token="[MASK]",
):
    spm_model_dir = Path(spm_model_dir)
    spm_model_prefix = Path(spm_model_dir) / Path("sp")
    spm_model_path = Path(spm_model_dir) / Path("sp.model")

    train_args = dict(
        model_prefix=spm_model_prefix,
        vocab_size=vocab_size,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece=pad_token,
        unk_piece=unk_token,
        bos_piece=bos_token,
        eos_piece=eos_token,
        user_defined_symbols=[sep_token, cls_token, mask_token],
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        add_dummy_prefix=add_dummy_prefix,
    )

    # Train model
    if not spm_model_dir.exists():
        spm_model_dir.mkdir(parents=True)
    spm.SentencePieceTrainer.train(input=train_file, **train_args)
    # Convert to Transformers model
    tokenizer = transformers.AlbertTokenizer(
        str(spm_model_path),
        bos_token=bos_token,
        eos_token=eos_token,
        cls_token=cls_token,
        sep_token=sep_token,
        pad_token=pad_token,
        unk_token=unk_token,
        mask_token=mask_token,
        do_lower_case=do_lower_case,
        keep_accents=keep_accents,
    )
    print(len(tokenizer))
    tokenizer.save_pretrained(tf_model_dir)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
