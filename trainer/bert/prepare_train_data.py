from convmodel.data import BertForPreTrainingDataset
from convmodel.data import BertSample
from convmodel.data import BufferedShuffleDataset
import transformers
import random
import json


def iter_file(filename):
    return (line.strip("\n") for line in open(filename))


def generate_sample(filename, buffer_size):
    original_generator = iter_file(filename)
    shuffled_generator = BufferedShuffleDataset(iter_file(filename), buffer_size=buffer_size)

    prev_sentence = None
    for sentence, random_sentence in zip(original_generator, shuffled_generator):
        if not prev_sentence:
            prev_sentence = sentence
            continue
        next_sentence_label = random.choice([0, 1])
        # 0 means the next sentence is continued, while 1 means the next sentence is randomly picked up
        s = BertSample(
            sentence=prev_sentence,
            next_sentence=sentence if next_sentence_label == 0 else random_sentence,
            next_sentence_label=next_sentence_label
        )
        yield s
        prev_sentence = sentence


def main(filename, buffer_size, tokenizer_model, max_seq_len, seed, use_fast=False):
    random.seed(seed)
    # use_fast = False と True で挙動が違う
    # False が期待した挙動なので、現状Falseを使う
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_model, use_fast=use_fast)
    dataset = BertForPreTrainingDataset.from_generator(
        generator_fn=lambda: generate_sample(filename, buffer_size),
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
    )

    for item in dataset:
        print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
