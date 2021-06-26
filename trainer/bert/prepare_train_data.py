from convmodel.data import BertSample
from convmodel.data import BufferedShuffleDataset
import random


def iter_file(filename):
    return (line.strip("\n") for line in open(filename))


def main(filename, buffer_size):
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
        prev_sentence = sentence
        print(s.json(ensure_ascii=False))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
