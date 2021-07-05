import torch
import json


class JsonLinesDataset(torch.utils.data.IterableDataset):
    def __init__(self, generator):
        self._generator = generator

    @classmethod
    def from_file(cls, filename):
        return cls(generator=lambda: (json.loads(line) for line in open(filename)))

    def __iter__(self):
        for item in self._generator():
            yield item
