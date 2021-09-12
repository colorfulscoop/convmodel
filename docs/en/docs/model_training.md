# Model Training

## Prepare model

First you need to load GPT-2 pretrained model.
The model is easily loaded by using `from_pretrained` method defined in `ConversationModel`.

```py
from convmodel import ConversationModel

model = ConverstationModel.from_pretrained("gpt2")
```

If you want to use GPU to train, `device` option is for that.


```py
# Load model in GPU
model = ConverstationModel.from_pretrained("gpt2", device="cuda")

# Load model in CPU
model = ConverstationModel.from_pretrained("gpt2", device="cpu")
```

If you do not specify any values to `device`, GPU is used if available.

## Training data

Before training, you also need to prepare training data.

convmodel provides `ConversationExample` class which shows one example of conversation to use in training.
You need to prepare iterator objects for train/valid data to provide one `ConversationExample` object in each step in the loop.

```py
from convmodel import ConversationExample

train_iterator = [
    ConversationExample(conversation=["Hello", "Hi, how are you?", "Good, thank you, how about you?", "Good, thanks!"]),
    ConversationExample(conversation=["I am hungry", "How about eating pizza?"]),
]
valid_iterator = [
    ConvesationExample(conversation=["Tired...", "Let's have a break!", "Nice idea!"]),
]
```

Although the above example is fine, the data is usually large and difficult to load all the data on memory at the same time.
In this case, it might be better to implement iterator class to provide one example in each step in the loop.

Following example assumes each data file contains one conversation example in one line. The file format is Json Lines and each line contains a list of string which shows one conversation examples.

```sh
# Assume that training/valid data is located in under input directory.

# Training file: input/train.jsonl
$ head -n2 input/train.jsonl
["Hello", "Hi, how are you?", "Good, thank you, how about you?", "Good, thanks!"]
["I am hungry", "How about eating pizza?"]

# Validation file: input/valid.jsonl
$ head -n1 input/valid.jsonl
["Tired...", "Let's have a break!", "Nice idea!"]
```

You can implement your own iterator class to load the file and return each conversation example at each time as follows.

```py

class JsonLinesIterator:
    """Json Lines data loader used in fit command"""
    def __init__(self, filename: str):
        self._filename = filename

    def __iter__(self):
        with open(self._filename) as fd:
            for line in fd:
                yield ConversationExample(conversation=json.loads(line))


train_iterator = JsonLinesIterator("input/train.jsonl")
valid_iterator = JsonLinesIterator("input/valid.jsonl")
```

## Training

Finally, you can start training by calling `fit` method with train/valid itarators.

```sh
model = ConversationModel.fit(train_iterator=train_iterator, valid_iterator=valid_iterator)
```
