# convmodel

![](https://github.com/colorfulscoop/convmodel/workflows/unittest/badge.svg)

**convmodel** provides a conversation model based on [transformers](https://github.com/huggingface/transformers) GPT-2 model :wink:

:sparkles: Features :sparkles:

* Utilizes GPT2 model to generate response
* Handles multi-turn conversation
* Provides useuful interfaces to fine-tune model and generate a response from a given context

A simple example of fine-tune GPT-2 model and generate a response:

```py
from convmodel import ConversationModel
from convmodel import ConversationExample

# Load model on GPU
model = ConversationModel.from_pretrained("gpt2")

# Define training/validation examples
train_iterator = [
    ConversationExample(conversation=[
        "Hello",
        "Hi, how are you?",
        "Good, thank you, how about you?",
        "Good, thanks!"
    ]),
    ConversationExample(conversation=[
        "I am hungry",
        "How about eating pizza?"
    ]),
]
valid_iterator = [
    ConversationExample(conversation=[
        "Tired...",
        "Let's have a break!",
        "Nice idea!"
    ]),
]

# Fine-tune model
model.fit(train_iterator=train_iterator, valid_iterator=valid_iterator)

# Generate response
model.generate(context=["Hello", "How are you"], do_sample=True, top_p=0.95, top_k=50)
# Output could be like below if sufficient examples were given.
# => ConversationModelOutput(responses=['Good thank you'], context=['Hello', 'How are you'])
```

Please refer to [document](docs/en/docs/index.md) for more details of installation, model architecture and usage.

* [Install](docs/en/docs/install.md)
* [Model Architecture Overview](docs/en/docs/model_architecture_overview.md)
* [Model Training](docs/en/docs/model_training.md)
* [Response Generation](docs/en/docs/response_generation.md)
* [CLI (Experimental)](docs/en/docs/cli.md)

Enjoy talking with your conversational AI :wink:
