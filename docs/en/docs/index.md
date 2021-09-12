# convmodel

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

Please refer to [Model Training](model_training.md) for more details of training.

ConversationModel adopts simple input schema by concatenating each utterance with `<sep>` token as below.
Please refer to [Model Architecture Overview](model_architecture_overview.md) for more details.

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| word | \<sep\> | Hello | \<sep\> | How | are | you | \<sep\> | Good | thank | you |
| input_ids | 50256 | 15496 | 50256 | 2437 | 389 | 345 | 50256 | 10248 | 5875 | 345 |
| token_type_ids | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| | | | | | | | ↓ | ↓ | ↓ | ↓ |
| generated word | - | - | - | - | - | - | Good | thank | you | \<sep\> |


Enjoy talking with your conversational AI :wink:
