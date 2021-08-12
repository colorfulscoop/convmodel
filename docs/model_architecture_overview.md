# Model Architecture Overview

convmodel provides `ConversationModel` class.
ConversationModel class adopts [GPT2LMHeadModel](https://huggingface.co/transformers/model_doc/gpt2.html?highlight=gpt2lmheadmodel#transformers.GPT2LMHeadModel) architecture provided by transformers library.


Although, in a initializer of `ConversationModel`, `ConversationTokenizer` is automatically initialized, let us first directly initialize `ConversationTokenizer` to see it encodes a given context to input to the model.
Assume that `ConversationTokenizer` gets a context `["Hello", "How are you"]` . Then `ConversationTokenizer` encodes it as follows.

```py
>>> from convmodel import ConversationTokenizer
>>> tokenizer = ConversationTokenizer.from_pretrained("gpt2")
>>> context = ["Hello", "How are you"]
>>> tokenizer(context)
{'input_ids': [50256, 15496, 50256, 2437, 389, 345, 50256], 'token_type_ids': [0, 0, 1, 1, 1, 1, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1]}
```

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| word | \<sep\> | Hello | \<sep\> | How | are | you | \<sep\> |
| input_ids | 50256 | 15496 | 50256 | 2437 | 389 | 345 | 50256 |
| token_type_ids | 0 | 0 | 1 | 1 | 1 | 1 | 0 |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

**Note:** if a tokenizer does not assign a value to `sep_token_id`, it is automatically set with `sep_token` of `<sep>`.

When initializing `ConversationModel`, `ConversationTokenizer` is automatically initialized inside.
`ConversationModel` implements `generate` method. In `generate` method, an input context is first encoded as above.
Then the encoded tensors are forwardded by the model to predict following tokens until `<sep>` token appears

**Note:** Here we assume that `model` directory contains a trained conversation model which was fine-tuned from gpt2 model. We will see how to train our own conversation model later.

```py
>>> from convmodel import ConversationModel
>>> model = ConversationModel.from_pretrained("model")
>>> model.generate(context, do_sample=True, top_p=0.95)
```

| position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| word | \<sep\> | Hello | \<sep\> | How | are | you | \<sep\> | Good | thank | you |
| input_ids | 50256 | 15496 | 50256 | 2437 | 389 | 345 | 50256 | 10248 | 5875 | 345 |
| token_type_ids | 0 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| attention_mask | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| | | | | | | | ↓ | ↓ | ↓ | ↓ |
| generated word | - | - | - | - | - | - | Good | thank | you | \<sep\> |
