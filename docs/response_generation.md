# Response Generation

Once ConversationModel is trained, the model can be loaded via `from_pretrained`.
Assume your trained model is saved under `model` directory. Then you can load it as follows.

```py
>>> from convmodel import ConversationModel
>>> model = ConversationModel.from_pretrained("model")
```

After loading your model, `generate` method generates response from a given context.
All the options defined in a transformers [generate method](https://huggingface.co/transformers/main_classes/model.html?highlight=generate#transformers.generation_utils.GenerationMixin.generate) can be also available in the `generate` method.
One example below uses `top_p` and `top_k` options with `do_sample`.

```py
>>> from convmodel import ConversationModel
>>> model = ConversationModel.from_pretrained("model")
>>> model.generate(context=["こんにちは"], do_sample=True, top_p=0.95, top_k=50)
ConversationModelOutput(responses=['こんにちは♪'], context=['こんにちは'])
```
