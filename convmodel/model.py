from convmodel.tokenizer import ConversationTokenizer
import transformers 
import torch
from typing import List, Any
from pydantic import BaseModel


class ConversationModelOutput(BaseModel):
    responses: List[str]
    model_output: Any


class ConversationModel:
    def __init__(self, tokenizer, model):
        self._tokenizer = tokenizer
        self._model = model

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        # Load model via transformers
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)

        # Convert transformers Tokenizer to ConversationTokenizer
        tokenizer = ConversationTokenizer(tokenizer=hf_tokenizer)

        return cls(tokenizer=tokenizer, model=hf_model)

    def generate(self, context: List[str], **kwargs):
        model_input = self._tokenizer(context)

        # Convert to Torch tensor
        model_input = {key: torch.tensor([val]) for key, val in model_input.items()}
        output = self._model.generate(
            **model_input,
            **kwargs,
            eos_token_id=self._tokenizer.sep_token_id
        )

        responses = [self._tokenizer.decode(item) for item in output[:, model_input["input_ids"].shape[-1]:]]

        return ConversationModelOutput(
            responses=responses,
            model_output=output
        )
