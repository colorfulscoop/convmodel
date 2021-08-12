from convmodel.tokenizer import ConversationTokenizer
import transformers 
import torch
from typing import List, Any
from pydantic import BaseModel


class ConversationModelOutput(BaseModel):
    responses: List[str]
    model_output: Any


class ConversationModel:
    def __init__(self, hf_tokenizer: ConversationTokenizer, hf_model: transformers.GPT2LMHeadModel):
        # Convert transformers Tokenizer to ConversationTokenizer
        tokenizer = ConversationTokenizer(tokenizer=hf_tokenizer)

        self._tokenizer = tokenizer
        self._hf_model = hf_model

    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        # Load model via transformers
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path)

        return cls(hf_tokenizer=hf_tokenizer, hf_model=hf_model)

    @property
    def hf_tokenizer(self):
        return self._tokenizer.hf_tokenizer

    @property
    def hf_model(self):
        return self._hf_model

    def save_pretrained(self, save_directory):
        self._tokenizer.save_pretrained(save_directory)
        self._hf_model.save_pretrained(save_directory)

    def generate(self, context: List[str], **kwargs):
        model_input = self._tokenizer(context)

        # Convert to Torch tensor
        model_input = {key: torch.tensor([val]) for key, val in model_input.items()}
        output = self._hf_model.generate(
            **model_input,
            **kwargs,
            eos_token_id=self._tokenizer.sep_token_id
        )

        responses = [self._tokenizer.decode(item) for item in output[:, model_input["input_ids"].shape[-1]:]]

        return ConversationModelOutput(
            responses=responses,
            model_output=output
        )
