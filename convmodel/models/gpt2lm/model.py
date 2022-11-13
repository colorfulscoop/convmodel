import transformers
from .dataset import LMDataset
from typing import List, Optional
import torch
from convmodel.model import ConversationModel
from pydantic import BaseModel


class ConversationModelOutput(BaseModel):
    responses: List[str]
    context: List[str]


class GPT2LMConversationModel(ConversationModel):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: Optional[str] = None):
        # Load model via transformers
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        hf_model = transformers.GPT2LMHeadModel.from_pretrained(model_name_or_path)

        return cls(
            hf_tokenizer=hf_tokenizer,
            hf_model=hf_model,
            dataset_class=LMDataset,
            device=device,
        )

    def generate(self, context: List[str], min_new_tokens: Optional[int] = None, **kwargs) -> ConversationModelOutput:
        model_input = self._tokenizer(context)

        # Convert to Torch tensor
        model_input = {key: torch.tensor([val]) for key, val in model_input.items()}

        #
        # New parameter implementation of `min_new_tokens`
        # - min_length is set by considering minimum #tokens to generate if min_length is not provided as arguments.
        #
        min_length = None
        if "min_length" in kwargs:
            min_length = kwargs["min_length"]
        elif min_new_tokens:
            min_length = model_input["input_ids"].shape[-1] + min_new_tokens

        # Generate response
        output = self._hf_model.generate(
            **model_input,
            **kwargs,
            eos_token_id=self._tokenizer.sep_token_id,
            min_length=min_length,
        )

        responses = [self._tokenizer.decode(item) for item in output[:, model_input["input_ids"].shape[-1]:]]

        return ConversationModelOutput(
            responses=responses,
            context=context,
        )