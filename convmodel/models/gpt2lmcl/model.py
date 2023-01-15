import transformers
from .hf_model import GPT2LMHeadWithClassification
from .dataset import LMWithClassificationDataset
from typing import List, Optional, Any
import torch
from convmodel.model import ConversationModel
from pydantic import BaseModel


class ConversationModelOutput(BaseModel):
    responses: List[str]
    context: List[str]
    score: List[float]


class GPT2LMWithClassificationConversationModel(ConversationModel):
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, device: Optional[str] = None):
        # Load model via transformers
        hf_tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
        hf_model = GPT2LMHeadWithClassification.from_pretrained(model_name_or_path)

        return cls(
            hf_tokenizer=hf_tokenizer,
            hf_model=hf_model,
            dataset_class=LMWithClassificationDataset,
            device=device,
        )

    def generate(self, context: List[str], min_new_tokens: Optional[int] = None, **kwargs) -> ConversationModelOutput:
        model_input = self._tokenizer(context)

        # Convert to Torch tensor
        model_input = {key: torch.tensor([val]).to(self.device) for key, val in model_input.items()}

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
        with torch.no_grad():
            output = self._hf_model.generate(
                **model_input,
                **kwargs,
                eos_token_id=self._tokenizer.sep_token_id,
                min_length=min_length,
            )
            responses = [self._tokenizer.decode(item) for item in output[:, model_input["input_ids"].shape[-1]:]]

            # Calculate score by classification head
            cl_input = {"input_ids": output}
            cl_output = self._hf_model(**cl_input)
            cl_logits = cl_output.cl_logits.cpu().numpy()  # shape [len_seq, 2]
            score = torch.nn.functional.softmax(cl_logits, dim=1)[:, 1].numpy().tolist()

            # Sort results
            score, responses = zip(*sorted(zip(score, responses), reverse=True))

            print(output)

            return ConversationModelOutput(
                responses=responses,
                context=context,
                score=score,
            )
