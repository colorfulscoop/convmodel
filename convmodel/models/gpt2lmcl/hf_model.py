from typing import Tuple, Optional
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2LMHeadModel
from transformers.utils import ModelOutput
from dataclasses import dataclass 


@dataclass
class GPT2LMHeadWithClassificationOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    lm_loss: Optional[torch.FloatTensor] = None
    cl_loss: Optional[torch.FloatTensor] = None
    lm_logits: torch.FloatTensor = None
    cl_logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    logits: torch.FloatTensor = None


class GPT2LMHeadWithClassification(GPT2LMHeadModel):
    """
    """
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 2
        self.lm_coef = 2.0
        self.cl_coef = 1.0
        self.cl_head = nn.Linear(config.n_embd, self.num_labels, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        lm_labels: Optional[torch.LongTensor] = None,
        cl_labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # Call GPT2LMHeadModel
        lm_outputs = super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=lm_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        assert (
            (input_ids is not None) and
            (self.config.pad_token_id is not None)
        )

        batch_size, sequence_length = input_ids.shape[:2]
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            sequence_lengths = torch.ne(input_ids, self.config.pad_token_id).sum(-1) - 1

        last_hidden_states = lm_outputs.hidden_states[-1]
        logits = self.cl_head(last_hidden_states)
        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        if cl_labels is None:
            loss = None
            cl_loss = None
        else:
            loss_fct = CrossEntropyLoss()
            cl_loss = loss_fct(pooled_logits.view(-1, self.num_labels), cl_labels.view(-1))

            loss = 2 * (self.lm_coef / (self.lm_coef+self.cl_coef)) * lm_outputs.loss + (self.cl_coef / (self.lm_coef+self.cl_coef)) * cl_loss

        return GPT2LMHeadWithClassificationOutput(
            loss=loss,
            lm_loss=lm_outputs.loss,
            cl_loss=cl_loss,
            lm_logits=lm_outputs.logits,
            cl_logits=pooled_logits,
            past_key_values=lm_outputs.past_key_values,
            hidden_states=lm_outputs.hidden_states,
            attentions=lm_outputs.attentions,
            # logits must be returned for `geenrate` method
            logits=lm_outputs.logits,
        )
