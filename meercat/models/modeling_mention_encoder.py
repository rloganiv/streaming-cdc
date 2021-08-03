from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from transformers.file_utils import ModelOutput


class MentionEncoderModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = transformers.BertModel(config)
        self.context_projection = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.entity_embedding_dim,
        )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        counts=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)

        # (batch_size, embedding_dim)
        context_embedding = self.context_projection(pooled_output)
        context_embedding = F.normalize(context_embedding, dim=-1)
        context_embedding = context_embedding

        return context_embedding

