import argparse
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import transformers
from transformers.file_utils import ModelOutput



class LinkerOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    scores: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class RelicModel(transformers.BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = transformers.BertModel(config)
        self.context_projection = torch.nn.Linear(
            in_features=config.hidden_size,
            out_features=config.entity_embedding_dim,
        )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.scaling_constant = torch.nn.Parameter(torch.tensor(1.0))
        self.entity_embeddings = torch.nn.Embedding(
            num_embeddings=config.entity_vocab_size,
            embedding_dim=config.entity_embedding_dim,
        )

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
        return_dict=None
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

        if labels is not None:
            # TODO(@rloganiv): If batch sizes are too small, we probably want
            # to sample outside of the batch as well.
            # (num_entities[subsampled], embedding_dim)
            entity_embeddings = self.entity_embeddings(labels)
        else:
            # (num_entities, embedding_dim)
            entity_embeddings = self.entity_embeddings.weight
        entity_embeddings = F.normalize(entity_embeddings, dim=-1)

        # (batch_size, num_entities)
        scores = self.scaling_constant * torch.mm(
            context_embedding,
            entity_embeddings.transpose(0, 1),
        )

        loss = None
        if labels is not None:
            log_probs = F.log_softmax(scores, dim=-1)
            loss = -torch.diag(log_probs).mean()

        if not return_dict:
            output = (scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return LinkerOutput(
            loss=loss,
            scores=scores,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

