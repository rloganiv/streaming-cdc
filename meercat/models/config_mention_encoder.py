from transformers import BertConfig


class MentionEncoderConfig(BertConfig):
    def __init__(
        self,
        entity_embedding_dim=300,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.entity_embedding_dim = entity_embedding_dim

