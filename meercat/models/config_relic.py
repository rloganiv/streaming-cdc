from transformers import BertConfig


class RelicConfig(BertConfig):
    def __init__(
        self,
        entity_vocab_size=10000,
        entity_embedding_dim=300,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.entity_vocab_size = entity_vocab_size
        self.entity_embedding_dim = entity_embedding_dim

