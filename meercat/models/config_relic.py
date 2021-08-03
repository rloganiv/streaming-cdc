from transformers import BertConfig


class RelicConfig(BertConfig):
    def __init__(
        self,
        entity_vocab_size=10000,
        entity_embedding_dim=300,
        use_batch_negatives=True,
        random_negatives=4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.entity_vocab_size = entity_vocab_size
        self.entity_embedding_dim = entity_embedding_dim
        self.use_batch_negatives = use_batch_negatives
        self.random_negatives = random_negatives

