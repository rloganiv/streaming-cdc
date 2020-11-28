import transformers
import torch

from meercat.models import RelicModel, RelicConfig


tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased')
config = RelicConfig(vocab_size=28996)
model = RelicModel.from_pretrained('bert-base-cased', config=config)

model_inputs = tokenizer(
    ['I am a banana', 'My spoon is too big'],
    return_tensors='pt',
    truncate=True,
    padding=True
)
labels = torch.tensor([0, 1])
model_ouput = model(**model_inputs, labels=labels)

