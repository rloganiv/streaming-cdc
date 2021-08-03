import transformers
import meercat.utils as utils


def test_encode_mention():
    data = {
        'left_context': 'I am a',
        'mention': 'bananaananan',
        'right_context': '!',
    }
    tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-cased', fast=True)
    breakpoint()
    out = utils._encode_mention(data, tokenizer)
    print(out)

def test_streaming_shuff():
    l = [1,2,3,4,5,6]
    for i in utils.streaming_shuffle(l, chunk_size=2):
        print(i)


