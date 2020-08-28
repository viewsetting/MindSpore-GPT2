import mindspore.nn as nn
from mindspore.ops import operation as P
from .GPT2_model import GPT2Model
from .weight_init import weight_variable

class GPT2LanguageModel(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2LanguageModel, self).__init__()
        if not is_training:
            config.hidden_dropout = 0.0

        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.dense1 = nn.Dense(config.hidden_size,
                               config.vocab_size,
                               weight_init=weight_variable([config.hidden_size, config.vocab_size]),
                               has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - config.hidden_dropout)

    def construct(self, input_ids, input_mask):
        output, _ = self.gpt2(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        output = self.dropout(output)
        logits = self.dense1(output)
        logits = self.cast(logits, self.dtype)
        logits = self.log_softmax(logits)

        return logits