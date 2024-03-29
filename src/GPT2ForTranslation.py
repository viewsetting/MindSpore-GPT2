import mindspore.nn as nn
from mindspore.ops import operations as P
from .GPT2_model import GPT2Model
from mindspore.common.initializer import TruncatedNormal


class GPT2TranslationModel(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2TranslationModel, self).__init__()
        if not is_training:
            config.hidden_dropout = 0.0

        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.shape = P.Shape()
        # self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.dense1 = nn.Dense(config.d_model,
                               config.vocab_size,
                               weight_init=TruncatedNormal(config.initializer_range),
                               has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - config.hidden_dropout)

    def construct(self, input_ids, input_mask):
        output, _ = self.gpt2(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        #output = self.dropout(output)
        batch_size, seq_length, d_model = self.shape(output)
        output_reshape = P.Reshape()(output, (-1, d_model)) # [batch_size * seq_len, d_model]
        logits = self.dense1(output_reshape)
        logits = self.cast(logits, self.dtype)
        #logits = self.log_softmax(logits)
        translation_logits = P.Reshape()(logits, (batch_size, seq_length, self.vocab_size)) # [batch_size, seq_len, vocab]

        return translation_logits
