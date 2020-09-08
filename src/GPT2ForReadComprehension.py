import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from GPT2_model import GPT2Model
from mindspore.ops import operations as P


class GPT2CoQAModel(nn.Cell):
    '''
        This class is responsible for SQuAD
        The returned output represents the final logits as the results of log_softmax is propotional to that of softmax.
    '''

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2CoQAModel, self).__init__()
        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.d_model, config.vocab_size, weight_init=self.weight_init, has_bias=True).to_float(
            config.compute_type)
        self.vocab_size = config.vocab_size
        self.dtype = config.dtype

    def construct(self, input_ids, input_mask):
        decoder_output, _ = self.gpt2(input_ids, input_mask)
        batch_size, seq_length, hidden_size = P.Shape()(decoder_output)
        sequence = P.Reshape()(decoder_output, (-1, hidden_size))
        logits = self.dense1(sequence)
        logits = P.Cast()(logits, self.dtype)
        logits = P.Reshape()(logits, (batch_size, seq_length, self.vocab_size))
        return logits
