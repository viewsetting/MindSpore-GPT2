import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore import context
from .GPT2ForLanguageModel import GPT2LanguageModel

class GPT2LM(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2LM, self).__init__()
        self.gpt2 = GPT2LanguageModel(config, is_training, use_one_hot_embeddings)
        self.loss = nn.SoftmxCrossEntropyWithLogits(is_grad=False, sparse=True)
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.last_idx = (-1,)

    def construct(self, input_ids, input_mask, label_ids):
        output = self.gpt2(input_ids, input_mask)  # no softmax
        output_shape = self.shape(output)
        output = self.reshape(output, (-1, output_shape[-1]))

        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)
            loss = self.loss(output, label_ids)
        else:
            logits = self.log_softmax(output)
            loss = logits * 1.0
        return loss