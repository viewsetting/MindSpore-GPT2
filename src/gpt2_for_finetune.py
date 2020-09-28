import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
import mindspore.common.dtype as mstype
from mindspore import context
from GPT2ForLanguageModel import GPT2LanguageModel
from GPT2ForReadComprehension import GPT2CoQAModel
from GPT2ForSummarization import GPT2ForPredictNext
from src.utils.CrossEntropy import CrossEntropyCalculation

class GPT2LM(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2LM, self).__init__()
        self.gpt2 = GPT2LanguageModel(config, is_training, use_one_hot_embeddings)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
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


class GPT2CoQA(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2CoQA, self).__init__()
        self.gpt2 = GPT2CoQAModel(config, is_training, use_one_hot_embeddings)
        self.loss = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=True)
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.last_idx = (-1,)

    def construct(self, input_ids, input_mask, label_ids):
        output = self.gpt2(input_ids, input_mask)
        output_shape = P.Shape()(output)
        output = P.Reshape()(output, (-1, output_shape[-1]))

        if self.is_training:
            label_ids = P.Reshape()(label_ids, self.last_idx)
            loss = self.loss(output, label_ids)
        else:
            logits = self.log_softmax(output)
            loss = logits * 1.0
        return loss

class GPT2Summarization(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2Summarization, self).__init__()
        self.gpt2 = GPT2ForPredictNext(config, is_training, use_one_hot_embeddings)
        self.is_training = is_training
        self.last_idx = (-1,)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.loss_function = CrossEntropyCalculation(num_labels = config.vocab_size,is_training=self.is_training)
    def construct(self, input_ids,input_mask):
        lm_logits = self.gpt2(input_ids,input_mask)

        pre_lm_logits = lm_logits[:self.batch_size, :self.seq_length-1, ...]

        shift_squeezed_logits = self.reshape(
            pre_lm_logits, (-1, pre_lm_logits.shape[-1]))
        shift_squeezed_ids = self.reshape(input_ids[..., 1:], (-1,))

        loss = self.loss_function(shift_squeezed_logits, shift_squeezed_ids)

        return loss


