from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
import mindspore.nn as nn


class CrossEntropyCalculation(nn.Cell):
    """
    Cross Entropy loss
    """
    def __init__(self, num_labels,is_training=True):
        super(CrossEntropyCalculation, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training
        self.print = P.Print()
        self.num_labels = num_labels

    def construct(self, logits, label_ids): # logits [batch * seq_length, vocab_size]   label_ids [batch, seq_length]
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx) # label_ids [batch * seq_length]
            one_hot_labels = self.onehot(label_ids, num_labels, self.on_value, self.off_value) # [batch * seq_length, vocab_size]
            per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx)) # [batch * seq_length]
            loss = self.reduce_mean(per_example_loss, self.last_idx) # a number
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0 # [batch * seq_length, vocab_size]
        return return_value