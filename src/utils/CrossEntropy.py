import numpy as np
from mindspore.common.tensor import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.ops import functional as F
import mindspore.nn as nn

# softmax
def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

# cross entropy with numpy
def cross_entropy_np(logits, labels):
    x_softmax = [softmax(logits[i]) for i in range(len(logits))]
    x_log = [np.log(x_softmax[i][labels[i]]) for i in range(len(labels))]
    loss = - np.sum(x_log) / len(labels)
    return loss

class CrossEntropyCalculationWithMask(nn.Cell):
    """
    Cross Entropy loss
    """
    def __init__(self, is_training=None, num_labels=None, config=None):
        super(CrossEntropyCalculationWithMask, self).__init__()
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
        self.num_labels = num_labels
        if config is not None:
            # for PPL calculation in evaluation
            self.input_mask_length = Tensor(config.batch_size * (config.seq_length - 1), mstype.float32)

    def construct(self, logits, label_ids, input_mask=None): # logits [batch * (seq_length-1), vocab_size]   label_ids [batch, seq_length-1]
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx) # label_ids [batch * (seq_length-1)]
            one_hot_labels = self.onehot(label_ids, self.num_labels, self.on_value, self.off_value) # [batch * (seq_length-1), vocab_size]
            per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx)) # [batch * (seq_length-1)]
            
            # for PPL calculation in evaluation 
            if input_mask is not None:
                input_mask = self.cast(self.reshape(input_mask, self.last_idx), mstype.float32) # [batch * (seq_length-1)]
                
                valid_loss_sum = self.reduce_sum(input_mask * per_example_loss, ())
                valid_element_sum = self.reduce_sum(input_mask, ()) + self.cast(F.tuple_to_array((1e-5,)), mstype.float32)
                loss = valid_loss_sum / valid_element_sum
                # useful_input_num = self.cast(self.reduce_sum(self.cast(input_mask, mstype.float32), self.last_idx), mstype.float32)
                # scale_useful_input = self.cast(self.input_mask_length / useful_input_num, mstype.float32)
                # per_example_loss_with_mask = input_mask * per_example_loss
                # loss = self.reduce_mean(per_example_loss_with_mask, self.last_idx) * scale_useful_input
            else:
                loss = self.reduce_mean(per_example_loss, self.last_idx) # a number
            
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0 # [batch * (seq_length-1), vocab_size]
        
        return return_value

if __name__ == "__main__":
    x = np.array([[0.093, 0.1939, -1.0649, 0.4476, -2.0769],
            [-1.8024, 0.3696, 0.7796, -1.0346, 0.473],
            [0.5593, -2.5067, -2.1275, 0.5548, -1.6639]])

    y = np.array([1, 2, 3])
    print('numpy result: ', cross_entropy_np(x, y)) # x:logits, y:labels


    cross_entropy = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss = cross_entropy(Tensor(x,mstype.float32), Tensor(y,mstype.int32))
    print("mindspore Cross Entropy :",loss)


    # numpy result:  1.0155949508195155
    # mindspore Cross Entropy : 1.015595