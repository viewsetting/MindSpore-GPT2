from GPT2_model import GPT2Model, GPT2Config
from mindspore import nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal
from scipy.special import softmax
import numpy as np


class CrossEntropyCalculation(nn.Cell):
    """
    Cross Entropy loss
    modified from  /mindspore/model_zoo/official/nlp/bert/src/utils.py

    Args:
        num_labels: number of labels (int), depth of operations.OneHot()
        is_training: True for training procedure, False for evaluation.

    Return:
        loss: Average Cross Entrophy of a squeezed logits tensor when training, else logits it self.
    """

    def __init__(self, num_labels, is_training=True):
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
        self.num_labels = num_labels

    def construct(self, logits, label_ids):
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)
            one_hot_labels = self.onehot(
                label_ids, self.num_labels, self.on_value, self.off_value)
            per_example_loss = self.neg(self.reduce_sum(
                one_hot_labels * logits, self.last_idx))
            loss = self.reduce_mean(per_example_loss, self.last_idx)
            return_value = self.cast(loss, mstype.float32)
        else:
            return_value = logits * 1.0
        return return_value


class GPT2ForPredictNext(nn.Cell):
    """
        GPT2ForPredictNext
        generate the next token, precisely, for now.
    """

    def __init__(self, config, is_training=True):
        super(GPT2ForPredictNext, self).__init__()
        self.transformer = GPT2Model(config, is_training)
        self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False,
                                weight_init=Normal(sigma=config.initializer_range))

        # dequote to use mindspore implement
        # self.loss_function = nn.SoftmaxCrossEntropyWithLogits(sparse = True)

        # modified loss_function from modelzoo/official/nlp/bert/src/utils.py
        self.loss_function = CrossEntropyCalculation(
            config.vocab_size, is_training=True)
        self.reshape = P.Reshape()
        self.softmax = nn.Softmax(axis=-1)
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot()


    def top_k_logits(self,logits,k):
        assert logits.dim() == 2, "logits should be a mindspore.Tensor with shape [config.batch_size, config.vocab_size]."

        topk = P.TopK(sorted=True)
        logits_sorted,indices = topk(logits,k)

        batch_size = logits.shape[0]
        vocab_size = logits.shape[-1]
        assert batch_size == self.batch_size, "Logits's shape[0] should be tantamount to config.batch_size"
        #assert vocab_length == self.vocab_size, "Logits's shape[1] should be tantamount to config.vocab_size"

        mask = Tensor(np.zeros((batch_size,vocab_size)),dtype=mstype.float32)
        on_value = Tensor(1.0, mstype.float32)
        off_value = Tensor(0.0, mstype.float32)

        for batch_idx in range(batch_size):
            top_k_indices = indices[batch_idx][:k]
            tmp_onehot = self.onehot(top_k_indices,vocab_size,on_value,off_value)
            for top_k_indice in range(k):
                real_index = int(indices[batch_idx][top_k_indice].asnumpy())
                if top_k_indice != 0:
                    tmp_onehot[top_k_indice] *= 1.0
                    tmp_onehot[0] += tmp_onehot[top_k_indice]*logits[batch_idx][real_index]
                else:
                    tmp_onehot[0] *= logits[batch_idx][real_index]
            mask[batch_idx] += tmp_onehot[0]


        return mask
        


    """
    This is the deprecated top_k_logits() implementation, using numpy functions to convert mindspore.Tensor
    """
    def _top_k_logits(self,logits, k):

        assert logits.dim() == 2, "logits should be a mindspore.Tensor with shape [config.batch_size, config.vocab_size]."

        def _top_k():

            topk = P.TopK(sorted=False)
            _, indexes = topk(logits, k)

            batch_size = logits.shape[0]
            vocab_length = logits.shape[-1]
            assert batch_size == self.batch_size
            #assert vocab_length == self.vocab_size

            top_k_logits_np = np.zeros([batch_size, vocab_length], dtype=float)

            for batch_idx in range(batch_size):
                topk_index = 0
                for vocab_idx in range(vocab_length):
                    if topk_index >= k:
                        top_k_logits_np[batch_idx][vocab_idx] = 0.0
                        continue

                    if vocab_idx != indexes[batch_idx][topk_index]:
                        top_k_logits_np[batch_idx][vocab_idx] = 0.0
                    else:
                        top_k_prob = float(logits[batch_idx][vocab_idx].asnumpy())
                        top_k_logits_np[batch_idx][vocab_idx] = top_k_prob
                        topk_index += 1

            top_k_logits = Tensor(top_k_logits_np, dtype=mstype.float32)
            return top_k_logits

        if k == 0:
            return logits
        else:
            return _top_k()

    def top_p_logits(self,logits, p):
        assert logits.dim() == 2, "logits should be a mindspore.Tensor with shape [config.batch_size, config.vocab_size]."
        
        batch_size = logits.shape[0]
        vocab_length = logits.shape[-1]

        assert batch_size == self.batch_size, "Logits's shape[0] should be tantamount to config.batch_size"
        #assert vocab_length == self.vocab_size, "Logits's shape[1] should be tantamount to config.vocab_size"

        top_k = P.TopK(sorted=True)
        values,indexes = top_k(logits,vocab_length)
        
        cumsum = P.CumSum()
        sum_values = cumsum(values,1)

        top_p_logits = self.get_top_p_onehot_mask(logits,sum_values,indexes,p)

        return top_p_logits
        
    def get_top_p_logits(self,logits,cumsum_logits,indices,p):
        raise NotImplementedError

    def get_top_p_onehot_mask(self,logits,cumsum_logits,indices,p):
        batch_size = cumsum_logits.shape[0]
        vocab_size = cumsum_logits.shape[1]

        mask = Tensor(np.zeros((batch_size,vocab_size)),dtype=mstype.float32)
        on_value = Tensor(1.0, mstype.float32)
        off_value = Tensor(0.0, mstype.float32)
        
        for batch_idx in range(batch_size):
            cumsum = cumsum_logits[batch_idx].asnumpy()
            
            #to prevent loss of float calculation
            top_p_pos = int(np.searchsorted(cumsum,p-1e6))
            
            top_p_index = indices[batch_idx][0:top_p_pos+1]
            tmp_onehot = self.onehot(top_p_index,vocab_size,on_value,off_value)
            for top_num in range(top_p_pos+1):
                real_index = int(indices[batch_idx][top_num].asnumpy())
                if top_num != 0:
                    tmp_onehot[top_num] *= 1.0
                    tmp_onehot[0] += tmp_onehot[top_num]*logits[batch_idx][real_index]
                else:
                    tmp_onehot[0] *= logits[batch_idx][real_index]
            mask[batch_idx] += tmp_onehot[0]
        return mask
            

    def get_output_embeddings(self):
        return self.lm_head

    """
        return:
        output: ( [batch_size,seq_length,vocab_size], loss),transformer outputs except for the first token.
        loss is the average cross entrophy loss over a batch, loss is not generated if labels is given.
        labels here could be the next-token sequence.
    """

    def construct(
        self,
        input_ids=None,
        input_mask=None,
        # position_ids = None
        # input_embeddings = None
        labels=None
    ):

        transformer_outputs = self.transformer(
            input_ids,
            input_mask
        )

        hidden_state = transformer_outputs[0]
        batch_size = hidden_state.shape[0]
        sequence_length = hidden_state.shape[1]

        hidden_state = self.reshape(hidden_state, (-1, hidden_state.shape[-1]))
        lm_logits = self.lm_head(hidden_state)
        lm_logits = self.softmax(lm_logits)
        lm_logits = self.reshape(lm_logits, (batch_size, sequence_length, -1))

        loss = None
        if labels is not None:

            pre_lm_logits = lm_logits[:batch_size, :sequence_length-1, ...]

            shift_squeezed_logits = self.reshape(
                pre_lm_logits, (-1, pre_lm_logits.shape[-1]))
            shift_squeezed_labels = self.reshape(labels[..., 1:], (-1,))

            loss = self.loss_function(
                shift_squeezed_logits, shift_squeezed_labels)

        output = (lm_logits,) + transformer_outputs[1:]
        return (output, loss) if loss is not None else (output,)


def top_k_sample(logits, top_k=2):
    """
    generate the next tokens from sampling of top-k tokens of gpt-2.
    After top-k of last logits that gpt-2 model returns being selected, this top-k
    array will be softmaxed, then generate the next token by the softmax distribution 
    (multivariate generalisation of the binomial distribution).

    Reference: Fan, A., Lewis, M., and Dauphin, Y. Hierarchical neural story generation. 
    arXiv preprint arXiv:1805.04833, 2018.

    Args:
        logits: last tensor of a sentence from a batch, [batch_size,vocab_size]
        top_k: top-K tokens selected for random text generation
    return:
        token_ids: selected token index tensor, [batch_size]
    """
    assert logits.dim() == 2
    top_k = min(top_k, logits.shape[-1])
    topk_op = P.TopK()
    topk_prob, topk_indices = topk_op(logits, 2)
    softmax = P.Softmax(axis=-1)
    topk_prob = softmax(topk_prob)

    topk_prob_np = topk_prob.asnumpy()
    topk_indices_np = topk_indices.asnumpy()

    batch_size = logits.shape[0]
    final_tokens = []
    for batch in range(batch_size):
        final_token = np.argmax(np.random.multinomial(
            1, topk_prob_np[batch], size=1)[0])
        final_tokens.append(topk_indices_np[batch][final_token])

    final_tokens = np.array(final_tokens, dtype=np.int32)
    final_tokens_tensor = Tensor(final_tokens, dtype=mstype.int32)

    return final_tokens_tensor


