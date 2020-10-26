from .GPT2_model import GPT2Model, GPT2Config
from mindspore import nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import operations as P
from mindspore.common.initializer import Normal,TruncatedNormal
#from .utils.CrossEntropy import CrossEntropyCalculation
from scipy.special import softmax
import numpy as np


class GPT2SummarizationModel(nn.Cell):
    """
        GPT2SummarizationModel
        generate the next token, precisely, for now.
    """

    def __init__(self, config, is_training=True, use_one_hot_embeddings=False):
        super(GPT2SummarizationModel, self).__init__()
        self.gpt2 = GPT2Model(
            config, is_training, use_one_hot_embeddings)
        self.lm_head = nn.Dense(config.d_model, config.vocab_size, has_bias=False,
                                weight_init=TruncatedNormal(sigma=config.initializer_range))
        self.reshape = P.Reshape()
        self.softmax = P.LogSoftmax(axis=-1)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.batch_size = config.batch_size
        self.vocab_size = config.vocab_size
        self.seq_length = config.seq_length
        self.onehot = P.OneHot()
        self.print= P.Print()

    
    def get_output_embeddings(self):
        return self.lm_head

    """
        return:
        output: ( [batch_size,seq_length,vocab_size], loss),transformer outputs except for the first token.
        loss is the average cross entrophy loss over a batch, loss is not generated if labels is given.
        labels here could be the next-token sequence.
    """

    def get_lm_head(self,input_ids):
        return self.lm_head(input_ids)


    def construct(
        self,
        input_ids=None,
        input_mask=None,
        # position_ids = None
        # input_embeddings = None
        # labels=None
    ):

        transformer_outputs,_= self.gpt2(
            input_ids,
            input_mask
        )

        hidden_state = transformer_outputs
        batch_size = hidden_state.shape[0]
        sequence_length = hidden_state.shape[1]

        hidden_state = self.reshape(hidden_state, (-1, hidden_state.shape[-1]))
        lm_logits = self.lm_head(hidden_state)
       
        
        lm_logits = self.reshape(lm_logits, (batch_size, sequence_length, -1))

        loss = None

        return lm_logits
        # return (output, loss) if loss is not None else (output,)






    

    
