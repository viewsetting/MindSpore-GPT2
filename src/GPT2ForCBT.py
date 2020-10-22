import mindspore.nn as nn
from mindspore.ops import operations as P
from .GPT2_model import GPT2Model
from .weight_init import weight_variable

class GPT2CBTModel(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False, num_labels=10):
        super(GPT2CBT, self).__init__()
        if not is_training:
            config.hidden_dropout = 0.0

        self.gpt2 = GPT2Model(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()

        self.dtype = config.dtype
        self.lm_head = nn.Dense(config.d_model,
                               config.vocab_size,
                               weight_init=weight_variable([config.d_model, config.vocab_size]),
                               has_bias=False).to_float(config.compute_type)
        self.multiple_choice_head = SequenceSummary(config, num_labels)
        self.dropout = nn.Dropout(1 - config.hidden_dropout)

    def construct(self, input_ids, input_mask, mc_token_ids):
        output, _ = self.gpt2(input_ids, input_mask)
        output = self.cast(output, self.dtype)
        output = self.dropout(output)
        lm_output = self.lm_head(output)
        lm_output = self.cast(lm_output, self.dtype)
        mc_output = self.multiple_choice_head(output, mc_token_ids)
        mc_output = self.cast(mc_output, self.dtype)

        return lm_output, mc_output


class SequenceSummary(nn.Cell):
    def __init__(self, config, num_labels):
        super(SequenceSummary, self).__init__()
        self.summary = nn.Dense(config.d_model,
                                num_labels,
                                weight_init=weight_variable([config.d_model, num_labels]),
                                has_bias=True).to_float(config.compute_type)
        self.gelu = nn.GELU()
        self.first_dropout = nn.Dropout(1 - config.hidden_dropout)
        self.last_dropout = nn.Dropout(1 - config.hidden_dropout)

        self.expand_dims = P.ExpandDims()
        self.shape = P.Shape()
        self.size = P.Size()
        self.slice = P.GatherV2()
        self.squeeze = P.Squeeze(-2)

    def construct(self, hidden_states, cls_index):
        cls_index = self.expand_dims(cls_index, -1)
        cls_index = self.expand_dims(cls_index, -1)

        # P.BroadcastTo算子可能有点问题
        cls_index_shape = self.shape(cls_index)
        broad_shape = (-1,) * self.size(cls_index_shape) + (self.shape(hidden_states)[-1],)
        cls_index = P.BroadcastTo(broad_shape)(cls_index)

        output = self.slice(hidden_states, cls_index, -2)
        output = self.squeeze(output)

        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.gelu(output)
        output = self.last_dropout(output)

        return output




