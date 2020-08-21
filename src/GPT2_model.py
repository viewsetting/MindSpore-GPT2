import math
import copy
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.nn as nn
import mindspore.ops.functional as F
from mindspore.ops import operations as P
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
from weight_init import normal_weight, weight_variable, zero_weight

class GPT2Config:
    """
       Configuration for `GPT2Model`.

       Args:
           batch_size (int): Batch size of input dataset. Default: 512.
           seq_length (int): Length of input sequence. Default: 1024.
           vocab_size (int): The shape of each embedding vector. Default: 50257.
           d_model (int): Size of the bert encoder layers. Default: 768.
           num_hidden_layers (int): Number of hidden layers in the GPT2Transformer decoder block. Default: 12.
           num_attention_heads (int): Number of attention heads in the GPT2Transformer decoder block. Default: 12.
           intermediate_size (int): Size of intermediate layer in the GPT2Transformer decoder block. Default: 3072.
           hidden_act (str): Activation function used in the GPT2Transformer decoder block. Default: "gelu".
           hidden_dropout (float): The dropout probability for GPT2Output. Default: 0.1.
           attention_dropout (float): The dropout probability for MaskedMultiHeadAttention. Default: 0.1.
           max_position_embeddings (int): Maximum length of sequences used in this model. Default: 1024.
           initializer_range (float): Initialization value of TruncatedNormal. Default: 0.02.
           input_mask_from_dataset (bool): Specifies whether to use the input mask that loaded from dataset. Default: True.
           dtype (:class:`mindspore.dtype`): Data type of the input. Default: mstype.float32.
           compute_type (:class:`mindspore.dtype`): Compute type in GPT2Transformer. Default: mstype.float32.
       """
    def __init__(self,
                 batch_size=512,
                 seq_length=1024,
                 vocab_size=50257,
                 d_model=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout=0.1,
                 attention_dropout=0.1,
                 max_position_embeddings=1024,
                 initializer_range=0.02,
                 input_mask_from_dataset=True,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.input_mask_from_dataset = input_mask_from_dataset
        self.dtype = dtype
        self.compute_type = compute_type

class EmbeddingLookup(nn.Cell):
    """
    A embeddings lookup table with a fixed dictionary and size.

    Args:
        vocab_size (int): Size of the dictionary of embeddings.
        embedding_size (int): The size of each embedding vector.
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 use_one_hot_embeddings=False):
        super(EmbeddingLookup, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.use_one_hot_embeddings = use_one_hot_embeddings
        self.embedding_table = Parameter(normal_weight([vocab_size, embedding_dim], embedding_dim), name='embedding_table')

        self.expand = P.ExpandDims()
        self.shape_flat = (-1, )
        self.gather = P.GatherV2() # axis=1 从列取  axis=0从行取 index_select
        self.one_hot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.array_mul = P.MatMul()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_ids):
        input_shape = self.shape(input_ids) # [batch_size, seq_length]
        flat_ids = self.reshape(input_ids, self.shape_flat) # [batch_size * seq_length]

        if self.use_one_hot_embeddings:
            one_hot_ids = self.one_hot(flat_ids, self.vocab_size, self.on_value, self.off_value)
            output_for_reshape = self.array_mul(one_hot_ids, self.embedding_table)
        else:
            output_for_reshape = self.gather(self.embedding_table, flat_ids, 0) # [batch_size * seq_length * embedding_dim]

        out_shape = input_shape + (self.embedding_dim, )
        output = self.reshape(output_for_reshape, out_shape) # [batch_size, seq_length, embedidng_dim]
        return output, self.embedding_table

class EmbeddingPostprocessor(nn.Cell):
    """
    Postprocessors apply positional embeddings to word embeddings.

    Args:
        embedding_dim (int): The size of each embedding vector.
        embedding_shape (tuple): [batch_size, seq_length, embedding_size], the shape of each embedding vector.
        max_position_embeddings (int): Maximum length of sequences used in this model. Default: 1024.
        dropout_prob (float): The dropout probability. Default: 0.1.
     """
    def __init__(self,
                 embedding_dim,
                 embedding_shape,
                 max_position_embeddings=1024,
                 dropout_prob=0.1):
        super(EmbeddingPostprocessor, self).__init__()

        self.position_embedding_table = Parameter(normal_weight([max_position_embeddings, embedding_dim], embedding_dim), name='position_embeddings')
        self.shape = tuple(embedding_shape) # [batch_size, seq_len, d_model]
        self.expand_dims = P.ExpandDims()
        self.add = P.TensorAdd()
        self.slice = P.StridedSlice()
        self.dropout = nn.Dropout(1 - dropout_prob, dtype=mstype.float32)
        self.use_dropout = dropout_prob > 0

    def construct(self, word_embeddings):
        output = word_embeddings
        _, seq_len, dim = self.shape
        position_embeddings = self.slice(self.position_embedding_table, (0, 0), (seq_len, dim), (1, 1))
        position_embeddings = self.expand_dims(position_embeddings, 0)
        output = self.add(word_embeddings, position_embeddings)

        if self.use_dropout:
            output = self.dropout(output)
        return output

class CastWrapper(nn.Cell):
    """
    Cast wrapper
    """
    def __init__(self,
                 dst_type=mstype.float32):
        super(CastWrapper, self).__init__()
        self.cast = P.Cast()
        self.dst_type = dst_type

    def construct(self, x):
        return self.cast(x, self.dst_type)

class LayerNorm(nn.Cell):
    """
    Do layer norm

    Args:
        in_channels (int): In channels number of layer norm
    """
    def __init__(self,
                 in_channels=None):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm((in_channels, ))
        self.cast = P.Cast()
        self.get_dtype = P.DType()

    def construct(self, input_tensor):
        output = self.cast(input_tensor, mstype.float32)
        output = self.layer_norm(output)
        output = self.cast(output, self.get_dtype(input_tensor))
        return output

class ResidualConnection(nn.Cell):
    """
    Add residual to output.

    Args:
        dropout_prob (float): Dropout rate.

    Returns:
        Tensor, with the same shape of hidden_tensor
    """
    def __init__(self,
                 dropout_prob=0.1):
        super(ResidualConnection, self).__init__()
        self.add = P.TensorAdd()
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.use_dropout = dropout_prob > 0

    def construct(self, hidden_tensor, input_tensor):
        # hidden_tensor is the output of sublayer
        output = hidden_tensor
        if self.use_dropout:
            output = self.dropout(output)
        output = self.add(output, input_tensor)
        return output


class Conv1D(nn.Cell):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nx (int): The number of input features.
        nf (int): The number of output features.
    """
    def __init__(self,
                 nx,
                 nf):
        super(Conv1D, self).__init__()
        self.nx = nx
        self.nf = nf
        self.weight = Parameter(normal_weight([nx, nf], nf), name='projection_weight')
        self.bias = Parameter(zero_weight(nf), name='projection_bias')

        self.matmul = P.MatMul()
        self.add = P.TensorAdd()

    def construct(self, input_tensor):  # [batch_size * seq_length, nx]
        output_tensor = self.matmul(input_tensor, self.weight)  # [batch_size * seq_length, self.nf]
        output_tensor = self.add(output_tensor, self.bias) # [batch_size * seq_length, self.nf]

        return output_tensor

class MaskedSelfAttention(nn.Cell):
    """
    Apply masked multi-head attention.

    Args:
        batch_size (int): Batch size of input datasets. Default: 512.
        d_model (int): Size of last dim of input tensor. Default: 768.
        seq_length (int): Length of input tensor sequence. Default: 1024.
        num_attention_heads (int): Number of attention heads. Default: 12.
        dim_per_head (int): Size of each attention head. Default: 64.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: True.
        attention_dropout (float): The dropout probability for MultiheadAttention. Default: 0.0.
        compute_type (:class:`mindspore.dtype`): Compute type in MultiheadAttention. Default: mstype.float32.

    Returns:
        Tensor, with the shape [batch_size, seq_length, d_model]

    """
    def __init__(self,
                 batch_size=512,
                 d_model=768,
                 seq_length=1024,
                 num_attention_heads=12,
                 dim_per_head=64,
                 has_attention_mask=True,
                 do_return_2d_tensor=True,
                 attention_dropout=0.0,
                 compute_type=mstype.float32):
        super(MaskedSelfAttention, self).__init__()

        self.batch_size = batch_size
        self.d_model = d_model
        self.seq_length = seq_length
        self.num_heads = num_attention_heads
        self.dim_per_head = dim_per_head
        self.has_attention_mask = has_attention_mask
        assert has_attention_mask

        self.scale = Tensor([1.0 / math.sqrt(float(self.dim_per_head))], dtype=compute_type) # attention scale
        self.mask_data = Tensor([-10000.0, ], dtype=compute_type)
        self.split_head_shape = (self.batch_size, self.seq_length, self.num_heads, self.dim_per_head)

        self.c_attn = Conv1D(d_model, d_model*3)
        self.c_proj = Conv1D(d_model, d_model)

        self.split_for_qkv = P.Split(1, 3) # P.Split(axis, output_num)
        # self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.trans_shape = (0, 2, 1, 3)
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.matmul = P.BatchMatMul()
        self.multiply = P.Mul()

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.TensorAdd()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

        if do_return_2d_tensor:
            self.shape_return = (batch_size * seq_length, d_model)
        else:
            self.shape_return = (batch_size, seq_length, d_model)

        self.softmax = nn.Softmax()
        self.softmax_cast = P.Cast()
        self.dropout = nn.Dropout(1 - attention_dropout)
        self.use_attention_dropout = attention_dropout > 0

    def construct(self, input_tensor, attention_mask): # input_tensor [batch_size * seq_length, d_mdoel]
        input_tensor = self.c_attn(input_tensor) # [batch_size * seq_length, d_model*3]---> eg.[1 * 3, 2304]
        input_tensor = self.split_for_qkv(input_tensor)
        query = input_tensor[0] # [batch_size * seq_length, d_model] ---> eg. [1 * 3, 768]
        key = input_tensor[1]
        value = input_tensor[2]

        # split head
        query = self.reshape(query, self.split_head_shape)
        query = self.transpose(query, self.trans_shape) # [batch_size, num_heads, seq_len, dim_per_head] ---> eg. [1, 12, 3, 64]

        key = self.reshape(key, self.split_head_shape)
        key = self.transpose(key, self.trans_shape) # [batch_size, num_heads, seq_len, dim_per_head] ---> eg. [1, 12, 3, 64]

        value = self.reshape(value, self.split_head_shape)
        value = self.transpose(value, self.trans_shape) # [batch_size, num_heads, seq_len, dim_per_head] ---> eg. [1, 12, 3, 64]

        # attention and mask
        attention_scores = self.matmul_trans_b(query, key)  # [batch_size, num_heads, seq_len, seq_len]
        attention_scores = self.multiply(attention_scores, self.scale)

        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1) # [batch_size, 1, seq_length, seq_length]
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), self.get_dtype(attention_scores)),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))
            adder = self.multiply(multiply_out, self.mask_data)
            attention_scores = self.add(adder, attention_scores)

        attention_scores = self.softmax_cast(attention_scores, mstype.float32)
        attention_probs = self.softmax(attention_scores) # [batch_size, num_heads, seq_len, seq_len]
        attention_probs = self.softmax_cast(attention_probs, self.get_dtype(key))

        if self.use_attention_dropout:
            attention_probs = self.dropout(attention_probs)

        outputs = self.matmul(attention_probs, value) # [batch_size, num_heads, seq_len, dim_per_head]

        # merge heads
        outputs = self.transpose(outputs, self.trans_shape) # [batch_size, seq_len, num_heads, dim_per_head]
        outputs = self.reshape(outputs, self.shape_return) # default True, the outputs shape [batch_size * seq_len, d_model]

        # project
        outputs = self.c_proj(outputs)
        return outputs

class FeedForward(nn.Cell):
    """
    Apply two-layer feed forward

    Args:
        in_channels (int): Size of the input layer. Default: 768.
        out_channels (int): Size of the output layers. Default: 768.
        hidden_size (int): Size of the hidden layer. Default: 3072.
        hidden_dropout (float): The dropout probability for hidden outputs. Default: 0.1.
    """
    def __init__(self,
                 in_channels=786,
                 out_channels=768,
                 hidden_size=3072,
                 hidden_dropout=0.1):
        super(FeedForward, self).__init__()

        self.c_fc = Conv1D(in_channels, hidden_size)
        self.c_proj = Conv1D(hidden_size, out_channels)

        self.layernorm = LayerNorm(in_channels=in_channels)
        self.residual_connect = ResidualConnection(dropout_prob=hidden_dropout)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(1 - hidden_dropout)
        self.use_dropout = hidden_dropout > 0

        self.reshape = P.Reshape()

    def construct(self, input_tensor): # input_tensor shape [batch_szie * seq_len, d_model]
        # LayerNorm
        output = self.layernorm(input_tensor)
        # Feed Forward
        output = self.c_fc(output)  # [batch_szie * seq_len, d_model * 4]
        if self.use_dropout:
            output = self.dropout(output)
        output = self.c_proj(output)  # [batch_szie * seq_len, d_model]
        # Add
        output = self.residual_connect(output, input_tensor) # [batch_szie * seq_len, d_model]
        return output

class MaskedMultiHeadAttention(nn.Cell):
    def __init__(self,
                 batch_size=512,
                 seq_length=2014,
                 d_model=768,
                 num_attention_heads=12,
                 attention_dropout=0.02,
                 hidden_dropout=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float32
                 ):
        super(MaskedMultiHeadAttention, self).__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (d_model, num_attention_heads))

        self.dim_per_head = int(d_model / num_attention_heads)  # 64

        self.masked_self_attention = MaskedSelfAttention(
            batch_size=batch_size,
            d_model=d_model,
            seq_length=seq_length,
            num_attention_heads=num_attention_heads,
            dim_per_head=self.dim_per_head,
            has_attention_mask=has_attention_mask,
            do_return_2d_tensor=True,
            attention_dropout=attention_dropout,
            compute_type=compute_type
        )

        self.layer_norm = LayerNorm(in_channels=d_model)
        self.residual_connection = ResidualConnection(dropout_prob=hidden_dropout)

        self.reshape = P.Reshape()
        self.new_shape = (-1, d_model)

    def construct(self, input_tensor, attention_mask): # input tensor shape[batch_size * seq_length, d_model]
        # layernorm
        output_tensor = self.layer_norm(input_tensor)
        # masked multi-head attention
        attention_output = self.masked_self_attention(output_tensor, attention_mask) # [batch_size * seq_length, d_model]
        # residual connection
        output = self.residual_connection(attention_output, input_tensor) # [batch_size * seq_length, d_model]
        return output

class DecoderBlock(nn.Cell):
    """
    decoder block used in GPT2.

    Args:
        batch_size (int): Batch size of input dataset. Default: 512.
        seq_length (int): Length of input sequence. Default: 1024.
        d_model (int): Size of the GPT2 decoder layers. Default: 768.
        num_attention_heads (int): Number of attention heads. Default: 12.
        intermediate_size (int): Size of intermediate layer. Default: 3072.
        attention_dropout (float): The dropout probability for MaskedMultiHeadAttention. Default: 0.02.
        hidden_dropout (float): The dropout probability for hidden outputs. Default: 0.1.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: True.
        compute_type (:class:`mindspore.dtype`): Compute type in attention. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size=512,
                 seq_length=1024,
                 d_model=768,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 attention_dropout=0.02,
                 hidden_dropout=0.1,
                 has_attention_mask=True,
                 compute_type=mstype.float32
                 ):
        super(DecoderBlock, self).__init__()
        if d_model % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number "
                             "of attention heads (%d)" % (d_model, num_attention_heads))

        self.dim_per_head = int(d_model / num_attention_heads)  # 64

        self.masked_multi_head_attention = MaskedMultiHeadAttention(
            batch_size=batch_size,
            seq_length=seq_length,
            d_model=d_model,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            hidden_dropout=hidden_dropout,
            has_attention_mask=has_attention_mask,
            compute_type=compute_type
        )
        self.feedforward = FeedForward(
            in_channels=d_model,
            out_channels=d_model,
            hidden_size=intermediate_size,
            hidden_dropout=hidden_dropout
        )

        self.reshape = P.Reshape()
        self.new_shape = (-1, d_model)

    def construct(self, input_tensor, attention_mask): # input tensor shape[batch_size, seq_length, d_model]
        input_tensor = self.reshape(input_tensor, self.new_shape) # [batch_size * seq_length, d_model]

        # masked multi head attention with ln, res
        attention_output = self.masked_multi_head_attention(input_tensor, attention_mask)
        # feed forward with ln, res
        output = self.feedforward(attention_output) # [batch_size * seq_length, d_model]

        return output

class GPT2Transformer(nn.Cell):
    """
    Multi-layer GPT2 transformer.

    Args:
        batch_size (int): Batch size of input dataset. Default: 512.
        d_model (int): Size of the decoder layers. Default: 768.
        seq_length (int): Length of input sequence. Default: 1024.
        num_hidden_layers (int): Number of hidden layers in decoder cells. Default: 12.
        num_attention_heads (int): Number of attention heads in decoder cells. Default: 12.
        intermediate_size (int): Size of intermediate layer in decoder cells. Default: 3072.
        has_attention_mask (bool): Specifies whether to use attention mask. Default: True.
        attention_dropout (float): The dropout probability for MaskedMultiHeadAttention. Default: 0.1.
        hidden_dropout (float): The dropout probability for GPT2Output. Default: 0.1.
        compute_type (:class:`mindspore.dtype`): Compute type in BertTransformer. Default: mstype.float32.
    """
    def __init__(self,
                 batch_size=512,
                 d_model=768,
                 seq_length=1024,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 has_attention_mask=True,
                 attention_dropout=0.1,
                 hidden_dropout=0.1,
                 compute_type=mstype.float32):
        super(GPT2Transformer, self).__init__()

        layers = []
        for _ in range(num_hidden_layers):
            layer = DecoderBlock(batch_size=batch_size,
                                 seq_length=seq_length,
                                 d_model=d_model,
                                 num_attention_heads=num_attention_heads,
                                 intermediate_size=intermediate_size,
                                 attention_dropout=attention_dropout,
                                 hidden_dropout=hidden_dropout,
                                 has_attention_mask=has_attention_mask,
                                 compute_type=compute_type)
            layers.append(layer)

        self.layers = nn.CellList(layers)

        self.reshape = P.Reshape()
        self.new_shape = (-1, d_model)
        self.out_shape = (batch_size, seq_length, d_model)

    def construct(self, input_tensor, attention_mask):
        prev_output = self.reshape(input_tensor, self.new_shape)
        for layer_module in self.layers:
            layer_output = layer_module(prev_output, attention_mask)
            prev_output = layer_output

        output = self.reshape(prev_output, self.out_shape)
        return output

class CreateAttentionMaskFromInputMask(nn.Cell):
    """
    Create attention mask according to input mask.

    Args:
        config (Class): Configuration for GPT2Model.
    """
    def __init__(self, config):
        super(CreateAttentionMaskFromInputMask, self).__init__()
        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.input_mask = None

        assert self.input_mask_from_dataset

        self.cast = P.Cast()
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.matmul = P.BatchMatMul()
        self.multiply = P.Mul()

        # mask future positions
        ones = np.ones(shape=(config.batch_size, config.seq_length, config.seq_length))
        self.lower_triangle_mask = Tensor(np.tril(ones), dtype=mstype.float32)

    def construct(self, input_mask, mask_future=True):
        """
        Construct network.

        Args:
            input_mask (Tensor): Tensor mask vectors with shape [batch_size, seq_len].
            mask_future (bool): Whether mask future (for decoder training). Default: True.

        Returns:
            attention_mask (Tensor): shape [batch_size, seq_len, seq_len].
        """
        input_shape = self.shape(input_mask)
        shape_right = (input_shape[0], 1, input_shape[1]) # [batch_size, 1, seq_len]
        shape_left = input_shape + (1,) # [batch_size, seq_len, 1]

        input_mask = self.cast(input_mask, mstype.float32)
        mask_left = self.reshape(input_mask, shape_left)
        mask_right = self.reshape(input_mask, shape_right)

        attention_mask = self.matmul(mask_left, mask_right) # [batch_szie, seq_len, seq_len]
        if mask_future:
            attention_mask = self.multiply(attention_mask, self.lower_triangle_mask)

        return attention_mask

class GPT2Model(nn.Cell):
    """
    Decoder Representations from Transformers.

    Args:
        config (Class): Configuration for GPT2Model.
        is_training (bool): True for training mode. False for eval mode. ######### training要写在这里吗？
        use_one_hot_embeddings (bool): Specifies whether to use one hot encoding form. Default: False.
    """
    def __init__(self,
                 config,
                 is_training,
                 use_one_hot_embeddings=False):
        super(GPT2Model, self).__init__()
        config = copy.deepcopy(config)
        self.is_training = is_training
        if not is_training:
            config.hidden_dropout = 0.0
            config.attention_dropout = 0.0

        self.input_mask_from_dataset = config.input_mask_from_dataset
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.d_model = config.d_model
        self.num_hidden_layers = config.num_hidden_layers
        self.embedding_dim = config.d_model

        self.last_idx = self.num_hidden_layers - 1

        self.gpt2_embedding_lookup = EmbeddingLookup(
            vocab_size=config.vocab_size,
            embedding_dim=self.embedding_dim,
            use_one_hot_embeddings=use_one_hot_embeddings
        )
        self.gpt2_embedding_postprocess = EmbeddingPostprocessor(
            embedding_dim=self.embedding_dim,
            embedding_shape=(self.batch_size, self.seq_length, self.d_model),
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout
        )
        self.gpt2_decoder = GPT2Transformer(
            batch_size=self.batch_size,
            d_model=self.d_model,
            seq_length=self.seq_length,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            has_attention_mask=True,
            attention_dropout=config.attention_dropout,
            hidden_dropout=config.hidden_dropout,
            compute_type=config.compute_type
        )

        self.cast_compute_type = CastWrapper(dst_type=config.compute_type)
        self.layer_norm = LayerNorm(in_channels=self.d_model)
        self.dropout = nn.Dropout(1 - config.hidden_dropout)
        self._create_attention_mask_from_input_mask = CreateAttentionMaskFromInputMask(config)

        self.reshape = P.Reshape()
        self.new_shape = (-1, self.d_model)

    def construct(self, input_ids, input_mask):
        """
        Construct network.

        Args:
            input_ids (Tensor): input sentences with shape [batch_size, seq_len].
            input_mask (Tensor): input sentences padding mask with shape [batch_size, seq_len],
                where 0 indicates padding position.

        Returns:
            decoder_output (Tensor): shape[batch_size, seq_len, d_model].
            embedding_tables (Tensor): word embeddings with shape [vocab_size, d_model]
        """
        # Embedding
        word_embeddings, embedding_tables = self.gpt2_embedding_lookup(input_ids)
        embedding_output = self.gpt2_embedding_postprocess(word_embeddings)
        embedding_output = self.dropout(embedding_output)

        # Attention mask with shape [batch_size, seq_len, seq_len]
        attention_mask = self._create_attention_mask_from_input_mask(input_mask, True)

        # GPT2 decoder
        decoder_output = self.gpt2_decoder(
            self.cast_compute_type(embedding_output),
            self.cast_compute_type(attention_mask)
        )

        # LayerNorm
        decoder_output = self.reshape(decoder_output, self.new_shape)
        decoder_output = self.layer_norm(decoder_output)
        decoder_output = self.reshape(decoder_output, (self.batch_size, self.seq_length, self.d_model))

        return decoder_output, embedding_tables