import numpy as np
import mindspore
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore.common.tensor import Tensor
from mindspore.common.parameter import Parameter
import mindspore.common.dtype as mstype
from mindspore.nn.wrap.grad_reducer import DistributedGradReducer
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import get_group_size
from mindspore.parallel._utils import _get_device_num, _get_parallel_mode

# from .grad_clip import GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE, ClipGradients
from .clip_grad_utils import clip_grad
from src.utils.CrossEntropy import CrossEntropyCalculationWithMask
# from .GPT2ForLambada import GPT2LambadaModel
# from .GPT2ForCBT import GPT2CBTModel
from .GPT2ForTranslation import GPT2TranslationModel
# from .GPT2ForLanguageModel import GPT2LanguageModel
# from .GPT2ForReadComprehension import GPT2CoQAModel
# from .GPT2ForSummarization import GPT2SummarizationModel


# grad_scale = C.MultitypeFuncGraph("grad_scale")
# reciprocal = P.Reciprocal()

# @grad_scale.register("Tensor", "Tensor")
# def tensor_grad_scale(scale, grad):
#     return grad * F.cast(reciprocal(scale), F.dtype(grad))

# _grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
# grad_overflow = P.FloatStatus()

# @_grad_overflow.register("Tensor")
# def _tensor_grad_overflow(grad):
#     return grad_overflow(grad)


# class GPT2FinetuneCell(nn.Cell):
#     """
#     Especifically defined for finetuning where only four inputs tensor are needed.
#     """
#     def __init__(self, network, optimizer, scale_update_cell=None):
#         super(GPT2FinetuneCell, self).__init__(auto_prefix=False)
#         self.network = network
#         self.network.set_grad()
#         self.weights = optimizer.parameters
#         self.optimizer = optimizer
#         self.grad = C.GradOperation(get_by_list=True, sens_param=True)
#         self.reducer_flag = False
#         self.allreduce = P.AllReduce()
#         self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
#         # self.parallel_mode = _get_parallel_mode()
#         # self.parallel_mode = "stand_alone"
#         if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
#             self.reducer_flag = True
#         self.grad_reducer = None
#         if self.reducer_flag:
#             mean = context.get_auto_parallel_context("gradients_mean")
#             # degree = get_group_size()
#             degree = _get_device_num()
#             self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
#         self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
#         self.clip_gradients = ClipGradients()
#         self.cast = P.Cast()
#         self.gpu_target = False
#         if context.get_context("device_target") == "GPU":
#             self.gpu_target = True
#             self.float_status = P.FloatStatus()
#             self.addn = P.AddN()
#             self.reshape = P.Reshape()
#         else:
#             self.alloc_status = P.NPUAllocFloatStatus()
#             self.get_status = P.NPUGetFloatStatus()
#             self.clear_before_grad = P.NPUClearFloatStatus()
#         self.reduce_sum = P.ReduceSum(keep_dims=False)
#         self.depend_parameter_use = P.ControlDepend(depend_mode=1)
#         self.base = Tensor(1, mstype.float32)
#         self.less_equal = P.LessEqual()
#         self.hyper_map = C.HyperMap()
#         self.loss_scale = None
#         self.loss_scaling_manager = scale_update_cell
#         if scale_update_cell:
#             self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
#                                         name="loss_scale")

#     def construct(self,
#                   input_ids,
#                   input_mask,
#                   label_ids,
#                   sens=None):
#         """
#         GPT-2 Finetune.
#         Construct network.

#         Args:
#             input_ids (Tensor): Source sentence.
#             input_mask (Tensor): Source padding mask.
#             label_ids (Tensor): Target sentence.
#             sens (Tensor): Loss sen.

#         Returns:
#             Tuple[Tensor, Tensor, Tensor], loss, overflow, sen.
#         """

#         weights = self.weights
#         init = False
#         loss = self.network(input_ids,
#                             input_mask,
#                             label_ids)
#         if sens is None:
#             scaling_sens = self.loss_scale
#         else:
#             scaling_sens = sens

#         if not self.gpu_target:
#             init = self.alloc_status()
#             clear_before_grad = self.clear_before_grad(init)
#             F.control_depend(loss, init)
#             # self.depend_parameter_use(clear_before_grad, scaling_sens)
#         grads = self.grad(self.network, weights)(input_ids,
#                                                  input_mask,
#                                                  label_ids,
#                                                  self.cast(scaling_sens,
#                                                            mstype.float32))
#         grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
#         grads = self.clip_gradients(grads, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE)
#         # grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
#         if self.reducer_flag:
#             # apply grad reducer on grads
#             grads = self.grad_reducer(grads)
#         # get the overflow buffer
#         if not self.gpu_target:
#             flag = self.get_status(init)
#             flag_sum = self.reduce_sum(init, (0,))
#             F.control_depend(grads, flag)
#             F.control_depend(flag, flag_sum)
#         else:
#             flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
#             flag_sum = self.addn(flag_sum)
#             # convert flag_num to scalar
#             flag_sum = self.reshape(flag_sum, (()))
#         if self.is_distributed:
#             flag_reduce = self.allreduce(flag_sum)
#             cond = self.less_equal(self.base, flag_reduce)
#         else:
#             cond = self.less_equal(self.base, flag_sum)
#         overflow = cond
#         if sens is None:
#             overflow = self.loss_scaling_manager(self.loss_scale, cond)
#         if overflow:
#             succ = False
#         else:
#             succ = self.optimizer(grads)
#         ret = (loss, cond, scaling_sens)
#         return F.depend(ret, succ)


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = C.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = C.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()
@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class GPT2FinetuneCell(nn.Cell):
    """
    Especifically defined for finetuning where only four inputs tensor are needed.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):

        super(GPT2FinetuneCell, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = C.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.reducer_flag = False
        self.allreduce = P.AllReduce()
        self.parallel_mode = context.get_auto_parallel_context("parallel_mode")
        if self.parallel_mode in [ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL]:
            self.reducer_flag = True
        self.grad_reducer = None
        if self.reducer_flag:
            mean = context.get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(optimizer.parameters, mean, degree)
        self.is_distributed = (self.parallel_mode != ParallelMode.STAND_ALONE)
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.ControlDepend(depend_mode=1)
        self.base = Tensor(1, mstype.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = C.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=mstype.float32),
                                        name="loss_scale")

    def construct(self,
                  input_ids,
                  input_mask,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        if not self.gpu_target:
            init = self.alloc_status()
            clear_before_grad = self.clear_before_grad(init)
            F.control_depend(loss, init)
            self.depend_parameter_use(clear_before_grad, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           mstype.float32))
        grads = self.hyper_map(F.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(F.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if self.reducer_flag:
            grads = self.grad_reducer(grads)
        if not self.gpu_target:
            flag = self.get_status(init)
            flag_sum = self.reduce_sum(init, (0,))
            F.control_depend(grads, flag)
            F.control_depend(flag, flag_sum)
        else:
            flag_sum = self.hyper_map(F.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))
        if self.is_distributed:
            flag_reduce = self.allreduce(flag_sum)
            cond = self.less_equal(self.base, flag_reduce)
        else:
            cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return F.depend(ret, succ)


class GPT2LM(nn.Cell):
    def __init__(self, config=None, is_training=None, use_one_hot_embeddings=False):
        super(GPT2LM, self).__init__()
        self.gpt2 = GPT2LanguageModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training, num_labels=self.num_labels, config=config)
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()
        # self.gather = P.GatherV2()
        # self.label_indices = Tensor(np.array([x for x in range(1, config.seq_length)]), mindspore.int32)

    def construct(self, input_ids, input_mask, label_ids):
        lm_logits = self.gpt2(input_ids, input_mask) # [batch_size, seq_length, vocab_size]

        shift_logits = lm_logits[::, :-1, ::] # [batch_size, seq_length - 1, vocab_size]
        shift_logits = self.reshape(shift_logits, (-1, self.num_labels)) # [batch * (seq_length - 1), vocab_size]
        # label_ids = self.gather(label_ids, self.label_indices, 1) # [batch, seq_len -1]
        label_ids = label_ids[::, 1:]
        input_mask = input_mask[::, 1:]

        loss = self.loss(shift_logits, label_ids, input_mask)
        # return self.cast(loss, mstype.float32)
        return loss


class GPT2Lambada(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2Lambada, self).__init__()
        self.gpt2 = GPT2LambadaModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training, num_labels=self.num_labels, config=config)
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask, label_ids):
        lm_logits = self.gpt2(input_ids, input_mask) # [batch_size, seq_length, vocab_size]
        
        if self.is_training:
            shift_logits = lm_logits[:, :-1, :] # [batch_size, seq_length - 1, vocab_size]
            shift_logits = self.reshape(shift_logits, (-1, self.num_labels)) # [batch * (seq_length - 1), vocab_size]
        
            label_ids = label_ids[::, 1:]
            input_mask = input_mask[::, 1:]
            loss = self.loss(shift_logits, label_ids, input_mask)
            return loss
        
        return lm_logits

class GPT2CBT(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False, num_labels=10):
        super(GPT2CBT, self).__init__()
        self.gpt2 = GPT2CBTModel(config, is_training, use_one_hot_embeddings, num_labels=num_labels)
        self.loss1 = CrossEntropyCalculation(is_training=is_training)
        self.loss2 = CrossEntropyCalculation(is_training=is_training)
        self.mc_num_labels = num_labels
        self.lm_num_labels = config.vocab_size
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.shape = P.Shape()
        self.reshape = P.Reshape()
        self.cast = P.Cast()

    def construct(self, input_ids, input_mask, mc_token_ids, label_ids, mc_labels):
        lm_logits, mc_logits = self.gpt2(input_ids, input_mask, mc_token_ids)

        mc_loss = None
        if mc_labels is not None:
            mc_loss = self.loss1(mc_logits, mc_labels, self.mc_num_labels)

        lm_loss = None
        if label_ids is not None:
            lm_logits_shape = self.shape(lm_logits)
            lm_logits = self.reshape(lm_logits, (lm_logits_shape[0], lm_logits_shape[1], lm_logits_shape[2]))
            shift_lm_logits = lm_logits[:, :-1, :]
            shift_labels = label_ids[:, 1:] # problem###############
            lm_loss = self.loss2(shift_lm_logits, shift_labels, self.lm_num_labels)

        if mc_loss is not None:
            loss = mc_loss + lm_loss
        else:
            loss = lm_loss

        return self.cast(loss, mstype.float32)


class GPT2Translation(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2Translation, self).__init__()
        self.gpt2 = GPT2TranslationModel(config, is_training, use_one_hot_embeddings)
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculationWithMask(is_training=is_training, num_labels=self.num_labels, config=config)
        self.is_training = is_training
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.gather = P.GatherV2()
        self.indices1 = Tensor(np.array([x for x in range(config.seq_length - 1)]), mindspore.int32)
        self.indices2 = Tensor(np.array([x for x in range(1, config.seq_length)]), mindspore.int32)

    def construct(self, input_ids, input_mask, label_ids):
        translation_logits = self.gpt2(input_ids, input_mask) # [batch_size, seq_length, vocab_size]
        translation_logits = self.log_softmax(translation_logits)

        # shift_logits = translation_logits[::, :-1, ::] # [batch_size, seq_length - 1, vocab_size]
        shift_logits = self.gather(translation_logits, self.indices1, 1)
        shift_logits = self.reshape(shift_logits, (-1, self.num_labels)) # [batch * (seq_length - 1), vocab_size]
        
        label_ids = self.gather(label_ids, self.indices2, 1) # [batch, seq_len -1]
        # label_ids = label_ids[::, 1:] # [batch, seq_len -1]
        # input_mask = input_mask[::, 1:]
        input_mask = self.gather(input_mask, self.indices2, 1) # [batch, seq_len -1]
        
        loss = self.loss(shift_logits, label_ids, input_mask)
        return loss


class GPT2CoQA(nn.Cell):
    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(GPT2CoQA, self).__init__()
        self.gpt2 = GPT2CoQAModel(config, is_training, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training=is_training)
        self.is_training = is_training
        self.num_labels = config.vocab_size
        self.loss = CrossEntropyCalculation(is_training=is_training)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.label_indices = Tensor(np.array([x for x in range(1, config.seq_length)]), mindspore.int32)

    def construct(self, input_ids, input_mask, label_ids):
        logits = self.gpt2(input_ids, input_mask)
        shift_logits = logits[:,:-1,:]
        shift_logits = P.Reshape()(shift_logits,(-1,self.num_labels))
        label_ids = P.GatherV2()(label_ids, self.label_indices, 1)

        loss = self.loss(shift_logits, label_ids, self.num_labels)
        return P.Cast()(loss, mstype.float32)


class GPT2Summarization(nn.Cell):
    def __init__(self, config=None, is_training=None, use_one_hot_embeddings=False):
        super(GPT2Summarization, self).__init__()
        self.gpt2 = GPT2SummarizationModel(config, is_training, use_one_hot_embeddings)
        self.is_training = is_training
        self.last_idx = (-1,)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.reshape = P.Reshape()
        self.shape = P.Shape()
        self.batch_size = config.batch_size
        self.seq_length = config.seq_length
        self.vocab_size = config.vocab_size
        self.cast = P.Cast()
        self.loss_function = CrossEntropyCalculationWithMask(num_labels=self.vocab_size,is_training=self.is_training,config=config)
    def construct(self, input_ids,input_mask,label_ids):
        output = self.gpt2(input_ids,input_mask)

        pre_lm_logits = output[::, :-1, ::]

        shift_squeezed_logits = self.reshape(pre_lm_logits, (-1, self.vocab_size))
        shift_squeezed_labels = label_ids[::, 1:]

        input_mask = input_mask[::, 1:]

        shift_squeezed_logits =  self.log_softmax(shift_squeezed_logits)

        loss = self.loss_function(shift_squeezed_logits, shift_squeezed_labels,input_mask)
        return loss
        #return self.cast(loss, mstype.float32)
