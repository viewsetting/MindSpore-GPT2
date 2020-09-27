import os
import argparse
from src.gpt2_for_finetune import GPT2Summarization
from src.finetune_eval_config import cfg, gpt2_net_cfg
from src.utils.metric_method import Rouge
from mindspore.nn import Accuracy
from src.dataset import lm_train_dataset, lm_eval_dataset
from src.utils.lr_schedule import GPT2LearningRate
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn import AdamWeightDecay, Lamb, Momentum
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
