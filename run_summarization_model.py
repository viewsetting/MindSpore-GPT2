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

def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")

    raise NotImplementedError

def do_eval(dataset=None, network=None, metric=None, load_checkpoint_path=""):
    raise NotImplementedError

def run_summarization():
    '''

    '''
    raise NotImplementedError


if __name__ == "__main__":
    run_summarization()
