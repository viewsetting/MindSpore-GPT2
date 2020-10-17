import os
import numpy as np
from typing import TypeVar, Union
from src.finetune_eval_config import cfg, gpt2_net_cfg
from mindspore import log as logger
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.utils.tokenization import Tokenizer
from mindspore.ops import operations as P
from src.GPT2ForSummarization import GPT2ForPredictNext
from src.GPT2ForLanguageModel import GPT2LanguageModel
from src.GPT2_generation import Sample
import mindspore.nn as nn
from mindspore import context,Tensor, Model, Parameter
from mindspore import dtype as mstype


def set_env(mode="GPU", device_id=0, ckpt_path="/datasets/pretrained_weights/ms_model_medium.ckpt"):
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=mode, device_id=device_id)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    print('set context as: {}, using device {}.'.format(mode, device_id))

    gpt2_loss = GPT2ForPredictNext(config=gpt2_net_cfg,
                                   is_training=False,
                                   use_one_hot_embeddings=False)
    load_checkpoint_path = ckpt_path
    gpt2_loss.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)

    param_dict_ = {}

    print("====process param_dict========")
    for msname in param_dict:
        param_dict_['gpt2.'+msname] = param_dict[msname]
    param_dict_[
        'lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
    print("====load params into model====")
    load_param_into_net(gpt2_loss, param_dict_)

    model = Model(gpt2_loss)
    return model, gpt2_net_cfg


def get_random_tensor(shape: Union[list, tuple], mode='randn', dtype=mstype.float32):
    if mode == 'randn':
        np_array = np.random.randn(*shape)
        return Tensor(np_array, dtype=dtype)
    if mode == 'uniform':
        np_array = np.random.uniform(size=shape)
        return Tensor(np_array, dtype=dtype)
    pass


def list2tensor(lst, dtype=mstype.float32):
    return Tensor(np.array(lst), dtype=dtype)


if __name__ == '__main__':
    print('*'*65)
    print('We are now in testing mode for GPT2 Interactive Generation Demo')
    print('*'*65)
    print('Set Running Env and Load Model')
    gpt2, config = set_env(mode="GPU",device_id=0)
    generate_length = 50

    tokenizer = Tokenizer(vocab_file='./src/utils/pretrain-data/gpt2-vocab.json',
                          merge_file='./src/utils/pretrain-data/gpt2-merges.txt')

    sample = Sample(gpt2, generate_length=generate_length, tokenizer=tokenizer,
                    model_config=config, topk_num=0, topp_prob=0.92, min_tokens_to_keep=1)

    official_unicorn_demo = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

    while True:
        raw_text = input("Model Prompt >>>")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        gen_str, full_str = sample.generate(input_str=raw_text)
        print("*"*100)
        print("GPT2 Generation >>>", gen_str)
        print("*"*100)
        print("Full Text Here >>>", full_str)
        print("*"*100)
