# -*- coding: utf-8 -*-
import os
import sys
import argparse
import math
import regex as re
import mindspore
from src.GPT2ForSummarization import GPT2SummarizationModel
from src.gpt2_for_finetune import GPT2Summarization,GPT2FinetuneCell
from src.finetune_eval_config import cfg, gpt2_net_cfg
from src.utils.metric_method import Rouge
from mindspore.nn import Accuracy
from src.dataset import create_cnn_dailymail_dataset
from src.utils.lr_schedule import GPT2LearningRate
from src.utils.losscallback import LossCallBack
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn import AdamWeightDecay, Lamb, Momentum
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.utils.tokenization import Tokenizer
from src.utils.generation_utils import GenerationConfig
from mindspore.ops import operations as P
from src.GPT2_generation import generate_for_CNN_DAILYMAIL

def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1,resume=False):
    """
    Do train
    Args:
        dataset: the train dataset.
        network:  the network with loss
        load_checkpoint_path: the file path which saved pretrain model checkpoint.
        save_checkpoint_path:  the file path which will save finetune model checkpoint.
        epoch_num: the number of epoch
    """
    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    
    steps_per_epoch = dataset.get_dataset_size() # samples / batch_size
    
    #print info
    print("="*30,"TRAIN INFO","="*30)
    
    print("optimizer: {}".format(cfg.optimizer))
    
    
    
    #Select Optimizer
    if cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = GPT2LearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=cfg.AdamWeightDecay.power)
        params = network.trainable_params() # return a list of all trainable parmeters of the network

        # Use parameter groups and set different values
        decay_params = list(filter(cfg.AdamWeightDecay.decay_filter, params)) # without layernorm and bias
        other_params = list(filter(lambda x: not cfg.AdamWeightDecay.decay_filter(x), params)) # with layernorm and bias
        group_params = [{'params': decay_params, 'weight_decay': cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]
        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=cfg.AdamWeightDecay.eps)
    elif cfg.optimizer == 'Lamb':
    
        #print info
        print("lr: {}".format(cfg.Lamb.learning_rate))
        print("end_learning_rate: {}".format(cfg.Lamb.end_learning_rate))
        #print("warmup_steps: {}".format(int(steps_per_epoch * epoch_num * 0.1)))
        print("power: {}".format(cfg.Lamb.power))
        
        lr_schedule = GPT2LearningRate(learning_rate=cfg.Lamb.learning_rate,
                                       end_learning_rate=cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), lr_schedule)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), cfg.Momentum.learning_rate, cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="gpt2_summarization",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)
    reorganized_param_dict = dict()
    if resume == False :
        print("Do not resume.\nRESUME STATE: {}".format(resume))
        for netName in param_dict:
            reorganized_param_dict['gpt2.gpt2.'+netName] = param_dict[netName]
        reorganized_param_dict['gpt2.lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
    else:
        print("Start to resume training.\nRESUME STATE: {}".format(resume))
        reorganized_param_dict = param_dict
    load_param_into_net(network, reorganized_param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = GPT2FinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads.set_train(True)
    loss_cb = LossMonitor(per_print_times=1)
    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), loss_cb, ckpoint_cb]
    print("============== Starting Training For Summrization Task ==============")
    model.train(epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=False)
    print("============== Summrization Training Success ==============")


def eval_result_print(metric="Rouge", callback=None):
    """ print eval result"""
    if metric == "Rouge":
        print("Rouge-1 {:.8f}, Rouge-2 {:.8f}, Rouge-L {:.8f}, Rouge-AVG{:.8f}".format(callback.Rouge1/callback.total_num, callback.Rouge2/callback.total_num,
                                                                 callback.RougeL / callback.total_num,(callback.Rouge1+callback.Rouge2+callback.RougeL) / (3.0*callback.total_num) ))
    else:
        raise ValueError("metric method '{}' not supported, support: [Rouge]. ".format(str(metric)))

def modify_paramdict(param_dict,mode="zero-shot",model_prefix="gpt2."):
    """
    modify keys of param_dict to fit model.

    Args:
        param_dic: dict, dictionary of parameters imported from a ckpt file
        mode:   str, "zero-shot" for an pretrained GPT2 model; 
                "finetune" for an finetuned model for certain task.
    Return:
        reorganized_param_dict: dict, new param_dict to fit in model for different tasks.
    """
    reorganized_param_dict = dict()
    if mode == "zero-shot":        
        for netName in param_dict:
            reorganized_param_dict[model_prefix+netName] = param_dict[netName]
        reorganized_param_dict['lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
        return reorganized_param_dict
    elif mode == "finetune":
        embedding_name = "gpt2_embedding_lookup.embedding_table"
        embedding_name_old = ""
        for netName in param_dict:
            netName_remove_prefix = netName[len(model_prefix):]
            netName_prefix = netName[:len(model_prefix)]
            reorganized_param_dict[netName_remove_prefix] = param_dict[netName]
            if embedding_name in netName and netName_prefix == model_prefix:
                embedding_name_old = netName
        reorganized_param_dict[embedding_name] = param_dict[embedding_name_old]
        return reorganized_param_dict

        
    else:
         raise NotImplementedError


def clean_hypo(text):
    """
    to prevent generation of empty string, and lower text

    Arg:
        text: str, input str
    Return:
        text: str, cleaned input str
    """
    text = text.lower()
    eng_re = re.compile(r'[a-z]+',re.I)
    if len(eng_re.findall(text)) == 0:
        return '<EMPTY>'
    else:
        return text


def do_eval(dataset=None, network=None, metric=None, load_checkpoint_path="",eval_load_param_mode="zero-shot",generation_config_path="",tokenizer_file=""):
    """
    Do evaluation on summarization
    Args:
        dataset: the eval dataset.
        network:  the network with loss.
        metric: the evaluation method.
        load_checkpoint_path: the file path which saved finetune model checkpoint.
    """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    if metric.lower() == "rouge":
        print("Prepare to calculate the Rouge score ...")
        callback = Rouge()
        
        #initialize network and load params
        gpt2_loss = network(config=gpt2_net_cfg,
                            is_training=False,
                            use_one_hot_embeddings=False)
        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)
       
        #get reorganized param_dict and load parms into network
        reorganized_param_dict = modify_paramdict(param_dict,mode=eval_load_param_mode,model_prefix="gpt2.")
        load_param_into_net(gpt2_loss, reorganized_param_dict)


        #load nn.Cell into Model and initiate tokenizer and Sample
        model = Model(gpt2_loss)
        tokenizer = Tokenizer(vocab_file=tokenizer_file+'gpt2-vocab.json',
        merge_file=tokenizer_file+'gpt2-merges.txt')
        generate_config = GenerationConfig( file_path=generation_config_path)
        TL_DR = generate_config.get_arg("tldr") if generate_config.get_arg("tldr") is not None else True
        tldr_str = generate_config.get_arg("tldr_str") if generate_config.get_arg("tldr_str") is not None else "TL;DR:"
        #sample = Sample(model,tokenizer=tokenizer,model_config=gpt2_net_cfg,topk_num = topk,topp_prob=topp,
        #min_tokens_to_keep=1,demo_mode=False,temperature=temperature,append_eos=append_eos)

        #load data and process text generation
        columns_list = ["input_ids", "input_mask", "label_ids"]
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data

            print("input_ids shape: {}".format(input_ids.shape))
            print("label_ids shape: {}".format(label_ids.shape))
            print("="*15," Summrization Testing ","="*15)
           
            hypo,ref = generate_for_CNN_DAILYMAIL(model,input_ids,
                                                select_sentence=3,
                                                TL_DR=TL_DR,
                                                tldr_str=tldr_str,
                                                tokenizer=tokenizer,
                                                generate_config=generate_config)

            print("REF str:\n ",ref,"\nHYPO str:\n",hypo,"\n")

            for i in range(gpt2_net_cfg.batch_size):
                hypo[i] = clean_hypo(hypo[i])
            
            for i in range(gpt2_net_cfg.batch_size):
                hypo[i] = hypo[i].lower()
                ref[i] = ref[i].lower()
            
            callback.update(hypo,ref)

        print("="*35)
        eval_result_print(metric, callback)
        print("="*35)
        print("*"*15," Summrization Testing Finished","*"*15)
    
    else:
        raise ValueError("metric method not supported in summarization, support: [Rouge]")


def run_summarization():
    '''
    run Summarization_task

    '''

    #set argument parser
    parser = argparse.ArgumentParser(description="Finetune and Evaluate Summrization")


    #context and task settings
    parser.add_argument("--device_target", type=str, default="GPU",
                        help="Device type. Default: GPU.")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of target device. ")
    parser.add_argument("--do_train", type=str, default="false",
                        help="Enable train. Default: false.")
    parser.add_argument("--do_eval", type=str, default="false",
                        help="Enable evaluation. Default: false.")
    parser.add_argument("--metric_method", type=str, default="Rouge",
                        help="The eval method including [Rouge(Rouge1,Rouge2,RougeL,Rouge Avg)]. Default: Rouge.") 
    parser.add_argument("--epoch_num", type=int, default=2,
                        help="Epoch number. Default: 2.")
    parser.add_argument("--resume", type=str, default="false",
                        help="resume trainning or not")

    #dataset and params_dict file settings
    parser.add_argument("--train_data_shuffle", type=str, default="true",
                        help="Enable train data shuffle. Default: true.")
    parser.add_argument("--eval_data_shuffle", type=str, default="false",
                        help="Enable eval data shuffle. Default: false.")
    parser.add_argument("--save_finetune_ckpt_path", type=str, default="/datasets/pretrained_weights/saved/",
                        help="Save the checkpoint path.")
    parser.add_argument("--load_pretrain_ckpt_path", type=str, default="/datasets/pretrained_weights/ms_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--load_finetune_ckpt_path", type=str, default="/datasets/pretrained_weights/ms_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--train_data_file_path", type=str, default="/datasets/cnn_dailymail",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="/datasets/cnn_dailymail",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_load_param_mode", type=str, default="zero-shot",
                        help="Mode for load param of evaluation, [zero-shot,finetune]")

    # sampling settings
    parser.add_argument("--top_k", type=int, default=2,
                        help="top k tokens chosen for sampling")
    parser.add_argument("--top_p", type=float, default=1.0,
                        help="top p accumulated probability thresold for logit to be counted")
    parser.add_argument("--temp", type=float, default=1.0,
                        help="temperature on logits for sampling")
    parser.add_argument("--append_eos", type=bool, default=False,
                        help="if append <EOS> token to the end of input str")
    parser.add_argument("--generation_config_path", type=str, default=".scripts/summary_generation_config.json",
                        help="if append <EOS> token to the end of input str")
    parser.add_argument("--tokenizer_file_path", type=str, default=sys.path[0]+"/src/utils/pretrain-data/",
                        help="vocab & merge file path") 

        
    #get args
    args_opt = parser.parse_args()

    epoch_num = args_opt.epoch_num
    metric = args_opt.metric_method
    save_finetune_ckpt_path = args_opt.save_finetune_ckpt_path
    load_finetune_ckpt_path = args_opt.load_finetune_ckpt_path
    load_pretrain_ckpt_path = args_opt.load_pretrain_ckpt_path
    eval_load_param_mode = args_opt.eval_load_param_mode

    tokenizer_file = args_opt.tokenizer_file_path
    
    #resume training True or False
    resume = True if args_opt.resume.lower() == "true" else False
    
    topk = args_opt.top_k
    topp = args_opt.top_p
    temperature = args_opt.temp
    append_eos = args_opt.append_eos
    generation_config_path = args_opt.generation_config_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    device = args_opt.device_target
    if device == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id,max_call_depth=3000)
        context.set_auto_parallel_context(parallel_mode="stand_alone")
    elif device == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
        context.set_auto_parallel_context(parallel_mode="stand_alone")
    else:
        raise Exception("Device target error, Ascend and Nvidia GPU is supported.")
    
    

    if args_opt.do_train.lower() == "true":
        train_data_file_path = args_opt.train_data_file_path
        gpt2_loss = GPT2Summarization(config=gpt2_net_cfg,
                         is_training=True,
                         use_one_hot_embeddings=False)
        print("============== Start Loading Train Dataset ==============")
        train_dataset = create_cnn_dailymail_dataset(
            dataset_path=train_data_file_path)
        do_train(train_dataset, gpt2_loss, load_pretrain_ckpt_path, save_finetune_ckpt_path, epoch_num,resume)

    if args_opt.do_eval.lower() == "true":
        eval_dataset_file_path = args_opt.eval_data_file_path
        print("============ Start Loading Evaluation Dataset ============")
        eval_dataset = create_cnn_dailymail_dataset(
            dataset_path=eval_dataset_file_path)
        do_eval(eval_dataset, GPT2SummarizationModel, metric, load_finetune_ckpt_path,eval_load_param_mode,generation_config_path,tokenizer_file)



if __name__ == "__main__":
    run_summarization()
