import os
import numpy
import argparse
import math
from src.gpt2_for_finetune import GPT2FinetuneCell, GPT2Lambada
from src.GPT2ForLambada import GPT2LambadaModel
from src.finetune_eval_config import cfg, gpt2_net_cfg
from src.utils.metric_method import LastTokenAccuracy,LastWordAccuracy 
from src.dataset import create_language_model_dataset
from src.utils.lr_schedule import GPT2LearningRate
from src.utils.losscallback import LossCallBack
from src.utils.extract_logits_lambada import extract_logits_for_lambada
from src.utils.lambada_utils import get_wholeword_pair,get_wholeword_label_str
from src.utils.tokenization import Tokenizer
from src.GPT2_generation import Sample
import mindspore
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.nn.wrap.loss_scale import DynamicLossScaleUpdateCell
# from mindspore.nn import AdamWeightDecay, Lamb, Momentum, DynamicLossScaleUpdateCell
from mindspore.nn import AdamWeightDecay, Lamb, Momentum
from mindspore.common.tensor import Tensor
from mindspore.train.model import Model
from mindspore.train.callback import CheckpointConfig, ModelCheckpoint, TimeMonitor, LossMonitor
from mindspore.train.serialization import load_checkpoint, load_param_into_net
# from src.GPT2_generation import Sample
def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
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

    steps_per_epoch = dataset.get_dataset_size() # samples / batch_size  doing####

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
    ckpoint_cb = ModelCheckpoint(prefix="gpt2_language_model_wiki2",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)

    final_param_dict = {}
    for k, v in param_dict.items():
        final_param_dict['gpt2_loss.gpt2.gpt2.' + k] = param_dict[k]
    # set the weights of final linear weights to weights of gpt2 token embedding
    final_param_dict['gpt2_loss.gpt2.dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']

    load_param_into_net(network, final_param_dict)
    print("Load new parameter successfully!\n")

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = GPT2FinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads.set_train(True)

    loss_cb = LossMonitor()

    model = Model(netwithgrads)
    # callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    callbacks = [TimeMonitor(dataset.get_dataset_size()), loss_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(epoch_num, dataset, callbacks=callbacks)
    print("============== Training Success ==============")


def eval_result_print(metric="accuracy", callback=None):
    """ print eval result"""
    if metric.lower() == "accuracy":
        print("acc_num {}, total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                 callback.acc_num / callback.total_num))
    else:
        raise ValueError("metric method not supported, support: [accuracy]")


def do_eval(dataset=None, metric=None, load_checkpoint_path=""):
    """
    Do eval
    Args:
        dataset: the eval dataset.
        network:  the network with loss.
        metric: the evaluation method.
        load_checkpoint_path: the file path which saved finetune model checkpoint.
    """
    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    

    if metric.lower() == "accuracy":
        print("Prepare to calculate the accuracy score ...")
        # callback = Accuracy()
        # callback = LastWordAccuracy()
        # callback = LastTokenAccuracy()
        callback = LastWordAccuracy(smooth=False)
        gpt2_loss = GPT2LambadaModel(config=gpt2_net_cfg,
                           is_training=False,
                           use_one_hot_embeddings=False)

        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)
        final_param_dict = {}
        for k, v in param_dict.items():
            final_param_dict['gpt2.' + k] = param_dict[k]


        # set the weights of final linear weights to weights of gpt2 token embedding
        final_param_dict['dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
        load_param_into_net(gpt2_loss, final_param_dict)
        model = Model(gpt2_loss)
        tokenizer = Tokenizer(vocab_file='./src/utils/pretrain-data/gpt2-vocab.json',
                            merge_file='./src/utils/pretrain-data/gpt2-merges.txt')
        
        sample = Sample(decoder = model,model_config=gpt2_net_cfg,tokenizer=tokenizer,topk_num=1,topp_prob=1,return_ids=True)
        columns_list = ["input_ids", "input_mask", "label_ids"]
        print("============= Testing LAMBADA ACC =============")
        cnt  = 0
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data
            print("===========LAMBADA ACC iteration:{}==========".format(cnt))
            # input_ids = Tensor(input_ids, mindspore.int32)
            # input_mask = Tensor(input_mask, mindspore.int32)
            # label_ids = Tensor(label_ids, mindspore.int32)
            print("input_ids_shape: {}".format(input_ids.shape))
            print("input_mask_shape: {}".format(input_mask.shape))
            print("label_ids_shape: {}".format(label_ids.shape))
            
            logits = model.predict(input_ids, input_mask)
            print("="*40)
            # print("after predict logits shape:",logits.shape)         (8,1024,50257)
            output_str = sample.generate_for_LAMBADA(input_ids = input_ids,logits = logits, max_generate_length=3, max_iterations=30)
            label_str = get_wholeword_label_str(input_ids=input_ids,config=gpt2_net_cfg,tokenizer=tokenizer)
            # print("logits shape: {}".format(logits.shape))
            # print("logits: \n{}".format(logits))
            # print("===================================")
            print("==============================================")
            print(output_str)
            print(label_str)
            callback.update(output_str, label_str)
            # callback.update(logits, label_ids)  
            cnt += 1
        print("==============================================")
        eval_result_print(metric, callback)
        print("************** Testing Finished **************")

    elif metric.lower() == "ppl":
        print("Prepare to calculate the ppl score ........")
        # ppl metric can be calculated by using the loss, so the difference is 'is_training'
        gpt2_loss = GPT2Lambada(config=gpt2_net_cfg,
                           is_training=True,
                           use_one_hot_embeddings=False)
        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)

        final_param_dict = {}
        for k, v in param_dict.items():
            final_param_dict['gpt2_loss.gpt2.gpt2.' + k] = param_dict[k]


        # set the weights of final linear weights to weights of gpt2 token embedding
        final_param_dict['gpt2_loss.gpt2.dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']

        load_param_into_net(gpt2_loss, final_param_dict)
        # dict_ = gpt2_loss.parameters_dict()
        # for k, v in dict_.items():
        #     print('name: {}\n'.format(k))
        #     print('value: {}'.format(dict_[k]))
        #     print("--------------------------------\n")

        # exit()


        print("load new parameter successfully!\n")
        model = Model(gpt2_loss)

        columns_list = ["input_ids", "input_mask", "label_ids"]
        print("================= Testing LAMBADA PPL =================")
        num_data = 0
        total_ppl = 0.0
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data
            print("input_ids_shape: {}".format(input_ids.shape))
            print("input_mask_shape: {}".format(input_mask.shape))
            print("label_ids_shape: {}".format(label_ids.shape))

            loss = model.predict(input_ids, input_mask, label_ids)
            loss = loss.asnumpy()
            ppl = math.exp(float(loss))
            print("Loss: {:.6f}".format(float(loss)))
            print("PPL: {}\n\n".format(ppl))
            num_data += 1
            total_ppl += ppl
        avg_ppl = total_ppl / num_data
        print("Average PPL: {:.6f}".format(avg_ppl))    
        print("************** Testing Finished **************")
    else:
        raise ValueError("metric method not supported, support: [accuracy, ppl]")


def run_lambada():
    """
    run Language Modeling task
    """
    parser = argparse.ArgumentParser(description="Finetune and Evaluate languagemodel")
    parser.add_argument("--device_target", type=str, default="GPU",
                        help="Device type. Default: Ascend.") ### modify
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of target device. ")
    parser.add_argument("--metric_method", type=str, default="Accuracy",
                        help="The eval method including [Accuracy, PPL]. Default: Accuracy.") # DOING
    parser.add_argument("--do_train", type=str, default="false",
                        help="Enable train. Default: false.")
    parser.add_argument("--do_eval", type=str, default="true",
                        help="Enable evaluation. Default: false.")
    parser.add_argument("--epoch_num", type=int, default=5,
                        help="Epoch number. Default: 1.")
    parser.add_argument("--train_data_shuffle", type=str, default="true",
                        help="Enable train data shuffle. Default: true.")
    parser.add_argument("--eval_data_shuffle", type=str, default="false",
                        help="Enable eval data shuffle. Default: false.")
    parser.add_argument("--save_finetune_ckpt_path", type=str, default="./pretrained-weight/",
                        help="Save the checkpoint path.")
    ## modify
    parser.add_argument("--load_pretrain_ckpt_path", type=str, default="./pretrained-weight/mindspore_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--load_finetune_ckpt_path", type=str, default="./pretrained-weight/mindspore_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--train_data_file_path", type=str, default="./src/mindspore-dataset/lambada-train-mindrecord",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="./src/mindspore-dataset/lambada-test-mindrecord",
                        help="Data path, it is better to use absolute path")
    args_opt = parser.parse_args()

    epoch_num = args_opt.epoch_num
    metric = args_opt.metric_method
    save_finetune_ckpt_path = args_opt.save_finetune_ckpt_path
    load_finetune_ckpt_path = args_opt.load_finetune_ckpt_path
    load_pretrain_ckpt_path = args_opt.load_pretrain_ckpt_path

    if args_opt.do_train.lower() == "false" and args_opt.do_eval.lower() == "false":
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train.lower() == "true" and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval.lower() == "true" and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    device = args_opt.device_target
    if device == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id)
        context.set_auto_parallel_context(parallel_mode="stand_alone")
    else:
        raise Exception("Device target error, Ascend is supported.")

    gpt2_loss = GPT2Lambada(config=gpt2_net_cfg,
                       is_training=True,
                       use_one_hot_embeddings=False)

    if args_opt.do_train.lower() == "true":
        print("==============    Start Loading Train Dataset   ============")
        train_dataset = create_language_model_dataset(dataset_path=args_opt.train_data_file_path)
        do_train(train_dataset, gpt2_loss, load_pretrain_ckpt_path, save_finetune_ckpt_path, epoch_num)

    if args_opt.do_eval.lower() == "true":
        print("============== Start Loading Evaluation Dataset ============")
        eval_dataset = create_language_model_dataset(dataset_path=args_opt.eval_data_file_path)
        do_eval(eval_dataset, metric, load_finetune_ckpt_path)


if __name__ == "__main__":
    run_lambada()
