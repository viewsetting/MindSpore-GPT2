import os
import argparse
import math
import mindspore
from src.GPT2ForSummarization import GPT2ForPredictNext
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
    load_param_into_net(network, param_dict)

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = GPT2FinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads.set_train(True)

    model = Model(netwithgrads)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    print("============== Starting Training For Summrization Task ==============")
    model.train(epoch_num, dataset, callbacks=callbacks)
    print("============== Summrization Training Success ==============")


def eval_result_print(metric="Rouge", callback=None):
    """ print eval result"""
    if metric == "Rouge":
        print("Rouge-1 {:.8f}, Rouge-2 {:.8f}, Rouge-L {:.8f}".format(callback.Rouge1/callback.total_num, callback.Rouge2/callback.total_num,
                                                                 callback.RougeL / callback.total_num))
    else:
        raise ValueError("metric method '{}' not supported, support: [Rouge]. ".format(str(metric)))


def do_eval(dataset=None, network=None, metric=None, load_checkpoint_path=""):
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
    if metric.lower() == "rouge":
        print("Prepare to calculate the Rouge score ...")
        callback = Rouge()
        gpt2_loss = network(config=gpt2_net_cfg,
                            is_training=False,
                            use_one_hot_embeddings=False)

        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)
        load_param_into_net(gpt2_loss, param_dict)
        model = Model(gpt2_loss)

        columns_list = ["input_ids", "input_mask", "label_ids"]
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data
            input_ids = Tensor(input_ids, mindspore.int32)
            input_mask = Tensor(input_mask, mindspore.int32)
            label_ids = Tensor(label_ids, mindspore.int32)
            print("input_ids shape: {}".format(input_ids.shape))
            print("label_ids shape: {}".format(label_ids.shape))
            print("============= Summrization Testing =============")
            logits = model.predict(input_ids, input_mask)
            print("logits shape: {}".format(logits.shape))
            str1="a a a test"
            str2="b b b test"
            callback.update(str1, str2)
        print("==============================================")
        eval_result_print(metric, callback)
        print("==============================================")
        print("************** Summarization Testing Finished **************")
    
    else:
        raise ValueError("metric method not supported in summarization, support: [Rouge]")



def run_summarization():
    '''
    run Summarization_task

    '''
    parser = argparse.ArgumentParser(description="Finetune and Evaluate Summrization")
    parser.add_argument("--device_target", type=str, default="GPU",
                        help="Device type. Default: GPU.")
    parser.add_argument("--device_id", type=int, default=0,
                        help="ID of target device. ")
    parser.add_argument("--metric_method", type=str, default="Rouge",
                        help="The eval method including [Rouge(Rouge1,Rouge2,RougeL,Rouge Avg)]. Default: Rouge.") 
    parser.add_argument("--do_train", type=str, default="false",
                        help="Enable train. Default: false.")
    parser.add_argument("--do_eval", type=str, default="true",
                        help="Enable evaluation. Default: false.")
    parser.add_argument("--epoch_num", type=int, default=2,
                        help="Epoch number. Default: 1.")
    parser.add_argument("--train_data_shuffle", type=str, default="true",
                        help="Enable train data shuffle. Default: true.")
    parser.add_argument("--eval_data_shuffle", type=str, default="false",
                        help="Enable eval data shuffle. Default: false.")
    parser.add_argument("--save_finetune_ckpt_path", type=str, default="/datasets/pretrained_weights/saved/",
                        help="Save the checkpoint path.")
    parser.add_argument("--load_pretrain_ckpt_path", type=str, default="/datasets/pretrained_weights/mindspore_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--load_finetune_ckpt_path", type=str, default="/datasets/pretrained_weights/mindspore_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--train_data_file_path", type=str, default="/datasets/cnn_dailymail",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="/datasets/cnn_dailymail",
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

    gpt2_loss = GPT2Summarization(config=gpt2_net_cfg,
                         is_training=True,
                         use_one_hot_embeddings=False)

    if args_opt.do_train.lower() == "true":
        print("============== Start Loading Train Dataset ==============")
        train_dataset = create_cnn_dailymail_dataset(
            dataset_path="/datasets/cnn_dailymail/cnn_dailymail-test-mindrecord")
        do_train(train_dataset, gpt2_loss, load_pretrain_ckpt_path, save_finetune_ckpt_path, epoch_num)

    if args_opt.do_eval.lower() == "true":
        print("============ Start Loading Evaluation Dataset ============")
        eval_dataset = create_cnn_dailymail_dataset(
            dataset_path="/datasets/cnn_dailymail/cnn_dailymail-test-mindrecord")
        do_eval(eval_dataset, GPT2ForPredictNext, metric, load_finetune_ckpt_path)



if __name__ == "__main__":
    run_summarization()
