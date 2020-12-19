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
from src.utils.extract_logits_lambada import extract_logits_for_lambada,extract_last_word_input_ids
from src.utils.lambada_utils import get_wholeword_label_str,get_lastword_range
from src.utils.tokenization import Tokenizer
from src.GPT2_generation import generate_for_LAMBADA_numpy_topk
from src.utils.CrossEntropy import CrossEntropyCalculationWithMask
from src.utils.CrossEntropy import cross_entropy_np
import mindspore
import mindspore.common.dtype as mstype
from mindspore import context
from mindspore import log as logger
from mindspore.ops import operations as P
from mindspore.nn import SoftmaxCrossEntropyWithLogits
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

    steps_per_epoch = dataset.get_dataset_size() # samples / batch_size

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
    prefix_name = "gpt2_" + "lambada_" + str(cfg.gpt2_network) + "_" + str(cfg.optimizer)+ "_" + str(epoch_num) + "_bs" +str(gpt2_net_cfg.batch_size)
    ckpoint_cb = ModelCheckpoint(prefix=prefix_name,
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)
    param_dict = load_checkpoint(load_checkpoint_path)

    final_param_dict = {}
    for k, v in param_dict.items():
        final_param_dict['gpt2.gpt2.' + k] = param_dict[k]
    # set the weights of final linear weights to weights of gpt2 token embedding
    final_param_dict['gpt2.dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']

    load_param_into_net(network, final_param_dict)
    
    
    
    #print("Load the 8epoch finetuned parameter successfully!\n")
    print("Load the pretrained parameter successfully!\n")

    update_cell = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = GPT2FinetuneCell(network, optimizer=optimizer, scale_update_cell=update_cell)
    netwithgrads.set_train(True)

    loss_cb = LossMonitor(per_print_times=1)

    model = Model(netwithgrads)
    # callbacks = [TimeMonitor(dataset.get_dataset_size()), LossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    callbacks = [TimeMonitor(dataset.get_dataset_size()), loss_cb, ckpoint_cb]

    print("============== Starting Training ==============")
    model.train(epoch_num, dataset, callbacks=callbacks, dataset_sink_mode=False)
    print("============== Training Success ==============")


def eval_result_print(metric="accuracy", callback=None):
    """ print eval result"""
    if metric.lower() == "accuracy":
        print("acc_num {}, total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                 callback.acc_num / callback.total_num))
    else:
        raise ValueError("metric method not supported, support: [accuracy]")


def do_eval(dataset=None, network=None, metric=None, load_checkpoint_path="", eval_type=None, generate_length_dynamically=True):
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
    
    tokenizer = Tokenizer(vocab_file='./src/utils/pretrain-data/gpt2-vocab.json',
                          merge_file='./src/utils/pretrain-data/gpt2-merges.txt')
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
        
        if eval_type == "zero-shot":
            final_param_dict = {}
            for k, v in param_dict.items():
                final_param_dict['gpt2.gpt2.' + k] = param_dict[k]
            # set the weights of final linear weights to weights of gpt2 token embedding
            final_param_dict['gpt2.dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
            load_param_into_net(gpt2_loss, final_param_dict)
            print("load pretrained parameter successfully!\n")
        
        elif eval_type == "finetuned":
            load_param_into_net(gpt2_loss, param_dict)
            print("load finetuned parameter successfully!\n")
            
        model = Model(gpt2_loss)
        
        # sample = Sample(decoder = model,model_config=gpt2_net_cfg,tokenizer=tokenizer,topk_num=1,topp_prob=1,return_ids=True)
        columns_list = ["input_ids", "input_mask", "label_ids"]
        print("============= Testing LAMBADA ACC =============")
        cnt  = 0
        for data in dataset.create_dict_iterator():
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data
            print("===========LAMBADA ACC DATA NUM:{}===========".format(cnt))
            # print("input_ids_shape: {}".format(input_ids.shape))
            # print("input_mask_shape: {}".format(input_mask.shape))
            # print("label_ids_shape: {}".format(label_ids.shape))
            
            logits = model.predict(input_ids, input_mask)
            # print("="*40)
            # print("after predict logits shape:",logits.shape)         (8,1024,50257)
            # output_str = sample.generate_for_LAMBADA(input_ids = input_ids,logits = logits, max_generate_length=3, max_iterations=200)
            
            output_str = generate_for_LAMBADA_numpy_topk(decoder=model,input_ids = input_ids, 
                                            logits = logits, tokenizer=tokenizer, max_iterations=200,
                                            generate_length_dynamically=generate_length_dynamically,
                                            stop_word_file="src/utils/pretrain-data/stopwords.txt")
                                                    
            label_str = get_wholeword_label_str(input_ids=input_ids,config=gpt2_net_cfg,tokenizer=tokenizer)
            
            # print("logits shape: {}".format(logits.shape))
            # print("logits: \n{}".format(logits))
            # print("===================================")
            # print("==============================================")
            # print("output_str:{}".format(output_str[0]))
            # print("label_str:{}".format(label_str[0].strip()))
            callback.update(output_str, label_str)
            eval_result_print(metric, callback)
            # callback.update(logits, label_ids)
            print("==============================================\n")  
            cnt += 1
        print("=============== Final score ==================")
        eval_result_print(metric, callback)
        print("************** Testing Finished **************")


    elif metric.lower() == "ppl":

        print("Prepare to calculate the ppl score ...")
        # ppl metric can be calculated by using the loss, so the difference is 'is_training'
        gpt2_loss = GPT2Lambada(config=gpt2_net_cfg,
                                is_training=False,
                                use_one_hot_embeddings=False)

        gpt2_loss.set_train(False)
        model = Model(gpt2_loss)

        param_dict = load_checkpoint(load_checkpoint_path)

        if eval_type == "zero-shot":
            final_param_dict = {}
            for k, v in param_dict.items():
                final_param_dict['gpt2.gpt2.' + k] = param_dict[k]
            # set the weights of final linear weights to weights of gpt2 token embedding
            final_param_dict['gpt2.dense1.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
            load_param_into_net(gpt2_loss, final_param_dict)
            print("load pretrained parameter successfully!\n")

        elif eval_type == "finetuned":
            load_param_into_net(gpt2_loss, param_dict)
            print("load finetuned parameter successfully!\n")

        columns_list = ["input_ids", "input_mask", "label_ids"]
        num_data = 0
        total_ppl = 0.0
        total_loss = 0.0

        print("================= Testing LAMBADA PPL =================")
        for data in dataset.create_dict_iterator():

            print("=========== LAMBADA PPL Test iteration:{}==========".format(num_data))
            input_data = []
            for i in columns_list:
                input_data.append(data[i])
            input_ids, input_mask, label_ids = input_data

            print("input_ids_shape: {}".format(input_ids.shape))
            print("input_mask_shape: {}".format(input_mask.shape))
            print("label_ids_shape: {}".format(label_ids.shape))

            logits = model.predict(input_ids, input_mask)  # (batch_size,seq_len,vocab_size)

            # print("*"*30)

            last_word_range_ = get_lastword_range(input_ids=input_ids, config=gpt2_net_cfg,tokenizer=tokenizer)  # [(left_pos,right_pos)]
            
            last_word_range = (last_word_range_[0][0] + 1, last_word_range_[0][1] + 1)
            last_word_logits_start_pos = last_word_range[0] - 1
            last_word_logits_end_pos = last_word_range[1] - 1

            # last_word_token_len = last_word_range[1] - last_word_range[0]
            # print(" | Last word token length:", last_word_token_len)

            # print(last_word_ids)

            # last_word_ids = P.Reshape()(last_word_ids,(-1,)).asnumpy().tolist()

            # print(last_word_ids)

            label_ids = extract_last_word_input_ids(input_ids=input_ids,seq_pos=last_word_range)  # (batch_size=1,x=lastword token num)
            label_input_mask = extract_last_word_input_ids(input_ids=input_mask,seq_pos=last_word_range)
            
            gold_logits = logits[::, last_word_logits_start_pos:last_word_logits_end_pos:1, ::]
            label_ids = P.Reshape()(label_ids, (-1,))  # (x,)

            gold_logits = P.Reshape()(gold_logits, (-1, gpt2_net_cfg.vocab_size))

            label_word_ids = label_ids.asnumpy().tolist()
            label_word = tokenizer.decode(label_word_ids)
            print("label word: ", label_word)

            # generate_word = tokenizer.decode([generate_ids])

            # print("generate word:", generate_word)

            
            # cross_entropy = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
            
            # cross_entropy = CrossEntropyCalculationWithMask(is_training=True, num_labels=tokenizer.vocab_size, config=gpt2_net_cfg)
            # loss = cross_entropy(gold_logits,label_ids,label_input_mask)

            # calculate cross entropy with numpy
            loss = cross_entropy_np(gold_logits.asnumpy(),label_ids.asnumpy())

            #print(" | after SoftmaxCrossEntropyWithLogits....")
            # loss = cross_entropy(gold_logits, label_ids)
            # print(" | after cross entropy...")
            # loss = model.predict(input_ids, input_mask, label_ids)

            # loss = loss.asnumpy()

            print(" | Loss: {:.6f}".format(float(loss)))

            num_data += 1
            total_loss += loss
            avg_loss = total_loss / num_data

            print(" | Current AVG loss:", avg_loss)
            print(" | Current AVG ppl:", math.exp(avg_loss))


        ppl = math.exp(avg_loss)
        # avg_ppl = total_loss / num_data
        print("-----------------------------------------")
        print(" PPL: {:.6f}".format(ppl))
        print("************** Testing Finished **************")

    else:

        raise ValueError("metric method not supported, support: [accuracy, ppl]")



def run_lambada():
    """
    run Language Modeling task
    """
    parser = argparse.ArgumentParser(description="Finetune and Evaluate languagemodel")
    parser.add_argument("--device_target", type=str, default="Ascend",
                        help="Device type. Default: Ascend.") 
    parser.add_argument("--device_id", type=int, default=2,
                        help="ID of target device. ")
    parser.add_argument("--metric_method", type=str, default="PPL",
                        help="The eval method including [Accuracy, PPL]. Default: Accuracy.") 
    parser.add_argument("--do_train", type=str, default="false",
                        help="Enable train. Default: false.")
    parser.add_argument("--do_eval", type=str, default="false",
                        help="Enable evaluation. Default: false.")
    parser.add_argument("--eval_type", type=str, default="zero-shot",
                        help="The type of evaluation including [zero-shot, finetuned]. Default: zero-shot.")
    parser.add_argument("--epoch_num", type=int, default=3,
                        help="Epoch number. Default: 1.")
    parser.add_argument("--train_data_shuffle", type=str, default="false",
                        help="Enable train data shuffle. Default: true.")
    parser.add_argument("--eval_data_shuffle", type=str, default="false",
                        help="Enable eval data shuffle. Default: false.")
                        
    parser.add_argument("--generate_length_dynamically", type=str, default="true",
                        help="Enable generate_length_Dynamically. Default: true.")
    parser.add_argument("--save_finetune_ckpt_path", type=str, default="/data/tju/pretrained-weight/lambada_saved/",
                        help="Save the checkpoint path.")
                        
    ## modify
    parser.add_argument("--load_pretrain_ckpt_path", type=str, default="/data/tju/pretrained-weight/mindspore_model_small.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--load_finetune_ckpt_path", type=str, default="/data/tju/pretrained-weight/mindspore_model_medium.ckpt",
                        help="Load the checkpoint file path.")
    parser.add_argument("--train_data_file_path", type=str, default="/data/tju/mindspore-dataset/lambada-development-mindrecord",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="/data/tju/mindspore-dataset/lambada-control-test-deep-mindrecord",
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
    if device == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
        context.set_auto_parallel_context(parallel_mode="stand_alone")
        print(" | Device: {}  | Device id: {}".format(device, args_opt.device_id))
    else:
        raise Exception("Device target error, Ascend is supported.")

    gpt2_loss = GPT2Lambada(config=gpt2_net_cfg,
                       is_training=True,
                       use_one_hot_embeddings=False)

    if args_opt.do_train.lower() == "true":
        print("==============    Start Loading Train Dataset   ============")
        print(" | Train Dataset: {}".format(args_opt.train_data_file_path))
        print(" | Checkpoint: {}".format(args_opt.load_pretrain_ckpt_path))
        train_dataset = create_language_model_dataset(do_shuffle=(args_opt.train_data_shuffle.lower() == "true"),
                                                      dataset_path=args_opt.train_data_file_path)
        do_train(train_dataset, gpt2_loss, load_pretrain_ckpt_path, save_finetune_ckpt_path, epoch_num)

    if args_opt.do_eval.lower() == "true":
        print("============== Start Loading Evaluation Dataset ============")
        print(" | Eval Dataset: {}".format(args_opt.eval_data_file_path))
        print(" | Checkpoint: {}".format(args_opt.load_finetune_ckpt_path))
        eval_dataset = create_language_model_dataset(do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"),
                                                     dataset_path=args_opt.eval_data_file_path)
        do_eval(eval_dataset, GPT2Lambada, metric, load_finetune_ckpt_path, args_opt.eval_type, args_opt.generate_length_dynamically)


if __name__ == "__main__":
    run_lambada()