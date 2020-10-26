import mindspore
# import mindspore.common.tensor as Tensor
from mindspore.common.tensor import Tensor
import mindspore.ops.operations as P
import numpy as np
from src.finetune_eval_config import gpt2_net_cfg

def extract_logits_for_lambada(logits=None, label_ids=None, input_mask=None):
    # equalcount = P.EqualCount()
    all_one = Tensor(np.ones(gpt2_net_cfg.seq_length), mindspore.int32)
    no_mask_length = []
    for i in range(input_mask.shape[0]):
        input_mask_row = input_mask[i, ::]
        valid_length = int(P.EqualCount()(input_mask_row, all_one).asnumpy()[0])
        # print("valid_length type is {}, length is {}".format(type(valid_length), valid_length)) # count the no padding token number 
        no_mask_length.append(valid_length)

        logit = logits[i:i+1:1, valid_length-3:valid_length-2:1, ::]
        label = label_ids[i:i+1:1, valid_length-2:valid_length-1:1]        
        # print("extract_logits logit shape: {}".format(logit.shape))
        if i == 0 :
            output_logits = logit
            final_label_ids = label
        else:
            output_logits = P.Concat()((output_logits, logit))
            final_label_ids = P.Concat()((final_label_ids, label))


    print("output_logits shape : {}".format(output_logits.shape))
    print("output logits:\n{}".format(output_logits[:3,:,7:15]))
    # print("final_label_ids shape : {}".format(final_label_ids.shape))
    # print("output final_label_ids:\n{}".format(final_label_ids))

    return output_logits, final_label_ids, no_mask_length