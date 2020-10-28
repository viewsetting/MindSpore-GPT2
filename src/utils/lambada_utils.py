from mindspore import Tensor
from tensor_manipulations import extract_string_from_tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from CrossEntropy import CrossEntropyCalculationWithMask
from typing import TypeVar, Union
from tokenization import Tokenizer
import numpy as np

def split_by_last_word(string_list):
    """
    split list of strings list by last word 
    """
    return [ ' '.join(s.split()[:-1]) for s in string_list],[ s.split()[-1:][0] for s in string_list]

def get_lastword_range(prefix,stringlist,tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
        print('[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer')
    
    prefix_ids_len = [len(tokenizer.encode(prefix_str)) for prefix_str in prefix]
    full_ids_len = [len(tokenizer.encode(full_str)) for full_str in stringlist]
    

    lastword_range = [(prefix_length,full_length) for prefix_length,full_length in zip(prefix_ids_len,full_ids_len)] 
    return lastword_range

def create_lambada_mask(input_ids,config=None,tokenizer=None):
    #assert config is not None,'GPT2_config should not be None'
    if config is None:
        config = MockConfig()
        config.batch_size = input_ids.shape[0]
        config.seq_length = input_ids.shape[1]

    string_list = extract_string_from_tensor(input_ids,mode='single',tokenizer=tokenizer,config=config)
    #print(string_list)
    prefix, _ = split_by_last_word(string_list)
    #print(prefix)
    lastword_range = get_lastword_range(prefix,string_list,tokenizer)

    batch_size = config.batch_size
    seq_length = config.seq_length

    mask_np = np.zeros((batch_size,seq_length),dtype=float)
    
    for batch_idx in range(batch_size):
        left_pos = lastword_range[batch_idx][0]
        right_pos = lastword_range[batch_idx][1]
        mask_np[batch_idx][left_pos:right_pos] = 1.0
    mask = Tensor(mask_np,dtype=mstype.float32)

    return mask
    
def calculate_lambada_loss(input_ids,logits,config=None,loss_net=None,tokenizer=None):
    if tokenizer is None:
        tokenizer = Tokenizer()
    if config is None:
        config = MockConfig()
        config.batch_size = input_ids.shape[0]
        config.seq_length = input_ids.shape[1]
        config.vocab_size = tokenizer.vocab_size
    if loss_net is None:
        loss_net = CrossEntropyCalculationWithMask(is_training=True,num_labels=config.vocab_size,config=config)


    reshape = P.Reshape()
    label_ids = input_ids[::,1:]
    logits = reshape(logits[::,:config.seq_length-1,::],(config.batch_size*(config.seq_length-1),config.vocab_size))
    lambada_mask = create_lambada_mask(input_ids,config,tokenizer)

    loss = loss_net(logits,label_ids,lambada_mask[::,:config.seq_length-1])
    return loss
    
    
class MockConfig:
    def __init__(self):
        pass

if __name__=='__main__':
    from mindspore import context
    context.set_context(device_target="GPU", device_id=2)
    s=['I am good.','She is mine']
    print(split_by_last_word(s))
    x,y = split_by_last_word(s)
    print(get_lastword_range(x,y))
    t = Tokenizer()
    print([ t.encode(sen) for sen in s])

    eos_id = t.eos_token_id
    
    pad = np.full((2,1024),eos_id,dtype=int)
    pad[0][1:9] = [1450,1112,2133,17809,3232,3214,31243,4214]
    pad[1][1:9] = [999,1000,1111,38198,6433,21331,5112,2412]
    mock_logits = np.full((2,1024,t.vocab_size),float(1/t.vocab_size),dtype=float)
    mock_logits[0][7][4214] = 100
    # mock_logits[1][6][5112] = 100
    # mock_logits[1][7][2412] = 100
    softmax = P.LogSoftmax(axis=-1)
    

    input_ids = Tensor(pad,dtype=mstype.int32)
    logits = Tensor(mock_logits,dtype=mstype.float32)
    logits = softmax(logits)
    print(logits[0][7][4210:4220])
    mask = create_lambada_mask(input_ids)
    #print(mask)
    print(mask[::,:10])
    print(mask.shape)
    print(t.decode([4214]))
    print(t.decode([5112,2412]))
    loss = calculate_lambada_loss(input_ids,logits)
    print(loss,np.exp(loss.asnumpy()[0]))

""" mock_logits with exactly matching

root@75843d3f0980:/gpt2/src/utils# python lambada_utils.py 
(['I am', 'She is'], ['good.', 'mine'])
[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer
[(2, 2), (2, 1)]
[[40, 716, 922, 13], [3347, 318, 6164]]
[-99.99998 -99.99998 -99.99998 -99.99998   0.      -99.99998 -99.99998
 -99.99998 -99.99998 -99.99998]
[WARNING] parameter: tokenizer is missing in utils.tensor_manipulations.extract_string_from_tensor, using Tokenizer() as default tokenizer
[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]]
(2, 1024)
 regist
 equipmentends
[0.] 1.0
"""

""" mock_logits with even distribution(max loss)

root@75843d3f0980:/gpt2/src/utils# python lambada_utils.py 
(['I am', 'She is'], ['good.', 'mine'])
[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer
[(2, 2), (2, 1)]
[[40, 716, 922, 13], [3347, 318, 6164]]
[-10.824905 -10.824905 -10.824905 -10.824905 -10.824905 -10.824905
 -10.824905 -10.824905 -10.824905 -10.824905]
[WARNING] parameter: tokenizer is missing in utils.tensor_manipulations.extract_string_from_tensor, using Tokenizer() as default tokenizer
[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]]
(2, 1024)
 regist
 equipmentends
[10.824869] 50255.19


"""

"""mock_logits with batch 0 fully matched
root@75843d3f0980:/gpt2/src/utils# python lambada_utils.py 
(['I am', 'She is'], ['good.', 'mine'])
[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer
[(2, 2), (2, 1)]
[[40, 716, 922, 13], [3347, 318, 6164]]
[-99.99998 -99.99998 -99.99998 -99.99998   0.      -99.99998 -99.99998
 -99.99998 -99.99998 -99.99998]
[WARNING] parameter: tokenizer is missing in utils.tensor_manipulations.extract_string_from_tensor, using Tokenizer() as default tokenizer
[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer
[[0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 0. 1. 1. 0. 0.]]
(2, 1024)
 regist
 equipmentends
[7.2165794] 1361.8229

"""