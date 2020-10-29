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
    
    Args:
        string_list: list(str), list of text in form of str
    
    Returns:
        list,list
        the list of text with last word removed and the last word text list

    """
    return [ ' '.join(s.split()[:-1]) for s in string_list],[ s.split()[-1:][0] for s in string_list]

def _get_lastword_range(prefix,stringlist,tokenizer=None):
    """
    Get the range of lastword tokenized index in label_ids

    Args:
        prefix: list(str), list of text with its last word removed(a.k.a. "prefix") in form of str
        stringlist: list(str), list of text, same as it is in split_by_last_word 
        tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization, optional
    
    Returns:
        lastword_range: list(tuple), start and end postion of last word of each text of stringlist that used in selecting tokenized 
        last word index in logits. lastword_logits --> logits[batch_index,start:end,::] 
    """
    if tokenizer is None:
        tokenizer = Tokenizer()
        print('[WARNING] parameter: tokenizer is missing in utils.lambada_utils.last_word_index, using Tokenizer() as default tokenizer')
    
    prefix_ids_len = [len(tokenizer.encode(prefix_str)) for prefix_str in prefix]
    full_ids_len = [len(tokenizer.encode(full_str)) for full_str in stringlist]
    

    lastword_range = [(prefix_length,full_length) for prefix_length,full_length in zip(prefix_ids_len,full_ids_len)] 
    return lastword_range

def create_lambada_mask(input_ids,config=None,tokenizer=None):
    """
    create whole word mask for the last word of text in lambada dataset.

    Args:

    input_ids: Tensor [batch_size,seq_length], tensor of tokenized index of input
    config: GPT2Config, config of GPT2 model, if not initiated, this function will create a MockConfig by params of input_ids, optional
    tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization, optional

    Return:

    mask: Tensor [batch_size,seq_length], tensor of whole-word masked tensor of last word
        example:
        ---input_ids-->   Tensor([[50256, 464, 34822, 6378, 11, 356, 821, 30780, 4980, 50256,... ,50256]]) ---extract_string_from_tensor--> ["The Milky Way, we're renegading"] 
        ---tokenizer.encode--> [[464, 34822, 6378, 11, 356, 821, 30780, 4980]]  ---whole-word mask-->  Tensor([[0, 0, 0, 0, 0, 0, 1, 1, 0, ..., 0]])

    """
    
    if config is None:
        config = MockConfig()
        config.batch_size = input_ids.shape[0]
        config.seq_length = input_ids.shape[1]

    string_list = extract_string_from_tensor(input_ids,mode='single',tokenizer=tokenizer,config=config)

    prefix, _ = split_by_last_word(string_list)

    lastword_range = _get_lastword_range(prefix,string_list,tokenizer)

    batch_size = config.batch_size
    seq_length = config.seq_length

    mask_np = np.zeros((batch_size,seq_length),dtype=float)
    
    for batch_idx in range(batch_size):
        left_pos = lastword_range[batch_idx][0]
        right_pos = lastword_range[batch_idx][1]
        mask_np[batch_idx][left_pos:right_pos] = 1.0
    mask = Tensor(mask_np,dtype=mstype.float32)

    return mask

def get_lastword_range(input_ids,config=None,tokenizer=None):
    """
    Get the range of lastword tokenized index in input_ids

    Args:
        input_ids: Tensor(batch_size,seq_length)
        config: GPT2Config, config of GPT2 model, if not initiated, this function will create a MockConfig by params of input_ids, optional
        tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization, optional
    
    Returns:
        lastword_range: list(tuple), start and end postion of last word of each text of stringlist that used in selecting tokenized 
        last word index in logits. lastword_logits --> logits[batch_index,start:end,::] 
    """
    if tokenizer is None:
        tokenizer = Tokenizer()
    if config is None:
        config = MockConfig()
        config.batch_size = input_ids.shape[0]
        config.seq_length = input_ids.shape[1]

    string_list = extract_string_from_tensor(input_ids,mode='single',tokenizer=tokenizer,config=config)
    prefix, _ = split_by_last_word(string_list)

    lastword_range = _get_lastword_range(prefix,string_list,tokenizer)

    return lastword_range


def calculate_lambada_loss(input_ids,logits,config=None,loss_net=None,tokenizer=None):
    """
    calculate loss value for lambada

    Args:
        input_ids: Tensor(batch_size,seq_length), indexs of input text
        logits: Tensor(batch_size,seq_length,vocab_size), distribution of each token in each batch of text
        config: GPT2Config, config of GPT2 model, if not initiated, this function will create a MockConfig by params of input_ids, optional
        loss_net: nn.Cell, a network to caluculate loss by taking a mask, if not initiated, it will use src.utils.CrossEntropyCalculationWithMask as default, optional
        tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization, optional
    Return:
        loss: float, cross entropy between last word of model's output and label given by lambada
    
    """
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
    return loss.asnumpy()[0]

def get_wholeword_pair(input_ids,logits,config=None,tokenizer=None):
    """
    get whole word str from input_ids and logits

    Args:
        input_ids: Tensor(batch_size,seq_length), indexs of input text
        logits: Tensor(batch_size,seq_length,vocab_size), distribution of each token in each batch of text
        config: GPT2Config, config of GPT2 model, if not initiated, this function will create a MockConfig by params of input_ids, optional
        tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization, optional

    Returns:
        output_str: [str], output of model, decoded from the maximum index of logits
        label_str: [str], lastword str given lambada as label

    """
    if tokenizer is None:
        tokenizer = Tokenizer()
    if config is None:
        config = MockConfig()
        config.batch_size = input_ids.shape[0]
        config.seq_length = input_ids.shape[1]
        config.vocab_size = tokenizer.vocab_size
    
    #initiate operators
    argmax = P.Argmax(axis=-1)
    reshape = P.Reshape()

    #lastword_range is a list of tuples, seems like [...,(start_position_i,end_position_i),...]
    lastword_range = get_lastword_range(input_ids,config)

    #input_ids requires to shift right for one step for its every first token is <BOS> 
    ids = input_ids[::,1:].asnumpy()
    #(batch_size,seq_length,vocab_size) --reshape--> (batch_size*seq_length,vocab_size) --argmax--> (batch_size*seq_length)
    logits_argmax = argmax(reshape(logits,(config.batch_size*config.seq_length,-1)))
    #(batch_size*seq_length) --reshape--> (batch_size,seq_length), get index with max prob for each token in output logits
    logits_argmax = reshape(logits_argmax,(config.batch_size,config.seq_length))
    #convert indexed logits to numpy
    logits_idx = logits_argmax.asnumpy()

    #filter out index of last word in output and label
    output_ids = [ logit[index[0]:index[1]].tolist() for index,logit in zip(lastword_range,logits_idx)]
    label_ids = [ id_[index[0]:index[1]].tolist() for index,id_ in zip(lastword_range,ids)]

    #use GPT2Tokenizer to decode them into list of str
    output_str = [ tokenizer.decode(output_id) for output_id in output_ids ]
    label_str = [ tokenizer.decode(label_id) for label_id in label_ids ]

    return output_str,label_str
    
class MockConfig:
    def __init__(self):
        pass

if __name__=='__main__':
    from mindspore import context
    context.set_context(device_target="GPU", device_id=2)
    s=['I am good.','She is mine']
    print(split_by_last_word(s))
    x,y = split_by_last_word(s)
    print(_get_lastword_range(x,y))
    t = Tokenizer()
    print([ t.encode(sen) for sen in s])

    eos_id = t.eos_token_id
    
    pad = np.full((2,1024),eos_id,dtype=int)
    print(t.decode([464, 34822, 6378, 11, 356, 821,30780, 4980]),"  ",t.decode([999,1000,1111,38198,6433,21331,5112,2412]))
    pad[0][1:9] = [464, 34822, 6378, 11, 356, 821,30780, 4980]
    pad[1][1:9] = [999,1000,1111,38198,6433,21331,5112,2412]
    mock_logits = np.full((2,1024,t.vocab_size),float(1/t.vocab_size),dtype=float)
    mock_logits[0][7][4980] = 100
    mock_logits[0][6][30780] = 100
    mock_logits[1][6][5112] = 100
    mock_logits[1][7][2412] = 100
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
    print(loss,np.exp(loss))
    print(t.encode('The Milky Way, we\'re renegading'))
    print(t.decode([464, 34822, 6378, 11, 356, 821]))
    print(t.decode([30780, 4980]))

    logits = P.Cast()(logits,mstype.float32)
    output_word,label_word = get_wholeword_pair(input_ids,logits)
    print("get_wholeword_pair:")
    print(output_word,label_word)

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