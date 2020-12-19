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
    # return [ ' '.join(s.split()[:-1]) for s in string_list],[ s.split()[-1:][0] for s in string_list]
    return [ ' '.join(s.split()[:-1]) for s in string_list]

def _get_lastword_range(prefix, stringlist, tokenizer=None):
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
    
    prefix_ids_len = [len(tokenizer.encode(prefix_str))  for prefix_str in prefix] # +1 for including bos 
    full_ids_len = [len(tokenizer.encode(full_str))  for full_str in stringlist] # +1 for including bos 
    
    #lastword_range = [(prefix_length, full_length) for prefix_length, full_length in zip(prefix_ids_len, full_ids_len)] 
    lastword_range_ = [(prefix_length, full_length) for prefix_length, full_length in zip(prefix_ids_len, full_ids_len)]
    lastword_range = []
    for i in range(len(lastword_range_)):
        full_ids = tokenizer.encode(stringlist[i])
        last_prefix_id = tokenizer.encode(prefix[i])[-1]
        range_left = prefix_ids_len[i]
        for j in range(len(full_ids)-2,0,-1):
            if full_ids[j]== last_prefix_id:
                range_left = j+1
                break

        lastword_range.append((range_left,lastword_range_[i][1])) 
    
    return lastword_range

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
    # prefix, _ = split_by_last_word(string_list)
    prefix = split_by_last_word(string_list)

    lastword_range = _get_lastword_range(prefix,string_list,tokenizer)

    return lastword_range

def extract_logits(logits = None, seq_pos = None):
    """
    Args
        logits: Tensor(batch_size,seq_length,vocab_size) e.g.(8,1024,50257)
        seq_pos: list(batch_size)  

    Return:
        output_logits: Tensor(batch_size,1,vocab_size) extract the Specified logit according to the seq_pos list .
    """

    batch_size = logits.shape[0]
    for i in range(batch_size):

        logit = logits[i:i+1:1, seq_pos[i]:seq_pos[i]+1:1, ::]
        # print("extract_logits logit shape: {}".format(logit.shape))
        if i == 0 :
            output_logits = logit
        else:
            output_logits = P.Concat()((output_logits, logit))

    # print("final logits:",output_logits)
    
    return output_logits


def get_wholeword_label_str(input_ids,config=None,tokenizer=None):
    """
    get whole word label_str from input_ids 
    Args:
        input_ids: Tensor(batch_size,seq_length), indexs of input text
        config: GPT2Config, config of GPT2 model, if not initiated, this function will create a MockConfig by params of input_ids, optional
        tokenizer: GPT2Tokenizer, if not initiated, it will be created using the default setting in utils.tokenization, optional
    Returns:
        label_str: [str], lastword str given lambada as label
    """
    if tokenizer is None:
        tokenizer = Tokenizer()
    if config is None:
        config = MockConfig()
        config.batch_size = input_ids.shape[0]
        config.seq_length = input_ids.shape[1]
        config.vocab_size = tokenizer.vocab_size

    #lastword_range is a list of tuples, seems like [...,(start_position_i,end_position_i),...]
    lastword_range = get_lastword_range(input_ids,config,tokenizer=tokenizer)

    #input_ids requires to shift right for one step for its every first token is <BOS> 
    ids = input_ids[::,1:].asnumpy()

    label_ids = [ id_[index[0]:index[1]].tolist() for index,id_ in zip(lastword_range,ids)]
    
    # use GPT2Tokenizer to decode
    label_str = [ tokenizer.decode(label_id) for label_id in label_ids ]

    return label_str

class MockConfig:
    def __init__(self):
        pass

if __name__ == "__main__":
    pass