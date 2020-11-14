from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from typing import TypeVar, Union
from .tokenization import Tokenizer
import numpy as np

def extract_string_from_tensor(input_ids: Tensor, mode="single",config = None, tokenizer = None):
        """
        Args:
            input_ids(Tensor): input tensor of sequence index. Shape: (self.batchsize,self.seq_length)
            mode(str):   "pair","single"and "CBT", "pair" for tasks with paired inputs, such as Text Summarization,
                    single output tasks with single input such as Language Modeling, "CBT" for the Children Book Test dataset, which has
                    3 componets of input (Leading text, Multiple choice to fill, Rest text).
        Return:
            prompt_list, list of prompt_text
            reference_list, list of reference_text, or second part of text
            rest_list , list of rest_text, or rest part of text
                Example:
                for pair mode,  it will return prompt_list, reference_list
        """

        assert config is not None, 'There is no GPT2-config, the configuration is compulsory parameter'

        if tokenizer is None:
            tokenizer = Tokenizer()
            print('[WARNING] parameter: tokenizer is missing in utils.tensor_manipulations.extract_string_from_tensor, using Tokenizer() as default tokenizer')

        reshape = P.Reshape()
        batch_size = config.batch_size
        seq_length = config.seq_length
        prompt_list = [""]*batch_size
        reference_list = [""]*batch_size
        rest_list = [""]*batch_size
        eos_text = tokenizer.eos_token
        len_eos_text = len(eos_text)
        input_ids = reshape(input_ids, (batch_size, seq_length))

        # For datasets with paired inputs, such as Text Summarization(Article, Summary), QA(Question, Answer).
        if mode == "pair":

            for batch_idx in range(batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = tokenizer.decode(sentence_list)
                prompt_start = 0
                prompt_end = sentence.find(eos_text, 0)
                reference_start = prompt_end+len_eos_text
                reference_end = sentence[reference_start:].find(
                    eos_text, 0)+reference_start
                prompt_list[batch_idx]=sentence[prompt_start:prompt_end]
                reference_list[batch_idx]=sentence[reference_start:reference_end]

            return prompt_list, reference_list

        # For single output datasets such as WikiText, etc.
        elif mode == "single":
            for batch_idx in range(batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = tokenizer.decode(sentence_list)
                prompt_start = 0
                prompt_end = sentence.find(eos_text, 0)
                prompt_list[batch_idx]=sentence[prompt_start:prompt_end]
            
            return prompt_list
        
        # For CBT dataset
        elif mode == "CBT":
            for batch_idx in range(batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = tokenizer.decode(sentence_list)
                prompt_start = 0
                prompt_end = sentence.find(eos_text, 0)
                reference_start = prompt_end+len_eos_text
                reference_end = sentence[reference_start:].find(
                    eos_text, 0)+reference_start
                rest_start = reference_end+len_eos_text
                rest_end = sentence[rest_start:].find(eos_text, 0)+rest_start

                prompt_list[batch_idx]=sentence[prompt_start:prompt_end]
                reference_list[batch_idx] = sentence[reference_start:reference_end]
                rest_list[batch_idx]=sentence[rest_start:rest_end]
            
            #return string lists splited from input_ids
            return prompt_list, reference_list, rest_list

        
        else:
            raise NotImplementedError('mode:{} not supported.'.format(mode))



def tensorize_ids_with_masks(src_str,config=None,tokenizer=None,add_special_tokens=False,append_eos=False):
        """
        Transform from string to tensor

        Args:
            src_str: string or list of strings
        Return:
            input_ids: Tensor(self.batch_size, self.seq_length)
            input_mask: Tensor(self.batch_size, self.seq_length)
            src_len: length of tokens of src_string after decoded by self.tokenzier
        """
        if type(src_str)==str:
            src_str = [src_str]
        
        assert config is not None, 'There is no GPT2-config, the configuration is compulsory parameter'

        if tokenizer is None:
            tokenizer = Tokenizer()
            print('[WARNING] parameter: tokenizer is missing in utils.tensor_manipulations.tensorize_ids_with_masks, using Tokenizer() as default tokenizer')


        batch_size = config.batch_size
        seq_length = config.seq_length
        reshape = P.Reshape()
        concat = P.Concat()

        input_shape = (batch_size, seq_length)
        single_sentence_shape = (1,seq_length)
        src_len_list = list()
        input_ids = None
        input_mask = None
        for batch_idx in range(batch_size):
            src_list=tokenizer.encode(src_str[batch_idx])
            #src_list = self.tokenizer.encode(src_str)
            src_len = len(src_list)
            if src_len > seq_length:
                src_list = src_list[:seq_length]
                src_len = seq_length
            
            #append_eos
            if append_eos is True:
                src_len += 1
                src_list.append(tokenizer.eos_token_id)

            src_len_list.append(src_len)
            ret_dict = tokenizer.prepare_for_model(
            src_list, max_length=config.seq_length, add_special_tokens=add_special_tokens)

            input_ids_list = ret_dict['input_ids']
            input_mask_list = ret_dict['attention_mask']

            input_ids_tensor = reshape(
            Tensor(np.array(input_ids_list, dtype=int), dtype=mstype.int32), single_sentence_shape)
            input_mask_tensor = reshape(
            Tensor(np.array(input_mask_list, dtype=int), dtype=mstype.int32), single_sentence_shape)
            if batch_idx == 0:
                input_ids = input_ids_tensor
                input_mask = input_mask_tensor
            else:
                input_ids = concat((input_ids,input_ids_tensor))
                input_mask = concat((input_mask,input_mask_tensor))
            

        return input_ids, input_mask, src_len_list


def extract_single_token_logits(logits = None, seq_pos = None):
    """
    Args
        logits: (batch_size,seq_length,vocab_size) e.g. when batchsize is 8, sequence length is 1024 and vocab_size is 50257,
        then logits is a Tensor with shape (8,1024,50257)
        seq_pos:(batch_size) list 

    Return:
        output_logits: (batch_size,1,vocab_size) extract the logit to predict the last token.
    """

    batch_size = logits.shape[0]
    for i in range(batch_size):
        logit = logits[i:i+1:1, seq_pos[i]:seq_pos[i]+1:1, ::]
        if i == 0 :
            output_logits = logit
        else:
            output_logits = P.Concat()((output_logits, logit))

    return output_logits

def get_last_one_pos(input_mask:Tensor):
    """
    Arg:
        input_mask (Tensor): (batch_size,seq_length)
    Return:
        pos (Tensor): (batch_size,)
    """
    pos = P.ReduceSum(keep_dims=False)(input_mask,axis=1) #(batch_size,)
    pos = pos -1
    return pos

def get_next_one_pos(input_mask:Tensor):
    """
    Arg:
        input_mask (Tensor): (batch_size,seq_length)
    """
    pos = P.ReduceSum(keep_dims=False)(input_mask,axis=1) #(batch_size,)
    return pos

def add_last_token_mask(input_mask:Tensor,overflow_strategy:str="shift"):
    pos = get_next_one_pos(input_mask).asnumpy()
    input_mask_np = input_mask.asnumpy()
    maximum_length = input_mask.shape[1]
    batch_size = input_mask.shape[0]
    for idx in range(batch_size):
        #not overflow
        if pos[idx] < maximum_length:
            input_mask_np[idx][pos[idx]] = 1

        #overflow
        else:
            if overflow_strategy == "shift":
                continue
            if overflow_strategy == "truncate":
                continue
            else:
                raise ValueError("{} not an option in ['shift','truncate'].".format(overflow_strategy))
    return Tensor(input_mask_np,dtype=mstype.int32)
    