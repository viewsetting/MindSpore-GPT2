""" For Beam Search and Nucleus Sampling etc. """
import numpy as np
from scipy.special import softmax
from typing import TypeVar, Union, Optional
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Model, Parameter
from mindspore import dtype as mstype
from .tensor_manipulations import extract_single_token_logits,tensorize_ids_with_masks,add_last_token_mask,get_next_one_pos,add_last_token
from mindspore.context import get_context
from .tokenization import GPT2Tokenizer
import json

INF = 1. * 1e9
class TopKTopP_Filter():
    """
    Top K sampling along with Top P sampling(Nucleus Sampling)
    
    Choose top-K probability of ids and those with top-P probability ids into candidate sample sets.
    Use np.random.multinomial to sample

    Args:
        batch_size and vocab_size of model, (int).
        k (int): Parameter for Top-K sampling, k should be in range of [0,vocab_size]. 0 for no filter for TopK sampling(do nothing). Default: 0.
        p (float): Parameter for Top-P sampling a.k.a. Necleus Sampling, p is in between 0.0 and 1.0. Default: 1.0, Optional.
        temperature: param for generation, greater if generation more diverse. Default: 1.0, Optional.
        min_tokens_to_keep: a number for a guareented generation. Default: 1, Optional.
        fp16: True for open float16 optimization. Defalut: False, Optional.

    """

    def __init__(self,
                 batch_size:int,
                 vocab_size:int,
                 k:int=0,
                 p:float=1.0,
                 temperature:float=1.0,
                 min_tokens_to_keep:int=1,
                 ):

        self.k = k
        self.p = p
        self.temp = temperature

        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.min_tokens_to_keep = min_tokens_to_keep
        
        assert self.temp > 0.0, 'temperature must be positive'
        assert self.k >= 0, 'the top_k number must be no negative.'
        if self.k > 0:
            assert self.min_tokens_to_keep <= self.k, 'K must be larger than or equal to min_token_to_keep for top p sampling'

        if self.k ==0:
            self.k = self.vocab_size

        self.safety_mask = np.concatenate((np.ones((self.batch_size,self.min_tokens_to_keep)),
                                         np.zeros((self.batch_size,self.k-self.min_tokens_to_keep))),
                                         axis = 1
                                         ).astype(np.bool)

    def calculate(self, distribution):
        """
        caclulate sampling procesure with setting initialied before, return a list of sampled ids.

        Inputs:
        distribution(numpy.ndarray): with shape (batch_size,vocab_size)
    
        Returns:
            sampled ids: a list, with length: batch_size

        """
        
        if self.temp != 1.0:
            distribution = distribution / float(self.temp)

        distribution_sorted = -np.sort(-distribution,axis=1)
        index_sorted = np.argsort(-distribution,axis=1)

        # if self.k == 0, topk_distribution will choose full of distribution_sorted
        topk_distribution = distribution_sorted[::,:self.k if self.k > 0 else self.vocab_size]
        topk_indices = index_sorted[::,:self.k if self.k > 0 else self.vocab_size]
        #topk_distribution = softmax(topk_distribution,axis=1)

        # safety check of probability
        self.p = max(0.0,min(1.0,self.p))
        cum_sum = np.cumsum(softmax(topk_distribution,axis=1),axis=1)
        bool_map = np.logical_or((cum_sum <= self.p),self.safety_mask).astype(np.float32)
        
        topk_distribution = topk_distribution * bool_map + np.float32(-1e5)*(1.0-bool_map)


        topk_distribution = softmax(topk_distribution,axis=1)

        # normalize for np.float64
        # choose np.float64 to avoid overflow in softmax operation
        topk_distribution = topk_distribution.astype(np.float64)
        for batch_idx in range(self.batch_size):
            topk_distribution[batch_idx] = topk_distribution[batch_idx] / np.sum(topk_distribution[batch_idx])

        ret_ids = []

        for batch_idx in range(self.batch_size):
            select_index = np.argmax(np.random.multinomial(1,topk_distribution[batch_idx]))
            ret_ids.append(topk_indices[batch_idx][select_index])
            
        return ret_ids

class Sample():
    """
    Initiate a Sample object for sampling next token(s) from previous text.

    Args:
        decoder (Model): GPT2 model to do generation.
        model_config (GPT2Config): configuration of given GPT2 model.
        generate_length (int): length of generation, if it is initailized, self.generate() will generate
                               text based on it, unless a new length parameter is passed to self.generate().
        tokenizer (GPT2Tokenizer): if choose to use input_str parameter in self.generate(), a tokenizer is compulsory.
        topk_num (int): number of K in top-K Sampling, 0 for no condition constrained, tantamount to K = self.vocab_size. Default:0
        topp_prob (float): probability parameter of topp sampling if p = 1.0, then it equals to do nothing. (nucleus sampling). Defalut: 1.0
        temperature (float): temperature for topk sampling. Default: 1.0
        min_tokens_to_keep (int): guarantee for there is at least min_tokens_to_keep token(s) generated. Default:1
        early_stop (bool): whether stop when the model generates <EOS> token. It is functioned when batch_size is 1. Default: False
        demo_mode(bool): True if input_str is a str not a List of str. self.batch_size reqiures to be 1 if it is True. Default: False
        return_ids (bool): whether return ids generated from Sample. Default: False
        return_last_token_logits (bool): whether return logits of last token for each time step during generation. Default: False
        append_eos (bool): whether append <EOS> token id to input_ids pass directly to GPT2Model class. Default: False
    """

    def __init__(self,
                 decoder:Model,
                 model_config=None,
                 generate_length:int=1,
                 tokenizer:Optional[GPT2Tokenizer]=None,
                 topk_num:int=0,
                 topp_prob:float=1.0,
                 temperature:float=1.0,
                 min_tokens_to_keep:int=1,
                 early_stop:bool=False,
                 demo_mode:bool=False,
                 return_ids:bool=False,
                 return_last_token_logits:bool=False,
                 append_eos:bool=False):

       
        assert model_config is not None, 'Config is a must for sampling.'
        
        self.model_config = model_config
        self.topk_num = topk_num
        self.topp_prob = topp_prob
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.reshape = P.Reshape()
        self.cumsum = P.CumSum()
        self.onehot = P.OneHot()
        self.generate_length = generate_length
        self.seq_length = model_config.seq_length
        self.batch_size = model_config.batch_size
        self.vocab_size = model_config.vocab_size
        
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.concat = P.Concat()
        self.early_stop = early_stop
        self.demo_mode = demo_mode
        self.return_ids = return_ids
        self.return_last_token_logits = return_last_token_logits
        self.append_eos = append_eos
        self.device_target = get_context("device_target")

        #different choice of sample function for adjusting several device target types
        if self.device_target == "GPU":
            self.sample_function = P.Multinomial(seed=1)
        elif self.device_target == "Ascend":
            self.sample_function = P.RandomCategorical(mstype.int32)
        else:
            raise NotImplementedError("Device Target {} not supported.".format(self.device_target))

        self.filter_distribution = TopKTopP_Filter(self.batch_size,
                                                   self.vocab_size,
                                                   k=self.topk_num,
                                                   p=self.topp_prob,
                                                   temperature=self.temperature,
                                                   min_tokens_to_keep=self.min_tokens_to_keep)

        if self.tokenizer is not None:
            self.eos_id = self.tokenizer.eos_token_id
        else:
            self.eos_id = model_config.vocab_size-1

        if self.tokenizer is not None:
            self.eos_text = self.tokenizer.eos_token
        else:
            self.eos_text = "<|endoftext|>"

        if self.demo_mode is True:
            assert self.batch_size == 1, 'Demo mode requires batchsize euqals to 1, but get batch_size={}'.format(
                self.batch_size)

    def _extract_string_from_tensor(self, input_ids: Tensor,  mode="pair"):
        """
        Args:
            input_ids(Tensor): input tensor of sequence index. Shape: (self.batchsize,self.seq_length)
            mode (str): ["pair", "single"] 
                        "pair" for tasks with paired inputs `<bos> A <eos> B <eos>`, such as summarization, reading comprehension task.
                        "single" for tasks with single input `<bos> A <eos>`, such as Language Modeling task.
        Returns:
            source_list (list): the list of source_text or first part of text.
            target_list (list): the list of target_text or second part of text.

            If self.batch_size is 1, it will return the first sentence of list, that is to say, the string.
            
            Example:
                for pair mode, if self.demo_mode is True, it will return source_list[0], target_list[0]
        """
        assert self.tokenizer is not None, 'There is no tokenizer'
        source_list = [""] * self.batch_size
        target_list = [""] * self.batch_size
        eos_text = self.tokenizer.eos_token
        len_eos_text = len(eos_text)
        input_ids = self.reshape(input_ids, (self.batch_size, self.seq_length))

        # For datasets with paired inputs, such as Text Summarization(Article, Summary), QA(Question, Answer).
        if mode == "pair":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
                source_start = 0
                source_end = sentence.find(eos_text, 0)
                target_start = source_end+len_eos_text
                target_end = sentence[target_start:].find(eos_text, 0) + target_start
                source_list[batch_idx] = sentence[source_start:source_end]
                target_list[batch_idx] = sentence[target_start:target_end]

            if self.batch_size == 1 and self.demo_mode is True:
                return source_list[0], target_list[0]
            else:
                return source_list, target_list

        # For single output datasets such as WikiText, etc.
        elif mode == "single":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
                source_start = 0
                source_end = sentence.find(eos_text, 0)
                source_list[batch_idx] = sentence[source_start:source_end]
            if self.batch_size == 1 and self.demo_mode is True:
                return source_list[0]
            else:
                return source_list
        elif mode == "full":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()

                sentence = self.tokenizer.decode(sentence_list)
                source_start = 0
                source_end = sentence.find(eos_text, 0)
                source_list[batch_idx] = sentence[source_start:source_end]
            if self.batch_size == 1 and self.demo_mode is True:
                return source_list[0]
            else:
                return source_list
        
        else:
            raise ValueError('mode:{} not supported.'.format(mode))

    def _tensorize_ids_with_masks(self, src_str,append_eos_flag=False):
        """
        Transform from string to tensor

        Args:
            src_str: string or list of strings
        Return:
            input_ids (Tensor): shape with [self.batch_size, self.seq_length]
            input_mask (Tensor): shape with [self.batch_size, self.seq_length]
            src_len (int): the length of tokens of src_string after decoded by self.tokenzier
        """
        if type(src_str) == str:
            src_str = [src_str]

        input_shape = (self.batch_size, self.seq_length)
        single_sentence_shape = (1, self.seq_length)
        src_len_list = list()
        input_ids = None
        input_mask = None
        for batch_idx in range(self.batch_size):
            src_list = self.tokenizer.encode(src_str[batch_idx])
            # src_list = self.tokenizer.encode(src_str)
            src_len = len(src_list)
            if src_len > self.seq_length:
                src_list = src_list[:self.seq_length]
                src_len = self.seq_length

            #append_eos deprecated now.
            # if append_eos_flag is True:
            #     if src_len<self.seq_length:
            #         src_len += 1
            #         src_list.append(self.tokenizer.eos_token_id)
            #     else:
            #         src_list[self.seq_length-1] = self.tokenizer.eos_token_id

            src_len_list.append(src_len)
            ret_dict = self.tokenizer.prepare_for_model(src_list,
                                                        max_length=self.model_config.seq_length,
                                                        add_special_tokens=False)

            input_ids_list = ret_dict['input_ids']
            input_mask_list = ret_dict['attention_mask']

            input_ids_tensor = self.reshape(Tensor(np.array(input_ids_list, dtype=int), dtype=mstype.int32), 
                                            single_sentence_shape)
            input_mask_tensor = self.reshape(Tensor(np.array(input_mask_list, dtype=int), dtype=mstype.int32), 
                                             single_sentence_shape)
            if batch_idx == 0:
                input_ids = input_ids_tensor
                input_mask = input_mask_tensor
            else:
                input_ids = self.concat((input_ids, input_ids_tensor))
                input_mask = self.concat((input_mask, input_mask_tensor))

        return input_ids, input_mask, src_len_list

    def _get_real_word(self, select_word, real_word_index):
        """
        Get word index in UNSORTED(logits generated from GPT2Model without passing through a TopKTopPFilter) ids and logits.
        """

        # mindspore.ops.Gather is supported in MindSpore v.1.0 on Ascend
        if self.device_target == "Ascend" or self.device_target == "GPU":
            select_word_np = select_word.asnumpy()
            range_index = np.arange(0, self.batch_size)
            select_word_merge = np.array([[index, word] for index, word in zip(range_index, select_word_np)], dtype=int)
            word_index_2D = Tensor(select_word_merge, dtype=mstype.int32)
            real_selected_word_ids = P.GatherNd()( real_word_index,word_index_2D)
            #Tensor shape: (batch_size,)
        else:
            raise NotImplementedError('CPU and other backend types have not been supported yet')

        return real_selected_word_ids
    

    class last_token_pos():
        """
        class for record input_strs and the position of their last tokens 

        Args:
            input_ (Union): list if input is a list containing strs, Tensor with shape (batch_size,seq_length) representing input_mask
        """
        def __init__(self, input_:Union[list,Tensor],seq_length=1024):
            self.input_strs = input_ if type(input_) is list else None
            self.input_mask = input_ if type(input_) is not list else None
            self.seq_length = seq_length
            if self.input_strs is not None:
                self.pos_list = [ len(input_str)-1 for input_str in self.input_strs]
            else:
                #Tensor (batch_size,seq_length) --> list ,len(list) = batch_size
                input_mask_ = P.Cast()(self.input_mask,mstype.float32)
                temp_pos_list = P.ReduceSum(keep_dims=False)(input_mask_,axis=1).asnumpy().astype(np.int32).tolist()
                #minimum value is always 0 for safety 
                self.pos_list = [max(0,pos-1) for pos in temp_pos_list]
        
        def get_pos(self, shift:int = 0):
            #return last token if overflow
            shift_list = [min(self.seq_length-1,pos+shift) for pos in self.pos_list]
            return shift_list


    def _sample_from_distribution(self,distribution:Tensor):
        """
        sample one token per batch from self.sample_function(). sample function may varies due to different type of Device target.

        Arg:
            distribution (Tensor): (batch_size,vocab_length) distribution or logits of the last token of different batches.
        
        Return:
            word_index (Tensor): (batch_size,)

        """
        # reshape if Ascend
        if self.device_target == "Ascend":
            distribution = self.reshape(distribution, (self.vocab_size, self.batch_size))
            topk_distribution = distribution[:self.topk_num, ::]
            topk_distribution = self.reshape(topk_distribution, (self.batch_size, -1))
            
            word_index = self.sample_function(P.Softmax()(topk_distribution), 1 , 1)
            word_index = self.reshape(word_index,(-1,))
            
            # GPU
        elif self.device_target == "GPU":
            word_index = self.sample_function(P.Softmax()(distribution),1)

        else:
            raise ValueError("Device type {} not supported yet.".format(self.device_target))

        return word_index


    def generate_one_step(self,input_ids:Tensor,input_mask:Tensor,beam_size=1):
        """
        generate next token for only one step, use softmax to regularize logits

        Arg:
            input_ids (Tensor): (batch_size,seq_length) ids of input text
            input_mask (Tensor): (batch_size,seq_length) mask of input text, 0 for mask, 1 for reserved
            beam_size (int): int, beam_size for each candidate text
        
        Return:
            topk_indices (Tensor): (batch_size,beam_size), topk (k = beam_size) num of the next token indices for each batch 
            topk_logits (Tensor): (batch_size,beam_size), topk (k = beam_size) num of the next token logits(distribution) for each batch 
        """
        logits = self.decoder.predict(input_ids, input_mask)
        last_token_pos_recorder = self.last_token_pos(input_mask)
        last_token_pos_list = last_token_pos_recorder.get_pos(shift=0)
        return_last_logits = extract_single_token_logits(logits, last_token_pos_list) #(batch_size,1,vocab_size)
        return_last_logits = self.reshape(return_last_logits,(self.batch_size,self.vocab_size)) #(batch_size,vocab_size)
        return_last_logits = P.Softmax()(return_last_logits)
        topk_logits,topk_indices = P.TopK(sorted=True)(return_last_logits,beam_size) #(batch_size,beam_size)
        return topk_indices,topk_logits



    def generate(self, input_str=None,input_ids=None,input_mask=None, generate_length=None):
        """
        base function for text generation given a batch-size list of str or str itself (when demo mode is on)
        
        Args
            input_str ([str] or str): prompt string
            generate_length: number of tokens to generate
    
        Returns:
            generate_str: string generated by the model
            full_str: input_str appended with generate_str
        """

        #check for tokenizer if input_str is given
        if input_str is not None:
            assert self.tokenizer is not None, 'if choose to give input_str, a tokenizer is necessary.'
        
        #initiate generate_str for record generations for each batch
        #generate_str = [""] * self.batch_size

        #check if both tensor exists
        if input_ids is not None:
            assert input_mask is not None,'if input_ids is given, input_mask is required either.'
        
        #warning if all params are passed through
        if input_str is not None and input_ids is not None and input_mask is not None:
            print('[WARNING] Sample.generate got input_str, input_ids and input_mask, choose input_str as default for input')      
        
        #type check for demo_mode: 1 batch, input_str is not None and initiate full_str as input_str
        if self.batch_size == 1 and self.demo_mode == True:
            assert input_str is not None,"demo mode should have input str"
            # type check
            if type(input_str) is list:
                assert type(input_str[0]) is str,"type of input_str is {}, which should be str instead.".format(type(input_str[0]))
                if len(input_str) != 1:
                    print("[WARNING] Sample.generate: length of input_str is larger than 1, choose input_str[0] as input_str.")
                input_str = input_str[0]
            assert type(input_str) is str,"type of input_str is {}, which should be str instead.".format(type(input_str))
            input_str = [input_str]
        
        

        if generate_length is not None:
            #reload generate_length
            generate_length = int(generate_length)
            assert generate_length >= 0, 'generate_length can not be negative.'
        else:
            generate_length = self.generate_length

        return_ids_list = [[] for i in range(self.batch_size)]

        if input_ids is None and input_mask is None:
            input_ids, input_mask,_ = self._tensorize_ids_with_masks(input_str)
        else:
            if input_str is None:
                if input_ids is not None:
                    input_str = self._extract_string_from_tensor(input_ids,mode="full")
        last_token = self.last_token_pos(input_mask,seq_length=self.seq_length)


        for i in range(generate_length):
            #only first input_ids 
            #input_ids, input_mask, len_str = self._tensorize_ids_with_masks(full_str)
            #get index of last_token in iteration i for different batch may have different length 
            last_token_pos_list = last_token.get_pos(shift=i)
            early_stop_mask = [0] * self.batch_size
            
            
            #raw, unsorted logits(distribution) of next word
            logits = self.decoder.predict(input_ids, input_mask)

            if self.return_last_token_logits is True:
                if i == 0:
                    #[batch_size,1,vocab_size]
                    return_last_logits = extract_single_token_logits(logits, last_token_pos_list)
                else:
                    #[batch_size,1,vocab_size] + [batch_size,i,vocab_size] --> [batch_size,i+1,vocaab_size]
                    return_last_logits = P.Concat(axis=1)((return_last_logits,
                                                          extract_single_token_logits(logits, last_token_pos_list)))
            
            nextword_distribution = self.reshape(logits[0, last_token_pos_list[0]:last_token_pos_list[0]+1:1, ::], (1, -1))

            # stack up nextword_distribution if batch_size is larger than 1
            if self.batch_size > 1:
                for batch_idx in range(1, self.batch_size):
                        nextword_distribution_rest = self.reshape(
                            logits[batch_idx, last_token_pos_list[batch_idx]:last_token_pos_list[batch_idx]+1:1, ::], (1, -1))
                        nextword_distribution = self.concat((nextword_distribution, nextword_distribution_rest))

            # get sampled ids
            real_next_word_index_list = self.filter_distribution.calculate(nextword_distribution.asnumpy().astype(np.float32))

            append_ids = []

            # tokenizer.decode and early_stop (if all batched generates a EOS, then it is time to say goodbye)
            for batch_idx in range(self.batch_size):
                next_word_index = real_next_word_index_list[batch_idx]
                # earlystop if the model generates a EOS token.
                if next_word_index == self.eos_id and self.early_stop is True and self.batch_size == 1:
                    break
                if next_word_index == self.eos_id and self.early_stop is True:
                    early_stop_mask[batch_idx] = 1
                if early_stop_mask[batch_idx] == 1 and self.early_stop is True:
                    continue
                
                return_ids_list[batch_idx].append(next_word_index)
                append_ids.append(next_word_index)
            
            
            # check early_stop mask at the end of each loop
            if 0 not in early_stop_mask:
                break
            input_ids,input_mask = add_last_token(input_ids,input_mask,overflow_strategy="shift",append_ids=append_ids,next_token_pos=last_token.get_pos(shift=i+1))

        #add str to full str
        generate_str = ["" for _ in range(self.batch_size)]
        full_str = ["" for _ in range(self.batch_size)]
        text_cnt = 0
        for text_ids in return_ids_list:
            text = self.tokenizer.decode(text_ids)
            generate_str[text_cnt]=text
            text_cnt += 1
            
        for batch_idx in range(self.batch_size):
            full_str[batch_idx] = input_str[batch_idx] + generate_str[batch_idx]

        #return by several conditions
        if self.batch_size == 1 and self.demo_mode is True:
            if self.return_ids == True:
                return generate_str[0], input_str[0],return_ids_list[0]
            else:
                return generate_str[0], input_str[0]
        else:
            if self.return_ids == True:
                if self.return_last_token_logits == True:
                    return return_ids_list,return_last_logits
                return return_ids_list
            return generate_str, full_str
    
    

class BeamSearch():
    """
    Beam Search (Default setting)
    Args:
        decoder (Model): Model for decoding
        mdoel_config (GPT2Config): configuration of GPT2 decoder
        tokenizer (Tokenizer): tokenizer for decoder
        beam_size (int): beam size
    """
    def __init__(self,
                decoder,
                model_config,
                tokenizer,
                beam_size=1):
        #super(BeamSearch,self).__init__(decoder=decoder,model_config=model_config,generate_length=1,tokenizer=tokenizer)
        self.decoder=decoder
        self.model_config=model_config
        self.tokenizer=tokenizer
        self.beam_size = beam_size
        self.eos_token_id = self.tokenizer.eos_token_id
        self.vocab_size = model_config.vocab_size
        self.batch_size =model_config.batch_size
        self.seq_length =model_config.seq_length
    
    def generate_one_step(self,input_ids:Tensor,input_mask:Tensor,beam_size=1):
        """
        generate next token for only one step, use softmax to regularize logits

        Arg:
            input_ids (Tensor): (batch_size,seq_length) ids of input text
            input_mask (Tensor): (batch_size,seq_length) mask of input text, 0 for mask, 1 for reserved
            beam_size (int): int, beam_size for each candidate text
        
        Return:
            topk_indices (Tensor): (batch_size,beam_size), topk (k = beam_size) num of the next token indices for each batch 
            topk_logits (Tensor): (batch_size,beam_size), topk (k = beam_size) num of the next token logits(distribution) for each batch 
        """
        logits = self.decoder.predict(input_ids, input_mask)
        last_token_pos_recorder = LastTokenPos(input_mask)
        last_token_pos_list = last_token_pos_recorder.get_pos(shift=0)
        return_last_logits = extract_single_token_logits(logits, last_token_pos_list) #(batch_size,1,vocab_size)
        return_last_logits = P.Reshape()(return_last_logits,(self.batch_size,self.vocab_size)) #(batch_size,vocab_size)
        return_last_logits = P.Softmax()(return_last_logits)
        topk_logits,topk_indices = P.TopK(sorted=True)(return_last_logits,beam_size) #(batch_size,beam_size)
        return topk_indices,topk_logits


    
                   

    def generate(self,input_str=None,input_ids=None,input_mask=None,generate_length=1):
        """
        generate string using beam search

        Args:
            input_str (list) : list of input strings
            input_ids (Tensor): (batch_size,seq_length)
            input_mask (Tensor): (batch_size,seq_length)
            generate_length (int): length of tokens to generate
        
        Returns:
            max_beams_str_gen(str): string of generation
            max_beams_str_all(str): full string (prompt + generation)
        """


        #init inputs
        assert type(input_str) is list, "input_str a list not a {}.".format(type(input_str))
        if input_str is None:
            assert input_ids is not None and input_mask is not None,"if input_str is None, input_ids and input_mask both required."
        if input_str is not None:
            self.input_ids,self.input_mask,self.input_str_len_list = tensorize_ids_with_masks(input_str,config=self.model_config ,tokenizer=self.tokenizer)
        else:
            self.input_ids = input_ids
            self.input_mask = input_mask
            self.input_str_len_list = get_next_one_pos(input_mask).asnumpy().tolist()

        ranker = self.rank(batch_size = self.batch_size,
                                beam_size = self.beam_size,
                                input_ids=self.input_ids,
                                input_mask=self.input_mask,
                                past_ids_len=self.input_str_len_list,
                                generate_one_step=self.generate_one_step
                                )
        #generate for length-1 for 1 token is prepared directly
        for time_step in range(generate_length-1):
            ranker.beam_generate(self.beam_size)

        max_beam_index = [np.argmax(score) for score in ranker.prev_scores]
        max_beams_ids = []
        for batch_idx in range(self.batch_size):
            beam_index = max_beam_index[batch_idx]
            max_beams_ids.append(ranker.past_ids[batch_idx][beam_index].tolist())
        
        max_beams_str_all = [self.tokenizer.decode(ids[:min(original_len+generate_length,self.seq_length)]) for ids,original_len in zip(max_beams_ids,self.input_str_len_list)]
        max_beams_str_gen = [self.tokenizer.decode(ids[original_len-max(original_len+generate_length-self.seq_length,0):min(original_len+generate_length,self.seq_length)]) for ids,original_len in zip(max_beams_ids,self.input_str_len_list)]
        return max_beams_str_gen,max_beams_str_all
            
    class penalty():
        def __init__(self):
            pass

    class rank():
        """
        Args:
            batch_size (int): batch size
            beam_size (int): beam size
            input_ids (Tensor)
        """
        def __init__(self,batch_size:int,beam_size:int,input_ids:Tensor,input_mask:Tensor,past_ids_len:list,generate_one_step):
            self.batch_size = batch_size
            self.beam_size = beam_size
            self.gen_scores = np.zeros((self.batch_size,self.beam_size,self.beam_size)).astype(np.float32)
            self.prev_scores = np.zeros((self.batch_size,self.beam_size)).astype(np.float32)
            self.gen_ids = np.zeros((self.batch_size,self.beam_size,self.beam_size)).astype(np.int32)
            self.step = 0
            self.input_ids = input_ids
            self.input_ids_np = self.input_ids.asnumpy()
            #initialize input_mask by given input_mask
            self.input_mask = input_mask
            self.maximum_length = input_mask.shape[1]
            #list (batch_size,beam_size(rank),seq_length)
            #initialize strategy: set every beam as the same as the input
            #past_ids (List): (batch_size,beam_size), each element containing input_ids of a batch(seq_length,)
            self.past_ids = [ [ self.input_ids_np[batch_idx] for _ in range(self.beam_size)] for batch_idx in range(self.batch_size)]
            self.past_ids_len = [[past_ids_len[batch_idx] for _ in range(self.beam_size)] for batch_idx in range(self.batch_size)]
            self.generate_one_step = generate_one_step
            self.prepare_for_rank()

        def prepare_for_rank(self):
            topk_indices,topk_logits = self.generate_one_step(self.input_ids,self.input_mask,beam_size=self.beam_size)
            #update prev_scores directly from topk_logits
            self.prev_scores = self.score_func(topk_logits.asnumpy(),mode="log")
            topk_indices_np = topk_indices.asnumpy()
            for batch_idx in range(self.batch_size):
                for beam_idx in range(self.beam_size):
                    if self.past_ids_len[batch_idx][beam_idx] < self.maximum_length:
                        index_ = self.past_ids_len[batch_idx][beam_idx]
                        self.past_ids[batch_idx][beam_idx][index_] = topk_indices_np[batch_idx][beam_idx]
                        self.past_ids_len[batch_idx][beam_idx] += 1
                    else:
                        #shift if overflow
                        self.past_ids[batch_idx][beam_idx][:-1]=self.past_ids[batch_idx][beam_idx][1:]
                        self.past_ids[batch_idx][beam_idx][-1] = topk_indices_np[batch_idx][beam_idx]
                        
            #update input_mask
            self.input_mask = add_last_token_mask(input_mask=self.input_mask,overflow_strategy="shift")
        
        def get_input(self,rank_id):
            rank_input = []
            for batch_idx in range(self.batch_size):
                rank_input.append(self.past_ids[batch_idx][rank_id].tolist())
            rank_input_ids = Tensor(np.array(rank_input),dtype=mstype.int32)
            return rank_input_ids,self.input_mask
            
        def score_func(self,val,mode="log"):
            if mode == "log":
                return np.log(val)
            else:
                raise ValueError("score function {} Not supported".format(mode))

        def sum_score(self,prev,gen):
            """
            add up score(log)

            same shape of np.array as self.gen_score 
            """
            gen = self.score_func(gen,mode="log")
            
            for batch_id in range(self.batch_size):
                for beam_id in range(self.beam_size):
                    for rank_id in range(self.beam_size):
                        gen[batch_id][beam_id][rank_id] =  gen[batch_id][beam_id][rank_id] + prev[batch_id][beam_id]        
            return gen

        def get_2D_topk_index(self,score):
            """
            Arg:
                score (beam_size,beam_size(rank))
            return:
                list of tuples (parent_beam,rank_id)
            """
            #flatten
            score_ = score.reshape(-1)
            argsort = score_.argsort()
            return_list = []
            # 2D-score for a batch:
            # [[0.0,0.1,0.2],
            # [0.5,0.4,0.3],
            # [0.7,0.8,0.6]] 
            # --flatten--> [0.0,0.1,0.2,0.5,0.4,0.3,0.7,0.8,0.6] --argsort--> [0,1,2,5,4,3,7,8,6] 
            # --choose top k = beam_size --> [7,9,6] -- return_list --> [(2,1),(2,2),(2,0)]
            #print("[DEBUG] get_2D_topk_index",score_,argsort)
            for idx in range(self.beam_size*self.beam_size):
                if argsort[idx]>=self.beam_size*(self.beam_size-1):
                    parent_beam = int(idx)//int(self.beam_size)
                    rank_id = int(idx)%int(self.beam_size)
                    return_list.append((parent_beam,rank_id))
            return return_list

        def update_prev_score(self,candidates_index):
            """
            Return:
                total_prev_index (list): (batch_size,beam_size) updated index of prefix(past_ids)
            """
            total_prev_index = []
            for batch_idx in range(self.batch_size):
                prev_text_index = []
                for rank_idx in range(self.beam_size):
                    parent_index = candidates_index[batch_idx][rank_idx][0]
                    rank_index = candidates_index[batch_idx][rank_idx][1]
                    
                    # for debug:
                    #
                    # print("[DEBUG] update_prev_score",parent_index,rank_index)
                    # print(self.prev_scores)
                    # print(self.gen_scores)
                    # print(candidates_index)

                    #maintaining prev_index to update past_ids
                    prev_text_index.append(parent_index)
                    #update prev_score
                    self.prev_scores[batch_idx][rank_idx] = self.gen_scores[batch_idx][parent_index][rank_index] + self.prev_scores[batch_idx][parent_index]
                total_prev_index.append(prev_text_index)
            
            return total_prev_index
                 
        
        def gather_candidates(self,scores):
            """
            scores (batch_size,beam_size,beam_size(rank_size))
            Return:
                total_candidates list(batch_size,beam_size) of tuples,each tuple containing the candidates info (parent_beam,rank_id)
            """
            total_candidates = []
            for batch_idx in range(self.batch_size):
                batch_candidates = []
                score_ = scores[batch_idx]
                index_ = self.get_2D_topk_index(score_)
                total_candidates.append(index_)
            return total_candidates                



        def calculate_score(self):
            """
            interface of score calculation
            sum_score for now
            """
            scores = self.sum_score(self.prev_scores,self.gen_scores)
            return scores

        
        def reform_past_ids(self,past_ids_index,candidates_index):
            
            for batch_idx in range(self.batch_size):
                past_id = []
                for beam_idx in range(self.beam_size):
                    parent_index = candidates_index[batch_idx][beam_idx][0]
                    rank_index = candidates_index[batch_idx][beam_idx][1]
                    next_token_id = self.gen_ids[batch_idx][parent_index][rank_index]
                    if self.past_ids_len[batch_idx][beam_idx] >= self.maximum_length:
                        #if overflow, shift to left for 1 token
                        self.past_ids[batch_idx ][beam_idx][:-1] = self.past_ids[batch_idx ][beam_idx][1:]
                        self.past_ids[batch_idx ][beam_idx][-1] = next_token_id
                    else:
                        next_token_index = self.past_ids_len[batch_idx][beam_idx]
                        self.past_ids[batch_idx][beam_idx][next_token_index] = next_token_id
                        self.past_ids_len[batch_idx][beam_idx] += 1
            pass

        def record_beam(self,topk_indices,topk_logits,rank_id):
            """
            record_result of a single run
            """
            
            topk_indices = topk_indices.asnumpy()
            topk_logits = topk_logits.asnumpy()
            
            #record
            for batch_idx in range(self.batch_size):
                self.gen_scores[batch_idx][:][rank_id] = topk_logits[batch_idx][:]
                self.gen_ids[batch_idx][:][rank_id]  = topk_indices[batch_idx][:]
            pass


            
        def beam_generate(self,beam_size):
            """

            beam generation for each rank in size of beam_size

            Args:
                beam_size: beam size(rank size here)
            """
            #for loop to run full beam in one step of generation
            for rank_id in range(beam_size):
                self.input_ids,self.input_mask = self.get_input(rank_id)
                topk_indices,topk_logits = self.generate_one_step(self.input_ids,self.input_mask,beam_size=self.beam_size)
                self.record_beam(topk_indices,topk_logits,rank_id)

            #calculate scores first
            scores = self.calculate_score()
            #get candidates
            candidates_index = self.gather_candidates(scores)
            #update prev_score
            past_ids_index = self.update_prev_score(candidates_index)
            #update past_ids
            self.reform_past_ids(past_ids_index,candidates_index)
            #update input_mask
            self.input_mask = add_last_token_mask(input_mask=self.input_mask,overflow_strategy="shift")
            
class LastTokenPos():
        """
        class for record input_strs and the position of their last tokens 

        Args:
            input_ (Union): list if input is a list containing strs, Tensor with shape (batch_size,seq_length) representing input_mask
        """
        def __init__(self, input_:Union[list,Tensor]):
            self.input_strs = input_ if type(input_) is list else None
            self.input_mask = input_ if type(input_) is not list else None
            if self.input_strs is not None:
                self.pos_list = [ len(input_str)-1 for input_str in self.input_strs]
            else:
                #Tensor (batch_size,seq_length) --> list ,len(list) = batch_size
                input_mask_ = P.Cast()(self.input_mask,mstype.float32)
                temp_pos_list = P.ReduceSum(keep_dims=False)(input_mask_,axis=1).asnumpy().astype(np.int32).tolist()
                #minimum value is always 0 for safety 
                self.pos_list = [max(0,pos-1) for pos in temp_pos_list]
        
        def get_pos(self, shift:int = 0):
            shift_list = [pos+shift for pos in self.pos_list]
            return shift_list
     
class GenerationConfig():
    def __init__(self,file_path:Optional[str]=None,**kargs):
        if file_path is not None:
            kargs = self._load_kargs(file_path)
            # print("DEBUG kargs:  ",kargs)
        kargs = self.normalize(kargs)
        self.topk:int = kargs["topk"] if "topk" in kargs else 0
        self.topp:float = kargs["topp"] if "topp" in kargs else 1.0 
        self.temperature:float = kargs["temperature"] if "temperature" in kargs else 1.0
        self.generate_length:int =  kargs["generate_length"] if "generate_length" in kargs else 1
        self.beam_size:int = kargs["beam_size"] if "beam_size" in kargs else 1
        self.generate_mode:str = kargs["generate_mode"] if "generate_mode" in kargs else "sample"
        self.args = kargs
    def normalize(self,kargs):
        n_kargs = {}
        for key,value in kargs.items():
            n_kargs[key.lower()] = value
        return n_kargs
    def _load_kargs(self,file_path):
        with open(file_path,"r") as config:
            loaded_kargs = json.load(config)
        return loaded_kargs
    def get_config(self):
        return {
                "generate_mode":self.generate_mode,
                "topk":self.topk,
                "topp":self.topp,
                "temperature":self.temperature,
                "generate_length":self.generate_length,
                "beam_size":self.beam_size
                }
    def get_args(self):
        return self.args
    def get_arg(self,key):
        if key.lower() in self.args:
            return self.args[key.lower()]
        else:
            return None

if __name__ == '__main__':
    # s = Sample(None)
    # filt = TopKTopP_Filter(2,10,k=3,p=0.9,temperature=1.0,min_tokens_to_keep=1)
    # input_ = np.random.uniform(low=0.0,high=3.0,size=(2,10))
    # print(input_)
    # ret = filt.calculate(input_)
    # print(ret)
    pass


    
