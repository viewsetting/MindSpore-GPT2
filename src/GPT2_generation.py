""" For Beam Search and Nucleus Sampling etc. """
import numpy as np
from typing import TypeVar, Union
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Model, Parameter
from mindspore import dtype as mstype
from src.utils.extract_logits_lambada import extract_logits

INF = 1. * 1e9


class LengthPenalty(nn.Cell):
    """
    Length penalty.

    Args:
        weight (float): The length penalty weight.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.
    """

    def __init__(self, weight=1.0, compute_type=mstype.float32):
        super(LengthPenalty, self).__init__()
        self.weight = weight

        self.add = P.TensorAdd()
        self.pow = P.Pow()
        self.div = P.RealDiv()

        self.five = Tensor(5.0, mstype.float32)
        self.six = Tensor(6.0, mstype.float32)

        self.cast = P.Cast()

    def construct(self, length_tensor):
        """
        Process source sentence

        Inputs:
            length_tensor (Tensor):  the input tensor.

        Returns:
            Tensor, after punishment of length.
        """
        length_tensor = self.cast(length_tensor, mstype.float32)
        output = self.add(length_tensor, self.five)
        output = self.div(output, self.six)
        output = self.pow(output, self.weight)
        return output


class TileBeam(nn.Cell):
    """
    Beam Tile operation.

    Args:
        beam_width (int): The Number of beam.
        compute_type (mstype): Mindspore data type. Default: mstype.float32.
    """

    def __init__(self, beam_width, compute_type=mstype.float32):
        super(TileBeam, self).__init__()
        self.beam_width = beam_width

        self.expand = P.ExpandDims()
        self.tile = P.Tile()
        self.reshape = P.Reshape()
        self.shape = P.Shape()

    def construct(self, input_tensor):
        """
        Process source sentence

        Inputs:
            input_tensor (Tensor):  with shape (N, T, D).

        Returns:
            Tensor, tiled tensor.
        """
        shape = self.shape(input_tensor)
        # add an dim
        input_tensor = self.expand(input_tensor, 1)
        # get tile shape: [1, beam, ...]
        tile_shape = (1,) + (self.beam_width,)
        for _ in range(len(shape) - 1):
            tile_shape = tile_shape + (1,)
        # tile
        output = self.tile(input_tensor, tile_shape)
        # reshape to [batch*beam, ...]
        out_shape = (shape[0] * self.beam_width,) + shape[1:]
        output = self.reshape(output, out_shape)

        return output


class TopKTopP_Filter(nn.Cell):
    """
    top K sampling along with top P sampling
    Args:
        batch_size and vocab_size of model
        k for Top-K sampling and p for Top-P a.k.a. Necleus Sampling
        min_tokens_to_keep: a number for a guareented generation
    Inputs:
        distribution(Tensor): with shape (batch_size,vocab_size)
    Outputs:
        distribution(Tensor): with shape(batch_size, vocab_size), masked logits
        sorted_indexes(Tensor or None): Tensor with shape(batch_size,vocab_size) or None if do no sampling

    if k = 0, sorted_indexes will be None


    """

    def __init__(self, batch_size, vocab_size, k=0, p=1.0, temperature = 1.0,min_tokens_to_keep=1):
        super(TopKTopP_Filter, self).__init__()

        self.topK = P.TopK(sorted=True)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.min_tokens_to_keep = min_tokens_to_keep
        self.k = k
        self.p = p
        self.temp = temperature
        self.cumsum = P.CumSum()
        self.sample_function = P.Multinomial(seed=1)
        self.onehot = P.OneHot()
        self.cast = P.Cast()
        self.mask = Tensor(
            np.zeros((batch_size, vocab_size)), dtype=mstype.float32)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.softmax = P.Softmax()
        self.safty_mask_left = np.zeros(
            (batch_size, min_tokens_to_keep), dtype=float)
        self.safty_mask_right = np.ones(
            (batch_size, vocab_size-min_tokens_to_keep), dtype=float)
        self.safty_mask = Tensor(np.concatenate(
            (self.safty_mask_left, self.safty_mask_right), axis=1), dtype=mstype.float32)
        assert self.temp > 0.0,'temperature must be positive'
        assert self.k >= 0, 'the top_k number must be no negative.'
        if self.k > 0:
            assert self.min_tokens_to_keep <= self.k, 'K must be larger than or equal to min_token_to_keep for top p sampling'

    def construct(self, distribution: Tensor):
        distribution = self.softmax(distribution)
        if self.temp != 1.0:
            distribution = distribution / float(self.temp)

        values, indices = self.topK(distribution, self.k)
        sorted_indices = None

        # TOP K SAMPLE
        if self.k > 0:
            last_value = values[::, -1::]
            binary_mask = distribution >= last_value
            mask = self.cast(binary_mask, mstype.float32)
            distribution = distribution * mask
            distribution, sorted_indices = self.topK(
                distribution, self.vocab_size)
        else:
            distribution, sorted_indices = self.topK(
                distribution, self.vocab_size)

        # THEN TOP P SAMPLE
        if self.p < 1.0:
            #distribution = self.softmax(distribution)
            cumsum = self.cumsum(distribution, 1)

            # calculate remove indices mask, 1 for remove_indices
            # safty_mask: 0 for min_tokens_to_keep, multiply with indices_to_remove, add more 0.
            index_remove_binary = cumsum > self.p
            index_to_remove = self.cast(index_remove_binary, mstype.float32)
            index_to_remove = index_to_remove*self.safty_mask

            # get masked distribution
            remove_distribution = distribution*index_to_remove
            # substract to remove from distribution
            distribution = distribution - remove_distribution

        return distribution, sorted_indices


class Sample():

    """
    Sample

    Args:
        decoder(Model): GPT2 model to do generation
        model_config(GPT2Config): configuration of given GPT2 model
        generate_length(int): length of generation, if it is initailized, self.generate() will generate text based on it, unless a new
        length parameter is passed to self.generate()
        tokenizer(GPT2Tokenizer): if choose to use input_str parameter in self.generate(), a tokenizer is compulsory
        topk_num(int): number of K in top-K Sampling, 0 for no condition constrained, tantamount to K = self.vocab_size. Default:0
        topp_prob(float): probability parameter of topp sampling if p = 1.0, then it equals to do nothing. (nucleus sampling)
        early_stop(bool): whether stop when the model generates <EOS> token. It is functioned when batch_size is 1.
    """

    def __init__(self, decoder, model_config=None, generate_length=1, tokenizer=None,  input_ids=None, input_mask=None,  topk_num=0, topp_prob=1.0, temperature = 1.0,min_tokens_to_keep=1, early_stop=False,demo_mode=False,return_ids=False):

        # several checks for string mode or input tensors a.k.a. Tensor mode
        assert model_config is not None, 'Config is a must for sampling.'
        # assert (input_ids is None and input_str is not None)or(
        #     input_ids is not None and input_str is None), 'input_ids or input_str'
        # if input_str is not None:
        #     assert (
        #         tokenizer is not None), 'if choose to give input_str, a tokenizer is necessary.'
        if input_ids is not None:
            assert (
                input_mask is not None), 'input_mask is not found which should be associated with input_ids'

        self.model_config = model_config
        self.topk_num = topk_num
        self.topp_prob = topp_prob
        self.temperature = temperature
        self.min_tokens_to_keep = min_tokens_to_keep
        self.input_ids = input_ids
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.reshape = P.Reshape()
        self.cumsum = P.CumSum()
        self.onehot = P.OneHot()
        self.generate_length = generate_length
        self.seq_length = model_config.seq_length
        self.batch_size = model_config.batch_size
        self.vocab_size = model_config.vocab_size
        self.sample_function = P.Multinomial(seed=1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()
        self.concat = P.Concat()
        self.early_stop = early_stop
        self.demo_mode = demo_mode
        self.return_ids=return_ids
        
        self.filter_distribution = TopKTopP_Filter(
                    self.batch_size, self.vocab_size, k=self.topk_num,p=self.topp_prob,
                    temperature=self.temperature,min_tokens_to_keep=self.min_tokens_to_keep)

        if self.tokenizer is not None:
            self.eos_id = self.tokenizer.eos_token_id
        else:
            self.eos_id = model_config.vocab_size-1
        
        if self.demo_mode is True:
            assert self.batch_size == 1,'Demo mode requires batchsize euqals to 1, but get batch_size={}'.format(self.batch_size)

    def _extract_string_from_tensor(self, input_ids: Tensor,  mode="pair"):
        """
        Args:
            input_ids(Tensor): input tensor of sequence index. Shape: (self.batchsize,self.seq_length)
            mode(str):   "pair","single"and "CBT", "pair" for tasks with paired inputs, such as Text Summarization,
                    single output tasks with single input such as Language Modeling, "CBT" for the Children Book Test dataset, which has
                    3 componets of input (Leading text, Multiple choice to fill, Rest text).
        Return:
            prompt_list, list of prompt_text, or first part of text(list or str)
            reference_list, list of reference_text, or second part of text
            rest_list , list of rest_text, or rest part of text
            If self.batch_size is 1, it will return the first sentence of list, that is to say, the string.
                Example:
                for pair mode, if self.demo_mode is True, it will return prompt_list[0], reference_list[0]
        """
        assert self.tokenizer is not None, 'There is no tokenizer'
        prompt_list = [""]*self.batch_size
        reference_list = [""]*self.batch_size
        rest_list = [""]*self.batch_size
        eos_text = self.tokenizer.eos_token
        len_eos_text = len(eos_text)
        input_ids = self.reshape(input_ids, (self.batch_size, self.seq_length))

        # For datasets with paired inputs, such as Text Summarization(Article, Summary), QA(Question, Answer).
        if mode == "pair":

            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
                prompt_start = 0
                prompt_end = sentence.find(eos_text, 0)
                reference_start = prompt_end+len_eos_text
                reference_end = sentence[reference_start:].find(
                    eos_text, 0)+reference_start
                prompt_list[batch_idx]=sentence[prompt_start:prompt_end]
                reference_list[batch_idx]=sentence[reference_start:reference_end]

            if self.batch_size == 1 and self.demo_mode is True:
                return prompt_list[0], reference_list[0]
            else:
                return prompt_list, reference_list

        # For single output datasets such as WikiText, etc.
        elif mode == "single":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
                prompt_start = 0
                prompt_end = sentence.find(eos_text, 0)
                prompt_list[batch_idx]=sentence[prompt_start:prompt_end]
            if self.batch_size == 1 and self.demo_mode is True:
                return prompt_list[0]
            else:
                return prompt_list
        
        # For CBT dataset
        elif mode == "CBT":
            for batch_idx in range(self.batch_size):
                sentence_tensor = input_ids[batch_idx]
                sentence_list = sentence_tensor.asnumpy().tolist()[1:]

                sentence = self.tokenizer.decode(sentence_list)
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
            
            #return string(s) or list(s), str mode was designed for the benefit of interactive demo which it will
            #return a str that make sense for user and easy to use.
            if self.batch_size == 1 and self.demo_mode is True:
                return prompt_list[0], reference_list[0], rest_list[0]
            else:
                return prompt_list, reference_list, rest_list
        else:
            assert True != True, ('mode:{} not supported.'.format(mode))

    

    def _tensorize_ids_with_masks(self, src_str):
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

        input_shape = (self.batch_size, self.seq_length)
        single_sentence_shape = (1,self.seq_length)
        src_len_list = list()
        input_ids = None
        input_mask = None
        for batch_idx in range(self.batch_size):
            src_list=self.tokenizer.encode(src_str[batch_idx])
            #src_list = self.tokenizer.encode(src_str)
            src_len = len(src_list)
            if src_len > self.seq_length:
                src_list = src_list[:self.seq_length]
                src_len = self.seq_length
            
            src_len_list.append(src_len)
            ret_dict = self.tokenizer.prepare_for_model(
            src_list, max_length=self.model_config.seq_length, add_special_tokens=False)

            input_ids_list = ret_dict['input_ids']
            input_mask_list = ret_dict['attention_mask']

            input_ids_tensor = self.reshape(
            Tensor(np.array(input_ids_list, dtype=int), dtype=mstype.int32), single_sentence_shape)
            input_mask_tensor = self.reshape(
            Tensor(np.array(input_mask_list, dtype=int), dtype=mstype.int32), single_sentence_shape)
            if batch_idx == 0:
                input_ids = input_ids_tensor
                input_mask = input_mask_tensor
            else:
                input_ids = self.concat((input_ids,input_ids_tensor))
                input_mask = self.concat((input_mask,input_mask_tensor))
            

        return input_ids, input_mask, src_len_list

    
    def generate(self, input_str=None, input_ids=None, generate_length=None):
        """
        base function for text generation
        
        Args
            input_str ([str] or str): prompt string
            generate_length: number of tokens to generate
    
        Return:
            generate_str: string generated by the model
            full_str: input_str appended with generate_str
        """

        if input_str is not None:
            assert self.tokenizer is not None, 'if choose to give input_str, a tokenizer is necessary.'
        generate_str = [""]*self.batch_size

        #type check
        full_str = None
        
        if self.batch_size == 1 and self.demo_mode:
            full_str = [input_str]
        else:
            full_str = input_str
        
        self.input_ids = input_ids

        if generate_length is not None:
            generate_length = int(generate_length)
            assert generate_length >= 0, 'generate_length can not be negative.'
            self.generate_length = generate_length

        return_ids_list = [[] for i in range(self.batch_size)]
        for i in range(self.generate_length):

            # Tensor Mode
            if input_str is None:
                logits = self.decoder(self.input_ids, self.input_mask)
                nextword_distribution = self.reshape(
                    logits[::, len_str[0]-1:len_str[0]:1, ::], (batch_size, -1))
                
                distribution, real_index = self.filter_distribution(
                    nextword_distribution)
                word_index = self.sample_function(distribution, 1)

                float_real_index = self.cast(real_index, mstype.float32)
                result = self.reshape(self.onehot(
                    word_index, self.vocab_size, self.on_value, self.off_value), (self.batch_size, self.vocab_size))

                _real_index = self.cumsum(result*float_real_index, 1)[::, -1::]
                real_index = self.cast(_real_index, mstype.int32)
                real_index = self.reshape(
                    real_index, (-1,))  # Tensor (batch_size,)

            # string mode
            else:

                
                
                input_ids, input_mask, len_str = self._tensorize_ids_with_masks(
                    full_str)

                early_stop_mask = [0] * self.batch_size
                
                
                logits = self.decoder.predict(input_ids, input_mask)
                #print("DECODER Finished")

                # (batch_size,seq_length,vocab_size) ---> concatenate number batch_size of (1,vocab_size) --> (batch_size,vocab_size)
                nextword_distribution = self.reshape(
                    logits[0, len_str[0]-1:len_str[0]:1, ::], (1, -1))

                if self.batch_size > 1:
                    for batch_idx in range(1,self.batch_size):
                        nextword_single_distribution = self.reshape(
                    logits[batch_idx, len_str[batch_idx]-1:len_str[batch_idx]:1, ::], (1, -1))
                        nextword_distribution = self.concat((nextword_distribution,nextword_single_distribution))
                # if i==0:
                #     print('[DEBUG INFO] len_str:{}'.format(len_str))
                #     print('[DEBUG INFO] nextword_distribution:{} shape:{}'.format(nextword_distribution[::,:50], nextword_distribution.shape))
                #next_word_distribution = self.softmax(nextword_distribution)
                # print("NEXT_WORD",nextword_distribution)
               
                # print("TOPKTOPP")
                distribution, real_index = self.filter_distribution(
                    nextword_distribution)
                # if i==0:
                #     print('[DEBUG INFO] distribution:{} shape:{}'.format(distribution[::,:50],distribution.shape))
                # (batch_size,vocab_size) --> (batch_size)
                word_index = self.sample_function(distribution, 1)

                # if i==0:
                #     print('[DEBUG INFO] word_index:{} shape:{}'.format(word_index,word_index.shape))

                float_real_index = self.cast(real_index, mstype.float32)
                result = self.reshape(self.onehot(
                    word_index, self.vocab_size, self.on_value, self.off_value), (self.batch_size, self.vocab_size))

                _real_index = self.cumsum(result*float_real_index, 1)[::, -1::]
                real_index = self.cast(_real_index, mstype.int32)
                sampled_next_word_index = self.reshape(
                    real_index, (-1,))  # Tensor (batch_size,)

                #print("REAL_INDEX: ",sampled_next_word_index)
                
                sampled_next_word_index_list = sampled_next_word_index.asnumpy().tolist()

                for batch_idx in range(self.batch_size):
                    next_word_index = sampled_next_word_index_list[batch_idx]
                    # earlystop if the model generates a EOS token. For batch_size = 1 situation only.
                    if next_word_index == self.eos_id and self.early_stop is True and self.batch_size == 1:
                        break
                    if next_word_index == self.eos_id and self.early_stop is True:
                        early_stop_mask[batch_idx] = 1
                    if early_stop_mask[batch_idx] == 1 and self.early_stop is True:
                        continue
                    next_word_str = self.tokenizer.decode([next_word_index])
                    return_ids_list[batch_idx].append(next_word_index)
                    full_str[batch_idx] += next_word_str
                    generate_str[batch_idx] += next_word_str

                if 0 not in early_stop_mask:
                    break

                        
                        



        if self.batch_size == 1 and self.demo_mode is True:
            if self.return_ids == True:
                return generate_str[0], full_str[0],return_ids_list[0]
            else:
                return generate_str[0], full_str[0]
        else:
            if self.return_ids == True:
                return return_ids_list
            return generate_str, full_str


    def generate_for_CNN_DAILYMAIL(self, input_ids, generate_length=100, select_sentence=0, TL_DR=True):

        """
        Args
            input_ids(Tennor): input_ids(shape: (self.batch_size,s self.eq_length)) of dataset which is sampled from mindrecord
            generate_length(int): tokens to generate
            select_sentence(int): number of leading sentences in generation to be selected for hypothesis string.
                            0 for return full generation, if there are less sentences in generation, full generation will
                            be returned, either.
            TL_DR(bool): True for one "TL,DR" token padded in article, False for no.
    
        Return:
            generated_summary: generated string of the model
            summary_str: summary string in dataset as label or reference string
        """

        article_str, summary_str = self._extract_string_from_tensor(
            input_ids, mode="pair")
        

        generated_summary_list= [""] * self.batch_size

        tldr_str = "TL;DR:"
        #pad a <TL,DR;> token(<EOS>) after the string of Article.
        if TL_DR:
            for article_idx in range(self.batch_size):
                article_str[article_idx]+=(" "+tldr_str)
        
        # print("[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL article_str:")
        # print(article_str)

        generate_str_list, _ = self.generate(
            input_str=article_str, generate_length=generate_length)

        # print("[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL generate_str_list:")
        # print(generate_str_list)
        
        for article_idx in range(self.batch_size):
            generate_str = generate_str_list[article_idx]
            generated_summary = ""
            if select_sentence > 0:
                # check if there are number of select_sentence of sentences in generated text,if not enough, it will return full generated string
                len_generate_str = len(generate_str)
                search_index = -1
                for i in range(select_sentence):
                    search_index = generate_str.find('.',search_index+1)
                    if search_index == -1 or search_index >= len_generate_str:
                        search_index = len_generate_str
                        break
            
                #increase search_index to add period token('.') if search_index does not overflow.
                search_index = search_index+1 if search_index < len_generate_str else len_generate_str
                generated_summary = generate_str[:search_index]

            else:
                generated_summary = generate_str
            if generated_summary == '':
                generated_summary = generate_str
               
            if self.tokenizer.eos_token in generated_summary:
                cut_pos = generated_summary.find(self.tokenizer.eos_token,0)
                generated_summary = generated_summary[:cut_pos]
                
            if generated_summary == '':
                generated_summary = '<empty>'
            generated_summary_list[article_idx] = generated_summary

            # print("[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL debugging info:\nGENERATED_SUMMARY:")
            # print(generated_summary_list[article_idx])
            # print(summary_str[article_idx])

        return generated_summary_list, summary_str  # Hypo and Ref

    def generate_for_Translation(self, input_ids, use_hint=True, select_first_sentence=True, max_generate_length=150):

        """
        Args
            input_ids (Tensor): input_ids(shape: (self.batch_size, self.seq_length) of dataset which is sampled from mindrecord
            use_hint (bool): wheather use the "=" hint to help infer in Translation task (english sentence = french sentence). Default: True.
            select_first_sentence (bool): wheather use the first generated sentence as the translation result. Default: True.
            max_generate_length: the max token length of generation sentence. Default: 150.
    
        Return:
            final_translation_list (List[String]): the final translation results, shape [batch_size].
            ref_str_list (List[String]): the traget/reference translation results, shape [batch_size].
        """

        self.early_stop = True
        source_str_list, ref_str_list = self._extract_string_from_tensor(
            input_ids, mode="pair")
        

        final_translation_list= [""] * self.batch_size

        if use_hint:
            for index in range(self.batch_size):
                source_str_list[index] += " =" # now source_str is "english sentence ="

        translation_str_list, _ = self.generate(input_str=source_str_list, generate_length=max_generate_length)
        
        for index in range(self.batch_size):
            generate_str = translation_str_list[index]
            predict_tarnslation = ""
            
            # Acording to the GPT2 paper, the select_first_sentence will be set "True"
            if select_first_sentence:
                # check if there are number of select_sentence of sentences in generated text,if not enough, it will return full generated string
                search_index = generate_str.find('.', 0, len(generate_str))
                if search_index == -1:
                    search_index = len(generate_str) # not find "."
                else:
                    search_index = search_index + 1 # find "." successfully
                predict_tarnslation = generate_str[:search_index]
            else:
                predict_tarnslation = generate_str

            if predict_tarnslation == '':
                predict_tarnslation = '<empty>'
            
            final_translation_list[index] = predict_tarnslation


        return final_translation_list, ref_str_list  # Hypo and Ref

    def generate_for_LAMBADA(self, input_ids,logits,max_generate_length=3,max_iterations=20):
        """
        Args:
            input_ids(Tennor): input_ids(shape: (self.batch_size,self.seq_length)) of dataset which is sampled from mindrecord
            logits: (batch_size,seq_length,vocab_size) (8,1024,50257)
            max_generate_length(int): the number of tokens to generate
    
        Return:
            generated_last_word: generated the last word of lambada
        """

        self.early_stop = False
        self.return_ids = True
        self.topk_num = 1
        self.topp_prob = 1.0
        source_str = self._extract_string_from_tensor(input_ids,mode="single")
        
        #True if generated
        generate_batch_flag = [False]*self.batch_size

        #All of the batches are generated 
        all_true = [True]*self.batch_size

        #lastword string list
        # final_generations = [""*self.batch_size]        # [""*bsz]
        final_generations = ["" for _ in range(self.batch_size)]        # ['','',',...'']

        MAX_NUM = 99999

        stop_word = [self.tokenizer.eos_token,'.',',','!','?','"',"~"]
       # tokenizer = Tokenizer(vocab_file="utils/pretrain-data/gpt2-vocab.json", merge_file="utils/pretrain-data/gpt2-merges.txt")
        # source_str = self.extract_string_from_tensor(input_ids, mode="single")
        lastword_start_pos_ = get_lastword_range(input_ids = input_ids,config=self.model_config,tokenizer=self.tokenizer)     # [(left_pos,right_pos)] -> batch_size for list length
        lastword_start_pos = []  
        for item in lastword_start_pos_:
            lastword_start_pos.append(item[0])

        logits = extract_logits(logits = logits, seq_pos=lastword_start_pos)  #(8,1,50257)
        #final_logits = np.argmax(output_logits,axis = -1)          #(8,) [ids of list]
        topk = P.TopK(sorted=True)
        _ , sorted_ids = topk(logits,max_iterations)
        sorted_ids = sorted_ids.asnumpy()                       # [batch_size,1,max_iterations]
        sorted_ids = sorted_ids.reshape((-1,max_iterations))    # [batch_size,max_iterations]
        sorted_ids = sorted_ids.T                               # [max_iterations,batch_size]
        sorted_ids = sorted_ids.tolist()                        # [[121,3,123,41],[3123,3123,43,12],...,]  (100,8)

        for i in range(max_iterations):
            # source_str +=
            ids = sorted_ids[i]
            ids_str = [ self.tokenizer.decode([x]) for x in ids]
            cat_str = [x+y for x,y in zip(source_str,ids_str)]
            generate_ids_list = self.generate(input_str=cat_str, generate_length=2) # [[23,34,45,78,90],[34,56,79,89,667] ]
            # print("generate_ids_list :",generate_ids_list.shape)
            cat_ids_list = [[x]+y for x,y in zip(ids,generate_ids_list)]
            res_str_list = [self.tokenizer.decode(word) for word in cat_ids_list]       # [" hel lo <|endoftext|>","word ",...]
            # print("===========[DEBUG] generate_for_LAMBADA res_str_list ===== iteration:{}==========".format(i))
            print(res_str_list)
            
            #res_str_list = [word.lstrip().rstrip() for word in res_str_list]            # ["hel lo","word",...]
            for j in range(self.batch_size):
                if generate_batch_flag[j]:
                    continue
                
                eos_pos = min( res_str_list[j].find(word) if res_str_list[j].find(word) >=0 else MAX_NUM for word in stop_word)
                # print("EOS_pos: ",eos_pos )
                #eos_pos = min(res_str_list[j].find('.'),res_str_list[j].find(self.tokenizer.eos_token),res_str_list[j].find('.'),)
                if eos_pos == MAX_NUM:
                    continue    
                else:
                    res_str_list[j] = res_str_list[j][:eos_pos]
                
                res_str_list[j] = res_str_list[j].lstrip().rstrip()
                print(res_str_list[j])

                if res_str_list[j].find(" ") == -1 :     # don't have space in a word, set True
                    if res_str_list[j] == "":
                        continue
                    
                    generate_batch_flag[j] = True
                    final_generations[j] = res_str_list[j] 

            #print(generate_batch_flag)
            if all_true == generate_batch_flag:
                # print("Success")
                break

        return final_generations

if __name__ == '__main__':
    #s = Sample(None)

    pass


    
