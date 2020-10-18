""" For Beam Search and Nucleus Sampling etc. """
import numpy as np
from typing import TypeVar, Union
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Model, Parameter
from mindspore import dtype as mstype

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

    def __init__(self, batch_size, vocab_size, k=0, p=1.0, min_tokens_to_keep=1):
        super(TopKTopP_Filter, self).__init__()

        self.topK = P.TopK(sorted=True)
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.min_tokens_to_keep = min_tokens_to_keep
        self.k = k
        self.p = p
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
        assert self.k >= 0, 'the top_k number must be no negative.'
        if self.k > 0:
            assert self.min_tokens_to_keep <= self.k, 'K must be larger than or equal to min_token_to_keep for top p sampling'

    def construct(self, distribution: Tensor):

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
            distribution = self.softmax(distribution)
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

    def __init__(self, decoder, model_config=None, generate_length=1, tokenizer=None,  input_ids=None, input_mask=None,  topk_num=0, topp_prob=1.0, min_tokens_to_keep=1, early_stop=False):

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
        self.early_stop = early_stop

        if self.tokenizer is not None:
            self.eos_id = self.tokenizer.eos_token_id
        else:
            self.eos_id = model_config.vocab_size-1

    def extract_string_from_tensor(self, input_ids: Tensor,  mode="pair"):
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
                for pair mode, if self.batchsize=1, it will return prompt_list[0], reference_list[0]
        """
        assert self.tokenizer is not None, 'There is no tokenizer'
        prompt_list = []
        reference_list = []
        rest_list = []
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
                prompt_list.append(sentence[prompt_start:prompt_end])
                reference_list.append(sentence[reference_start:reference_end])

            if self.batch_size == 1:
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
                prompt_list.append(sentence[prompt_start:prompt_end])
            if self.batch_size == 1:
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

                prompt_list.append(sentence[prompt_start:prompt_end])
                reference_list.append(sentence[reference_start:reference_end])
                rest_list.append(sentence[rest_start:rest_end])
            
            #return string(s) or list(s), str mode was designed for the benefit of interactive demo which it will
            #return a str that make sense for user and easy to use.
            if self.batch_size == 1:
                return prompt_list[0], reference_list[0], rest_list[0]
            else:
                return prompt_list, reference_list, rest_list
        else:
            assert True != True, ('mode:{} not supported.'.format(mode))

    

    def tensorize_ids_with_masks(self, src_str):
        """
        Transform from string to tensor

        Args:
            src_str: string
        Return:
            input_ids: Tensor(self.batch_size, self.seq_length)
            input_mask: Tensor(self.batch_size, self.seq_length)
            src_len: length of tokens of src_string after decoded by self.tokenzier
        """

        input_shape = (self.batch_size, self.seq_length)

        src_list = self.tokenizer.encode(src_str)
        src_len = len(src_list)
        if src_len > self.seq_length:
            src_list = src_list[:self.seq_length]
            src_len = self.seq_length
        ret_dict = self.tokenizer.prepare_for_model(
            src_list, max_length=self.model_config.seq_length, add_special_tokens=False)

        input_ids_ = ret_dict['input_ids']
        input_mask_ = ret_dict['attention_mask']

        input_ids = self.reshape(
            Tensor(np.array(input_ids_, dtype=int), dtype=mstype.int32), input_shape)
        input_mask = self.reshape(
            Tensor(np.array(input_mask_, dtype=int), dtype=mstype.int32), input_shape)

        return input_ids, input_mask, src_len

    
    def generate(self, input_str=None, input_ids=None, generate_length=None):
        """
        base function for text generation
        
        Args
            input_str: prompt string
            generate_length: number of tokens to generate
    
        Return:
            generate_str: string generated by the model
            full_str: input_str appended with generate_str
        """

        if input_str is not None:
            assert self.tokenizer is not None, 'if choose to give input_str, a tokenizer is necessary.'
        generate_str = ""
        full_str = input_str
        self.input_ids = input_ids

        if generate_length is not None:
            generate_length = int(generate_length)
            assert generate_length >= 0, 'generate_length can not be negative.'
            self.generate_length = generate_length

        for _ in range(self.generate_length):

            # Tensor Mode
            if input_str is None:
                logits = self.decoder(self.input_ids, self.input_mask)
                nextword_distribution = self.reshape(
                    logits[::, len_str-1:len_str:1, ::], (batch_size, -1))
                filter_distribution = TopKTopP_Filter(
                    self.batch_size, self.vocab_size)
                distribution, real_index = filter_distribution(
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
                input_ids, input_mask, len_str = self.tensorize_ids_with_masks(
                    full_str)

                logits = self.decoder.predict(input_ids, input_mask)
                #print("DECODER Finished")

                # (batch_size,seq_length,vocab_size) ---> (batch_size,1,vocab_length) --> (batch_size,vocab_length)
                nextword_distribution = self.reshape(
                    logits[::, len_str-1:len_str:1, ::], (self.batch_size, -1))
                #next_word_distribution = self.softmax(nextword_distribution)
                # print("NEXT_WORD",nextword_distribution)
                filter_distribution = TopKTopP_Filter(
                    self.batch_size, self.vocab_size, self.topk_num, self.topp_prob, self.min_tokens_to_keep)

                # print("TOPKTOPP")
                distribution, real_index = filter_distribution(
                    nextword_distribution)

                # (batch_size,vocab_size) --> (batch_size)
                word_index = self.sample_function(distribution, 1)

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

                    next_word_str = self.tokenizer.decode([next_word_index])
                    full_str += next_word_str
                    generate_str += next_word_str

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

        article_str, summary_str = self.extract_string_from_tensor(
            input_ids, mode="pair")
        
        #pad a <TL,DR;> token(<EOS>) after the string of Article.
        if TL_DR:
            article_str += (" "+self.tokenizer.eos_token)
        
        print("Sample.generate_for_CNN_DAILYMAIL debugging info:\nARTICLE STR:")
        print(article_str)
        generate_str, _ = self.generate(
            input_str=article_str, generate_length=100)
        generated_summary = ""
        if int(select_sentence) > 0:
                # check if there are number of select_sentence of sentences in generated text,if not enough, it will return full generated string
            stop_pos = generate_str.find('.', select_sentence-1)
            if stop_pos == -1:
                stop_pos = len(generate_str)
            generated_summary = generate_str[:stop_pos]
        else:
            generated_summary = generate_str
        return generated_summary, summary_str  # Hypo and Ref


class BeamSearchDecoder(nn.Cell):
    def __init__(self, model_config, decoder, input_ids=None, beam_width=4, length_penalty_weight=1.0, max_decode_length=64, bos_id=50256):
        self.model_config = model_config
        self.batch_size = self.model_config.batch_size
        self.seq_length = self.model_config.seq_length
        self.vocab_size = self.model_config.vocab_size
        self.length_penalty_weight = length_penalty_weight
        self.max_decode_length = max_decode_length
        self.topK = P.TopK(sorted=True)
        self.bos_id = bos_id
        if input_ids == None:
            pass
        else:
            self.input_ids = input_ids

    def construct(self):
        for _ in range(self.max_decode_length):
            pass


if __name__ == '__main__':
    print('*'*65)
    print('We are now in testing mode for GPT2_generation.py')
    print('*'*65)

    def set_env(mode="GPU", device_id=0,ckpt_path="/datasets/pretrained_weights/ms_model_medium.ckpt"):
        from mindspore import context
        import os
        from finetune_eval_config import cfg, gpt2_net_cfg
        import mindspore.common.dtype as mstype
        from mindspore import log as logger
        from mindspore.train.model import Model
        from mindspore.common.tensor import Tensor
        from mindspore.train.serialization import load_checkpoint, load_param_into_net
        from utils.tokenization import Tokenizer
        from mindspore.ops import operations as P
        from GPT2ForSummarization import GPT2ForPredictNext
        from GPT2ForLanguageModel import  GPT2LanguageModel
        #from GPT2_generation import Sample
        import numpy as np
        context.set_context(mode=context.GRAPH_MODE,
                            device_target=mode, device_id=device_id)
        context.set_auto_parallel_context(parallel_mode="stand_alone")
        print('set context as: {}, using device {}.'.format(mode, device_id))

        gpt2_loss =  GPT2ForPredictNext(config=gpt2_net_cfg,
                               is_training=False,
                               use_one_hot_embeddings=False)
        load_checkpoint_path = ckpt_path
        gpt2_loss.set_train(False)
        param_dict = load_checkpoint(load_checkpoint_path)

        param_dict_ = {}

        print("====process param_dict========")
        for msname in param_dict:
            param_dict_['gpt2.'+msname] = param_dict[msname]
        param_dict_['lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
        print("====load params into model====")
        load_param_into_net(gpt2_loss, param_dict_)

        model = Model(gpt2_loss)
        return model,gpt2_net_cfg

    def get_random_tensor(shape: Union[list, tuple], mode='randn', dtype=mstype.float32):
        if mode == 'randn':
            np_array = np.random.randn(*shape)
            return Tensor(np_array, dtype=dtype)
        if mode == 'uniform':
            np_array = np.random.uniform(size=shape)
            return Tensor(np_array, dtype=dtype)
        pass

    def list2tensor(lst, dtype=mstype.float32):
        return Tensor(np.array(lst), dtype=dtype)

    print('Set Running Env and Load Model')
    gpt2,config = set_env()
    gen_len = 200

    
    tokenizer = Tokenizer(vocab_file='./src/utils/pretrain-data/gpt2-vocab.json',
                      merge_file='./src/utils/pretrain-data/gpt2-merges.txt')

    sample = Sample(gpt2,generate_length=gen_len,tokenizer = tokenizer,
            model_config=config,topk_num=0,topp_prob=0.92,min_tokens_to_keep=1)
    
    while True:
        raw_text = input("Model Prompt >>>")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        gen_str,full_str = sample.generate(input_str=raw_text,generate_length=gen_len)
        print("*"*100)
        print("GPT2 Generation >>>",gen_len)
        print("*"*100)
        print("Full Text Here >>>",full_str)
        print("*"*100)

    