from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.ops import operations as P
from typing import TypeVar, Union
from tokenization import Tokenizer

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
        #assert tokenizer is not None, 'There is no tokenizer'

        assert config is not None, 'There is no GPT2-config'

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
            #assert True != True, ('mode:{} not supported.'.format(mode))
            raise NotImplementedError('mode:{} not supported.'.format(mode))
