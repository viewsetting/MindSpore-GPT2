from mindspore import Tensor
from mindspore import dtype as mstype
import numpy as np
from mindspore.ops import operations as P
from typing import TypeVar, Union, Optional
from .tokenization import Tokenizer,GPT2Tokenizer

class MockConfig:
    def __init__(self,input_ids:Optional[Tensor]=None,
                    input_mask:Optional[Tensor]=None,
                    tokenizer:Optional[GPT2Tokenizer]=None,
                    input_str:Optional[list]=None):
        if input_ids is not None:
            assert len(input_ids.shape) is 2,"input_ids should have 2 dims, but got {} dims.".format(len(input_ids.shape))
        if input_mask is not None:
            assert len(input_mask.shape) is 2,"input_mask should have 2 dims, but got {} dims.".format(len(input_mask.shape))
        if input_mask is None and input_ids is None and input_str is None :
            raise ValueError("There should at least one noNone param between input_mask ,input_str and input_ids.")
        
        self.tokenizer = tokenizer if tokenizer is not None else Tokenizer()
        self.input_ = input_ids if input_ids is not None else input_mask
        self.seq_length = self.input_.shape[1] if self.input_ is not None else 1024
        self.batch_size = self.input_.shape[0] if self.input_ is not None else len(input_str)
        self.vocab_size = self.tokenizer.vocab_size

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_input_tensors(self):
        return {"input_ids": self.input_ids, "input_mask":self.input_mask}

    def get_info(self):
        return {"seq_length":self.seq_length,"batch_size":self.batch_size,"vocab_size":self.vocab_size}