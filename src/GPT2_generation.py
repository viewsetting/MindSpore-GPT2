""" For Beam Search and Nucleus Sampling etc. """
import numpy as np
from typing import TypeVar, Union
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor,Model,Parameter
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
    def __init__(self,batch_size,vocab_size,k=0,p=1.0,min_tokens_to_keep=1):
        super(TopKTopP_Filter,self).__init__()

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
        self.mask = Tensor(np.zeros((batch_size, vocab_size)), dtype=mstype.float32)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.softmax = P.Softmax()
        self.safty_mask_left = np.zeros((batch_size,min_tokens_to_keep),dtype=float)
        self.safty_mask_right = np.ones((batch_size,vocab_size-min_tokens_to_keep),dtype=float)
        self.safty_mask = Tensor(np.concatenate((self.safty_mask_left,self.safty_mask_right),axis=1),dtype=mstype.float32)
        assert self.min_tokens_to_keep < self.k,'K must be larger than min_token_to_keep for top p sampling'
        
   
    def construct(self,distribution:Tensor):
        
        values,indices = self.topK(distribution,self.k)
        sorted_indices = None
        
        #TOP K SAMPLE
        if self.k > 0:
            last_value = values[::,-1::]
            binary_mask = distribution >=last_value
            mask = self.cast(binary_mask,mstype.float32)
            distribution = distribution * mask
            distribution,sorted_indices = self.topK(distribution,self.vocab_size)
            
        #THEN TOP P SAMPLE
        if self.p < 1.0:
            distribution = self.softmax(distribution)
            cumsum = self.cumsum(distribution,1)

            #calculate remove indices mask, 1 for remove_indices
            #safty_mask: 0 for min_tokens_to_keep, multiply with indices_to_remove, add more 0.
            index_remove_binary = cumsum > self.p
            index_to_remove = self.cast(index_remove_binary,mstype.float32)
            index_to_remove = index_to_remove*self.safty_mask

            #get masked distribution
            remove_distribution = distribution*index_to_remove
            #substract to remove from distribution
            distribution = distribution - remove_distribution
        

        return distribution,sorted_indices
            


class Sample(nn.Cell):
    def __init__(self,decoder,generate_length=1,tokenizer=None,model_config=None,input_ids=None,input_mask=None,input_str= None,topk_num=40,topp_prob=1.0,early_stop=False):
        
        #several checks for string mode or input tensors a.k.a. Tensor mode
        assert  model_config is not None,'Config is a must for sampling.'
        assert  (input_ids is  None and input_str is not None)or(input_ids is not None and input_str is  None),'input_ids or input_str'
        if input_str is not None:
            assert (tokenizer is not None ),'if choose to give input_str, a tokenizer is necessary.'
        if input_ids is not None:
            assert  ( input_mask is None),'input_mask is not found which should be associated with input_ids'

        self.model_config = model_config
        self.topk_num = topk_num
        self.topp_prob = topp_prob
        self.input_ids = input_ids
        self.input_str = input_str
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.reshape = P.Reshape()
        self.cumsum = P.CumSum()
        self.softmax = P.Softmax(axis = -1)
        self.generate_length = generate_length
        self.seq_length = model_config.seq_length
        self.batch_size = model_config.batch_size
        self.vocab_size = model_config.vocab_size
        self.sample_function = P.Multinomial(seed=1)
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.cast = P.Cast()

        if self.tokenizer is not None:
            self.eos_id = self.tokenizer.eos_token_id
        else:
            self.eos_id = model_config.vocab_size-1

    def tensorize_ids_with_masks(self,tokenzier,src_str):
        src_list = tokenzier.encode(src_str)
        src_len = len(src_list)
        if src_len > self.seq_length:
            src_list = src_list[:self.seq_length]
            src_len = self.seq_length
        ret_dict = tokenizer.prepare_for_model(src_list,max_length=self.model_config.sequence_length,add_special_tokens=False)
        
        input_ids_ = ret_dict['input_ids']
        input_mask_ = ret_dict['attention_mask']

        input_ids = Tensor(np.array(input_ids_,dtype=int),dtype=mstype.int32)
        input_mask = Tensor(np.array(input_mask_,dtype=int),dtype=mstype.int32)

        return input_ids,input_mask,src_len

    def construct(self,input_ids):
        
        generate_str = ""
        full_str = self.input_str

        for _ in range(self.generate_length):
        
            # Tensor Mode
            if self.input_ids is None:
                logits = self.decoder(self.input_ids,self.input_mask)

            # string mode
            else:
                input_ids, input_mask,len_str = self.tensorize_ids_with_masks(self.input_str)
                logits = self.decoder(input_ids,input_mask)
                #(batch_size,seq_length,vocab_size) ---> (batch_size,1,vocab_length) --> (batch_size,vocab_length)
                nextword_distribution = self.reshape(logits[::,len_str-1:len_str:1,::],(batch_size,-1))
                #next_word_distribution = self.softmax(nextword_distribution)
                filter_distribution = TopKTopP_Filter(self.batch_size,self.vocab_size)
                distribution,real_index = filter_distribution(nextword_distribution)
                
                #(batch_size,vocab_size) --> (batch_szie)
                word_index = self.sample_function(distribution,1)

                
                float_real_index = self.cast(real_index,mstype.float32)
                result = reshape(onehot(word_index,self.vocab_size, self.on_value, self.off_value),(2,3))
                
                _real_index = self.cumsum(result*float_real_index,1)[::,-1::]
                real_index = self.cast(_real_index,mstype.int32)
                real_index = self.reshape(real_index,(-1,)) #Tensor (batch_size,)

                #print(real_index)
    


                if self.early_stopping and self.batch_size == 1 and False:
                    
                    break
        

        return generate_str,full_str



class BeamSearchDecoder(nn.Cell):
    def __init__(self,model_config,decoder,input_ids=None,beam_width=4,length_penalty_weight=1.0,max_decode_length=64,bos_id=50256):
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



if __name__=='__main__':
    print('*'*65)
    print('We are now in testing mode for GPT2_generation.py')
    print('*'*65)
    def set_env(mode="GPU",device_id=0):
        from mindspore import context
        context.set_context(mode=context.GRAPH_MODE, device_target=mode, device_id=device_id)
        print('set context as: {}, using device {}.'.format(mode,device_id))

    set_env()

    def get_random_tensor(shape:Union[list,tuple],mode='randn',dtype=mstype.float32):
        if mode == 'randn':
            np_array = np.random.randn(*shape)
            return Tensor(np_array,dtype=dtype)
        if mode == 'uniform':
            np_array = np.random.uniform(size=shape)
            return Tensor(np_array,dtype=dtype)
        pass
    def list2tensor(lst,dtype=mstype.float32):
        return Tensor(np.array(lst),dtype=dtype)
    
    
    #test = get_random_tensor((2,6),mode='uniform')
    lst = [[0.1,0.8,0.15],[0.7,0.1,0.2]]
    lst2 = [[0.1,0.8,0.2],[0.7,0.1,0.1]]
    test = list2tensor(lst)
    test2 = list2tensor(lst2)
    # print(test<=test2)
    # mask = (test<=test2)
    cast = P.Cast()
    # mask = cast(mask,mstype.float32)
    # print(mask)
    # print(test*mask)
   # print(test.shape[0],float(test[0][0].asnumpy()))
    topk = P.TopK(sorted=True)
    last_val = topk(test,2)[0][::,-1::]
    mask = test >= last_val
    mask = cast(mask,mstype.float32)
    # print(mask)
    # print(mask*test)
    print(test)
    topk = TopKTopP_Filter(2,3,2,0.6)
    ret,ind = topk(test)
    print(ret)

    muno = P.Multinomial(seed=1)
    samples = muno(ret,1)
    print(samples)
    print(ind)
    reshape=P.Reshape()
    # ind = cast(ind,mstype.float32)
    # axis = 1
    # gather = P.GatherV2()(ind,samples,axis)
    # print("GAHTERV2")
    # print(gather)
    onehot = P.OneHot()
    depth, on_value, off_value = 3, Tensor(1.0, mstype.float32), Tensor(0.0, mstype.float32)
    ind = cast(ind,mstype.float32)
    result = reshape(onehot(samples,depth, on_value, off_value),(2,3))
    print(result)
    cs = P.CumSum()
    real_index = cs(result*ind,1)[::,-1::]
    real_index = cast(real_index,mstype.int32)
    real_index = reshape(real_index,(-1,))
    print(real_index)
    
    
