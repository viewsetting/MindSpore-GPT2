""" For Beam Search and Nucleus Sampling etc. """
import numpy as np
from typing import TypeVar, Union, Optional
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor, Model, Parameter
from mindspore import dtype as mstype
from .utils.lambada_utils import extract_logits
from .utils.tensor_manipulations import extract_string_from_tensor,extract_single_token_logits,tensorize_ids_with_masks,add_last_token_mask,get_next_one_pos
from mindspore.context import get_context
from .utils.tokenization import GPT2Tokenizer,Tokenizer
from .utils.mock_config import MockConfig
from .GPT2_model import GPT2Config
from .utils.generation_utils import Sample, BeamSearch, GenerationConfig

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
        self.cast = P.Cast()

        self.five = Tensor(5.0, mstype.float32)
        self.six = Tensor(6.0, mstype.float32)

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

     
def generate_for_CNN_DAILYMAIL( decoder:Model,
                                input_ids:Tensor, 
                                model_config:Optional[GPT2Config]=None,
                                tokenizer:Optional[GPT2Tokenizer]=None,
                                generate_length:Optional[int]=None, 
                                select_sentence=3, 
                                TL_DR=True,
                                tldr_str="TL;DR:",
                                generate_config:Optional[GenerationConfig]=None):

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
        #set model_config and tokenizer
        mock_config = MockConfig(input_ids=input_ids,tokenizer=tokenizer)
        if model_config is None:
            model_config = mock_config
        if tokenizer is None:
            tokenizer = mock_config.get_tokenizer()
        
        #set generate_config
        if generate_config is None:
            generate_config = GenerationConfig()

        #load param from generate
        topk = generate_config.topk
        topp = generate_config.topp
        temperature = generate_config.temperature
        beam_size = generate_config.beam_size
        generate_mode = generate_config.generate_mode

        #print("[DEBUG] topk: {},  topp: {}, temperature: {}, generate_mode: {}".format(topk,topp,temperature,generate_mode))

        #reload generate_length from config if not specified from param of function
        generate_length = generate_config.generate_length if generate_length is None else generate_length

        if generate_mode == "sample":
            generator = Sample(decoder,tokenizer=tokenizer,model_config=model_config,topk_num = topk,topp_prob=topp,
        min_tokens_to_keep=1,demo_mode=False,temperature=temperature)
        elif generate_mode == "beam":
            generator = BeamSearch(decoder,model_config=model_config,tokenizer=tokenizer,beam_size=beam_size)
        else:
            raise NotImplementedError("Mode: {} not implemented yet!".format(generate_mode))
        
        #prepare input_str
        article_str, summary_str = extract_string_from_tensor(
            input_ids=input_ids, config = model_config,mode="pair",tokenizer=tokenizer)
        generated_summary_list= [""] * model_config.batch_size

        
        # tldr_str = "TL;DR:"
        # pad a <TL,DR;> token(<EOS>) after the string of Article.
        if TL_DR:
            for article_idx in range(model_config.batch_size):
                article_str[article_idx]+=(" "+tldr_str)
        
        # print("[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL article_str:")
        # print(article_str)

        generate_str_list, _ = generator.generate(
            input_str=article_str, generate_length=generate_length)

        # print("[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL generate_str_list:")
        # print(generate_str_list)
        
        for article_idx in range(model_config.batch_size):
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

                # increase search_index to add period token('.') if search_index does not overflow.
                search_index = search_index+1 if search_index < len_generate_str else len_generate_str
                generated_summary = generate_str[:search_index]
                if generated_summary.find(tokenizer.eos_token) != -1:
                    cut_pos = generated_summary.find(tokenizer.eos_token,0)
                    generated_summary = generated_summary[:cut_pos]

            else:
                generated_summary = generate_str

            #if all of str hs been clipped, restore it to beginning state.
            if generated_summary == '':
                generated_summary = generate_str  
            
            #empty str check
            if generated_summary == '':
                generated_summary = '<empty>'
            generated_summary_list[article_idx] = generated_summary

            # print("[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL debugging info:\nGENERATED_SUMMARY:")
            # print(generated_summary_list[article_idx])
            # print(summary_str[article_idx])

        return generated_summary_list, summary_str  # Hypo and Ref      

def generate_for_LAMBADA_numpy_topk(decoder, input_ids, logits, tokenizer, generate_length_dynamically=True, max_iterations=200, stop_word_file=None):
    """
    Args:
        input_ids(Tennor): input_ids(shape: (self.batch_size,self.seq_length)) of dataset which is sampled from mindrecord
        logits: (batch_size,seq_length,vocab_size) (8,1024,50257)
        max_generate_length(int): the number of tokens to generate

    Return:
        generated_last_word: generated the last word of lambada
    """


    generator = Sample(decoder,model_config=gpt2_net_cfg,tokenizer=tokenizer,topk_num=1,topp_prob=1,return_ids=True)
    #True if generated
    generate_batch_flag = [False]*gpt2_net_cfg.batch_size

    #All of the batches are generated 
    all_true = [True]*gpt2_net_cfg.batch_size

    # final_generations: lastword string list
    final_generations = ["" for _ in range(gpt2_net_cfg.batch_size)]        # ['','',',...'']

    stop_eos = ['.',',','!','?','"'," '"," and"," says"," said"]
    
    source_str = generator._extract_string_from_tensor(input_ids,mode="single")
    # print("source_string:",source_str)
    # print("*"*60)
    label_str = [ ' ' +str.split()[-1] for str in source_str]
    print("label_word: ",label_str[0])
    last_word_token_num = [len(tokenizer.encode(str)) for str in label_str][0]
    # print("label_string token num: ",last_word_token_num)
    # remove the last word
    source_str = [' '.join(str.split()[:-1]) for str in source_str]

    lastword_start_pos_ = get_lastword_range(input_ids = input_ids,config=gpt2_net_cfg,tokenizer=tokenizer)     # [(left_pos,right_pos)] -> batch_size for list length
    lastword_pos = []  # the previous token index of the last word
    for item in lastword_start_pos_:
        lastword_pos.append(item[0])
        # print("idx:",item[0] - 1)

    logits = extract_logits(logits = logits, seq_pos=lastword_pos)  #(8,1,50257)
    # print("last logits:",logits)

    logits = logits.asnumpy()
    logits = logits.reshape((-1,tokenizer.vocab_size))

    sorted_ids = np.argsort(-logits,axis=-1)[::,:max_iterations]
    sorted_ids = sorted_ids.T                               # [max_iterations,batch_size]
    # print("sortd ids T:",sorted_ids)
    sorted_ids = sorted_ids.tolist()                        # [[121,3,123,41],[3123,3123,43,12],...,]  (100,8)
    
    for i in range(max_iterations):
        # print("=============== iteration:{} ============== ".format(i))
        ids = sorted_ids[i]
        ids_str = [ tokenizer.decode([x]) for x in ids]
        cat_str = [x+y for x,y in zip(source_str,ids_str)]
        if generate_length_dynamically:
            generate_length =  last_word_token_num
        else:
            generate_length = 3 # default generate length
        generate_ids_list = generator.generate(input_str=cat_str, generate_length=generate_length,do_sample=False) # [[23,34,45],[34,56,79]... ]
        cat_ids_list = [[x]+y for x,y in zip(ids,generate_ids_list)]
        res_str_list = [tokenizer.decode(word) for word in cat_ids_list]       # [" hel lo <|endoftext|>","word ",...]
        
        for j in range(gpt2_net_cfg.batch_size):
            if generate_batch_flag[j]:
                continue

            generate_string = res_str_list[j]
            eos_pos = min( res_str_list[j].find(word) if res_str_list[j].find(word) >=0 else INF for word in stop_eos)

            if eos_pos == INF:
                continue
            else:
                res_str_list[j] = res_str_list[j][:eos_pos]


            res_str_list[j] = res_str_list[j].lstrip().rstrip()

            if res_str_list[j].find(" ") == -1 :     # don't have space in a word, set True
                # if res_str_list[j] == "":
                #     continue
                if is_stop_word(stop_word_file=stop_word_file,word=res_str_list[j].lower()):
                    continue
                
                generate_batch_flag[j] = True
                final_generations[j] = res_str_list[j] 
                # print("*"*50)
                print("generate_word:{}".format(res_str_list[j]))
                print("label_token_num:{}".format(label_token_num))
                print("gen_token_num:{}".format(gen_token_num))

        if all_true == generate_batch_flag:
            # print("Success")
            break

    return final_generations

if __name__ == '__main__':
    # s = Sample(None)

    pass


    
