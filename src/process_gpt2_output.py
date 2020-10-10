from typing import List, Optional, Tuple
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


def generate(
        model=None,
        config=None,
        input_ids: Optional[Tensor] = None,
        input_mask: Optional[Tensor] = None,
        max_length: Optional[int] = 1024,
        min_length: Optional[int] = 200,
        do_sample: Optional[bool] = False,
        early_stopping: Optional[bool] = False,
        num_beams: Optional[int] = 1,
        temperature: Optional[float] = 1.0,
        top_k: Optional[int] = 50,
        top_p: Optional[float] = 1.0,
        repetition_penalty: Optional[float] = 1.0,
        bos_token_id: Optional[int] = 50256,
        pad_token_id: Optional[int] = 50256,
        eos_token_id: Optional[int] = 50256,
        length_penalty: Optional[float] = 1.0,
        no_repeat_ngram_size: Optional[int] = 0,
        num_return_sequences: Optional[int] = 1,
        attention_mask: Optional[Tensor] = None,
        use_cache: Optional[bool] = True,
):
    r"""
    Generates sequences for models with a language modeling head. The method currently supports greedy decoding,
    beam-search decoding, sampling with temperature, sampling with top-k or nucleus sampling.
    Args:
        config: the config of gpt2 model which you want to use to generate.
        input_ids (Tensor): shape with (batch_size, seq_length)
        max_length (int): The maximum length of the sequence to be generated.
        min_length: The minimum length of the sequence to be generated.
        do_sample: Whether or not to use sampling ; use greedy decoding otherwise.
        early_stopping: Whether to stop the beam search when at least ``num_beams`` sentences are finished per batch or not.
        num_beams: Number of beams for beam search. 1 means no beam search.
        temperature: The value used to module the next token probabilities.
        top_k: The number of highest probability vocabulary tokens to keep for top-k-filtering.
        top_p: If set to float < 1, only the most probable tokens with probabilities that add up to ``top_p`` or higher are kept for generation.
        repetition_penalty: Default 1.0 .The parameter for repetition penalty. 1.0 means no penalty. See `this paper
                            <https://arxiv.org/pdf/1909.05858.pdf>`__ for more details.
        bos_token_id: The id of the `padding` token.
        pad_token_id: The id of the `beginning-of-sequence` token.
        eos_token_id: The id of the `end-of-sequence` token.
        length_penalty: Exponential penalty to the length. 1.0 means no penalty. Default: 1.0.
        no_repeat_ngram_size: If set to int > 0, all ngrams of that size can only occur once. Default: 0.
        num_return_sequences: The number of independently computed returned sequences for each element in the batch. Default: 1.
        attention_mask: shape with (batch_size, seq_length)
                        Mask to avoid performing attention on padding token indices. Mask values are in ``[0, 1]``, 1 for
                        tokens that are not masked, and 0 for masked tokens.
        use_cache: Whether or not the model should use the past last key/values attentions (if applicable to the model) to
                   speed up decoding. Default: True .

    Returns:
        Tensor of shape (batch_size * num_return_sequences, seq_length)
        The generated sequences. The second dimension (sequence_length) is either equal to :obj:`max_length` or shorter
        if all batches finished early due to the :obj:`eos_token_id`.

    """

    if input_ids is not None:
        batch_size, seq_len = P.Shape()(input_ids)
    else:
        batch_size = 1

    assert model is not None, "model should not be a None object."
    assert config is not None, "config of gpt2_model is a must input param."
    assert isinstance(max_length, int) and max_length > 0, "`max_length` should be a strictly positive integer."
    assert isinstance(min_length, int) and min_length >= 0, "`min_length` should be a positive integer."
    assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
    assert isinstance(early_stopping, bool), "`early_stopping` should be a boolean."
    assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
    assert isinstance(num_beams, int) and num_beams > 0, "`num_beams` should be a strictly positive integer."
    assert temperature > 0, "`temperature` should be strictly positive."
    assert isinstance(top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
    assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
    assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
    assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
    ), "If input_ids is not defined, `bos_token_id` should be a positive integer."
    assert pad_token_id is None or (
            isinstance(pad_token_id, int) and (pad_token_id >= 0)
    ), "`pad_token_id` should be a positive integer."
    assert (eos_token_id is None) or (
            isinstance(eos_token_id, int) and (eos_token_id >= 0)
    ), "`eos_token_id` should be a positive integer."
    assert length_penalty > 0, "`length_penalty` should be strictly positive."
    assert (
            isinstance(no_repeat_ngram_size, int) and no_repeat_ngram_size >= 0
    ), "`no_repeat_ngram_size` should be a positive integer."
    assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
    ), "`num_return_sequences` should be a strictly positive integer."
    # not allow to duplicate outputs when greedy decoding
    if do_sample is False:
        if num_beams == 1:
            # no_beam_search greedy generation conditions
            assert (
                    num_return_sequences == 1
            ), "Greedy decoding will always produce the same output for num_beams == 1 and num_return_sequences > 1. Please set num_return_sequences = 1"

        else:
            # beam_search greedy generation conditions
            assert (
                    num_beams >= num_return_sequences
            ), "Greedy beam search decoding cannot return more sequences than it has beams. Please set num_beams >= num_return_sequences"

    assert attention_mask is not None, "`attention_mask` should be provided。"
    vocab_size = config.vocab_size

    # set effective batch size and effective batch multiplier according to do_sample
    if do_sample:
        effective_batch_size = batch_size * num_return_sequences
        effective_batch_mult = num_return_sequences
    else:
        effective_batch_size = batch_size
        effective_batch_mult = 1

    if num_return_sequences > 1 or num_beams > 1:
        expand_shape = (batch_size, effective_batch_mult * num_beams, seq_len)
        broadcast_to = P.BroadcastTo(expand_shape)

        input_ids = P.ExpandDims()(input_ids, 1) # [batch_size, 1, seq_len]
        input_ids = broadcast_to(input_ids)

        attention_mask = P.ExpandDims()(attention_mask, 1)
        attention_mask = broadcast_to(attention_mask)

        input_ids = P.Reshape()(input_ids, (effective_batch_size * num_beams, seq_len))
        # shape: (batch_size * num_return_sequences * num_beams, cur_len)
        attention_mask = P.Reshape()(attention_mask, (effective_batch_size * num_beams, seq_len))
        # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        cur_len = seq_len
        assert (cur_len < max_length), f"The context has {cur_len} number of tokens, but `max_length` is only {max_length}. Please make sure that `max_length` is bigger than the number of tokens, by setting either `generate(max_length=...,...)` or `config.max_length = ...`"

        if num_beams > 1:
            
            output = generate_beam_search(
                model=model,
                config=config,
                input_ids=input_ids,
                input_mask=input_mask,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                #batch_size=effective_batch_size,
                #num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                #vocab_size=vocab_size,
                #attention_mask=attention_mask,
                use_cache=use_cache,
            )
            
        else:
            '''
            output = generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                attention_mask=attention_mask,
                use_cache=use_cache,
            )
            '''

def generate_no_beam_search(
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        pad_token_id,
        eos_token_id,
        batch_size,
        attention_mask,
        use_cache,
):
    raise NotImplementedError('not implemented yet')
    past = None
    while cur_len < max_length:
        pass


def generate_beam_search(
        model,
        config,
        input_ids,
        input_mask,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        pad_token_id,
        eos_token_id,
        #batch_size,
        length_penalty,
        num_beams:int,
        #attention_mask,
        use_cache
):
    generated_ids = []

    max_length = min(max_length,config.seq_length)
    batch_size = config.batch_size
    vocab_size = config.vocab_size
    assert batch_size == 1, "For now, it only generates 1 sentence per batch."

    #initialize beam_score as 0 tensor
    init_beam_prob = np.zeros((batch_size,num_beams),dtype=float)
    

    reshape = P.Reshape()
    squeeze_shape = (-1,)
    top_k = P.TopK(sorted=False)


    if do_sample is False:
        init_beam_prob[:,1:] = -1e9

    # beam_scores in form of Tensor:
    # beam_scores = Tensor(init_beam_prob,dtype=mstype.float32)
    # beam_scores = reshape(beam_scores,squeeze_shape)

    #Use numpy for now, since batch size is only 1
    beam_scores = init_beam_prob
    #beam_scores: shape (batch_size*num_beams,)

    #cache states
    past_states = None

    done_sentences = [False for _ in range(batch_size)]

    input_ids_expand = replicate_input(input_ids,time=num_beams)
    log_softmax = P.LogSoftmax(axis = -1)

    first_token = True
    
    while cur_len < max_length:
        lst_logits = []
        generated_ids.append([])
        for i in range(num_beams):
            lst_logits.append( model.predict(input_ids_expand,input_mask))
        
        tuple_logits = tuple(lst_logits)
        concat = P.Concat(axis = 0)

        #concat from tuple of logits
        logits = concat(tuple_logits)

        next_token_logits = logits[::,cur_len,0:vocab_size]
        # (num_beams,vocab_size)

        scores = log_softmax(next_token_logits)
        
        candidates = None
        sentence_prefix = None

        squeezed_scores = reshape(scores,squeeze_shape)
        #(num_beam*vocab_size)

        #indices_np = None
        if first_token :
            first_token = False
            values,indices = top_k(squeezed_scores[0:vocab_size],num_beams)
            #indices (num_beams)


            indices_np = indices.asnumpy()
            values_np = indices.asnumpy()
            candidates = indices_np.tolist()
            #for the first token, we choose 0 as default for all situations since the model is not .
            sentence_prefix = [ 0 for _ in range(num_beams)]
            for i in range(num_beams):
                beam_scores[i] += values_np[i]
                generated_ids[-1].append(candidates[i])
                           

        else:
            # need to choose top beams^2 prob of token
            values,indices = top_k(squeezed_scores,num_beams*num_beams)
            indices_np = indices.asnumpy()

            indices_np = indices.asnumpy()
            values_np = indices.asnumpy()

            tmp_candidates = indices_np.tolist()
            tmp_candidates_scores = []
            for i in range(num_beams*num_beams):
                sentence_index = indices_np[i]//vocab_size
                # index of token, tmp_beam_score, sentence_index of token
                tmp_candidates_scores.append((tmp_candidates[i]%vocab_size,values_np[i]+beam_scores[sentence_index],sentence_index))
            
            #sort by beam_score
            tmp_candidates_scores.sort(key=lambda x:x[1],reverse=True)

            sentence_prefix = []
            candidates = []
            for i in range(num_beams):
                sentence_prefix.append(tmp_candidates_scores[i][2])
                candidates.append(tmp_candidates_scores[i][0])
                beam_scores[i] += tmp_candidates_scores[i][1]
            
        input_np = input_ids_expand.asnumpy()
            #(num_beams,seq_length)
        new_input = np.zeros_like(input_np)
            
        for i in range(num_beams):
            new_input[i] = input_np[sentence_prefix[i]]
            new_input[i][cur_len] = candidates[i]
            generated_ids[-1].append(candidates[i])
        
        input_ids_expand = Tensor(input_np,dtype = mstype.float32)
        
        cur_len += 1
        pass

    #(seq_length,num_beams) -> (num_beams,seq_length)
    generated_ids_np = np.array(generated_ids).T
    token_ids = generated_ids_np.tolist()

    return token_ids[0]


def top_k_top_p_filtering(
    logits: Tensor,
    top_k: int = 0,
    top_p: float = 1.0,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> Tensor:

    raise NotImplementedError



'''
Replicate input_ids from (batch_size,seq_length) --> (batch_size*time,seq_length)
'''
def replicate_input(input_ids:Tensor,time:int):
    tile = P.Tile()
    replicate_shape = (time,1)
    ret = tile(input_ids,replicate_shape)

    return ret 
    
      
        