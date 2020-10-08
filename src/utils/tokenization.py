import json
import regex as re
from functools import lru_cache
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
import logging
from enum import Enum

logger = logging.getLogger(__name__)

# class ExplicitEnum(Enum):
#     """
#     Enum with more explicit error message for missing values.
#     """
#
#     @classmethod
#     def _missing_(cls, value):
#         raise ValueError(
#             "%r is not a valid %s, please select one of %s"
#             % (value, cls.__name__, str(list(cls._value2member_map_.keys())))
#         )
#
# class TruncationStrategy(ExplicitEnum):
#     """
#     Possible values for the ``truncation`` argument.
#     """
#     ONLY_FIRST = "only_first"
#     LONGEST_FIRST = "longest_first"
#     DO_NOT_TRUNCATE = "do_not_truncate"
#
# class PaddingStrategy(ExplicitEnum):
#     """
#     Possible values for the ``padding`` argument.
#     """
#
#     LONGEST = "longest"
#     MAX_LENGTH = "max_length"
#     DO_NOT_PAD = "do_not_pad"


@lru_cache()
def bytes_to_unicode():

    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(i) for i in cs]
    return dict(zip(bs, cs))

def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer():
    def __init__(
        self,
        vocab_file,
        merge_file,
        add_prefix_space=False,
    ):
        with open(vocab_file,'r',encoding="utf-8") as vocab_handle:
            self.encoder = json.load(vocab_handle)
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.vocab_size = len(self.decoder)
        with open(merge_file,'r',encoding="utf-8") as merge_handle:
            bpe_merges = merge_handle.read().split('\n')[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]

        self.bpe_ranks = dict(zip(bpe_merges,range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.add_prefix_space = add_prefix_space
        self.cache = {}

        self.unk_token = "<|endoftext|>"
        self.unk_token_id = 50256
        self.bos_token = "<|endoftext|>"
        self.bos_token_id = 50256
        self.eos_token = "<|endoftext|>"
        self.eos_token_id = 50256
        self.pad_token = "<|endoftext|>"
        self.pad_token_id = 50256


    def bpe(self, token):
        
        if token in self.cache:
            return self.cache[token]
        
        word = tuple(token)
        pairs = get_pairs(token) 
        if not pairs:
            return token
            
        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair,float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first,second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first,i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j
                    
                if word[i] == first and i+1 < len(word) and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def _tokenize(self, text):
        """ Tokenize a string using bpe encode. """
        text = self.prepare_for_tokenization(text,is_pretokenized = False)
        print(text)
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(
                self.byte_encoder[b] for b in token.encode("utf-8")
            )
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens
    

    def _convert_token_to_id(self, token):
        """ the index of the token in the vocabulary. """       
        return self.encoder.get(token, self.encoder.get(self.unk_token))


    def _convert_id_to_token(self, id):
        """ return the orgin bpe token according to id"""   
        return self.decoder.get(id)


    def _convert_tokens_to_string(self, tokens):
        """ return a string according to the list of tokens"""       
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8",errors='ignore')
        return text  


    def encode(self, text):
        """ get the index list of text"""        
        text_id = []
        bpe_tokens = self._tokenize(text)
        for token in bpe_tokens:
            text_id.append(self._convert_token_to_id(token))
        return text_id
    

    def decode(self, ids):
        """ return a string according to the index list of tokens"""
        tokens = []
        for id_ in ids:
            tokens.append(self._convert_id_to_token(id_))
        return self._convert_tokens_to_string(tokens)
        

    def prepare_for_tokenization(self, text, is_pretokenized=False, **kwargs):
        """ whether to add a whitespace in the front of text """        
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_pretokenized or add_prefix_space:
            text = " " + text
        return text

    def add_special_tokens(self, special_tokens_dict):
        """
        Add a dictionary of special tokens (eos, pad, cls, etc.) to the encoder and link them to class attributes. If
        special tokens are NOT in the vocabulary, they are added to it (indexed starting from the last index of the
        current vocabulary).
        Args:
            special_tokens_dict (dictionary `str` to `str`):
                Keys should be in the list of predefined special attributes: [``bos_token``, ``eos_token``,
                ``unk_token``, ``sep_token``, ``pad_token``, ``cls_token``, ``mask_token``,
                ``additional_special_tokens``].

        Returns:
            added_tokens (int): Number of tokens added to the vocabulary

        """
        # special_tokens_dict = {'cls_token': '<CLS>'}
        if not special_tokens_dict:
            return 0

        added_tokens = 0
        for key, value in special_tokens_dict.items():
            setattr(self, key, value)
            assert isinstance(value, str), f"Token {value} for key {key} should be a str instance"
            added_tokens += self.add_tokens([value], special_tokens=True)
        return added_tokens

    def add_tokens(self, new_tokens, special_tokens=False):
        if not new_tokens:
            return 0
        if not isinstance(new_tokens, (list, tuple)):
            new_tokens = [new_tokens]
        return self._add_tokens(new_tokens, special_tokens=special_tokens)

    def _add_tokens(self, new_tokens, special_tokens=False):
        """

        Args:
            new_tokens (list[str]): Token(s) to add in vocabulary.
            special_tokens (bool): Whether or not the tokens should be added as special tokens.

        Returns:
            the number of the new added tokens.
        """
        new_tokens = [str(token) for token in new_tokens]

        tokens_to_add = []
        for token in new_tokens:
            assert isinstance(token, str)
            tokens_to_add.append(token)
            logger.info("Adding %s to the vocabulary ! ", token)

        added_tok_encoder = dict((tok, self.vocab_size + i)for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.encoder.update(added_tok_encoder)
        self.decoder.update(added_tok_decoder)
        return len(tokens_to_add)

    def num_special_tokens_to_add(self, pair: bool = False):
        token_ids_0 = []
        token_ids_1 = []
        return len(self.build_inputs_with_special_tokens(token_ids_0, token_ids_1 if pair else None))


    def build_inputs_with_special_tokens(self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None):
        """
        Build model inputs from a sequence or a pair of sequence by concatenating and adding special tokens.

        A GPT2 sequence has the following format:
        - single sequence: ``<bos> X <eos>``
        - pair of sequences: ``<bos> A <eos> B <eos>``

        Args:
            token_ids_0 (List[int]): List of IDs to which the special tokens will be added
            token_ids_1 (List[int], `optional`, defaults to `None`): Optional second list of IDs for sequence pairs.
        """
        bos = [self.bos_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return bos + token_ids_0 + eos
        return bos + token_ids_0 + eos + token_ids_1 + eos

    def truncate_sequences(self, ids, num_tokens_to_remove, truncation_strategy="ONLY_FIRST", direction="RIGHT"):
        if num_tokens_to_remove <= 0:
            return ids, []

        overflowing_tokens = []
        if truncation_strategy == "ONLY_FIRST":
            if len(ids) > num_tokens_to_remove:
                if direction == "RIGHT":
                    overflowing_tokens = ids[-num_tokens_to_remove:]
                    ids = ids[:-num_tokens_to_remove]
                if direction == "LEFT":
                    overflowing_tokens = ids[:num_tokens_to_remove]
                    ids = ids[num_tokens_to_remove:]
            else:
                logger.error(f"We need to remove {num_tokens_to_remove} to truncate the input"
                             f"but the first sequence has a length {len(ids)}. ")
        else:
            logger.error(f"Please select correct truncation strategy, for instance 'ONLY_FIRST'")
        return (ids, overflowing_tokens)

    def _pad(self, encoded_inputs, max_length=None, padding_strategy=None, return_attention_mask:Optional[bool]=None):
        needs_to_be_padded = (len(encoded_inputs["input_ids"]) != max_length)
        if needs_to_be_padded:
            if padding_strategy == "MAX_LENGTH":
                difference = max_length - len(encoded_inputs["input_ids"])
                if return_attention_mask:
                    encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"]) + [0] * difference
                    encoded_inputs["input_ids"] = encoded_inputs["input_ids"] + [self.pad_token_id] * difference
            else:
                raise ValueError("Invalid padding strategy")
        else:
            if return_attention_mask:
                encoded_inputs["attention_mask"] = [1] * len(encoded_inputs["input_ids"])

        return encoded_inputs

    def pad(self, encoded_inputs, max_length:Optional[int] = None, padding_strategy="MAX_LENGTH", return_attention_mask=True):
        # no batch encoded_inputs["input_ids"]--->[98, 67, 32388, 318, 1912, 287, 170, 8496, 318, 905, 2667, 32]
        if encoded_inputs["input_ids"] and not isinstance(encoded_inputs["input_ids"][0], (list, tuple)):
            encoded_inputs = self._pad(
                encoded_inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask
            )
            return encoded_inputs

        # encoded_inputs with batch_size
        batch_size = len(encoded_inputs["input_ids"])
        assert all(
            len(v) == batch_size for v in encoded_inputs.values()
        ), "Some items in the output dictionary have a different batch size than others."

        if padding_strategy == "LONGEST":
            max_length = max(len(inputs) for inputs in encoded_inputs["input_ids"])
            padding_strategy = "MAX_LENGTH"

        batch_outputs = {}
        for i in range(batch_size):
            inputs = dict((k, v[i]) for k, v in encoded_inputs.items())
            outputs = self._pad(
                encoded_inputs=inputs,
                max_length=max_length,
                padding_strategy=padding_strategy,
                return_attention_mask=return_attention_mask
            )
            for key, value in outputs.items():
                if key not in batch_outputs:
                    batch_outputs[key] = []
                batch_outputs[key].append(value)

        return batch_outputs


    def prepare_for_model(self,
                          ids,
                          pair_ids=None,
                          add_special_tokens=True,
                          max_length=None,
                          padding=None,
                          return_overflowing_tokens=False,
                          return_attention_mask=True):

        pair = bool(pair_ids is not None)
        len_ids = len(ids)
        len_pair_ids = len(pair_ids) if pair else 0

        encoded_inputs = {}
        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids + (self.num_special_tokens_to_add(pair=pair) if add_special_tokens else 0)

        # Truncation: Handle max sequence length
        if max_length and total_len > max_length:
            ids, overflowing_tokens = self.truncate_sequences(ids=ids,
                                                              num_tokens_to_remove=total_len - max_length,
                                                              truncation_strategy="ONLY_FIRST",
                                                              direction="RIGHT")
            if return_overflowing_tokens:
                encoded_inputs["overflowing_tokens"] = overflowing_tokens
                encoded_inputs["num_truncated_tokens"] = total_len - max_length

        if add_special_tokens:
            sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        else:
            sequence = ids + pair_ids if pair else ids

        # build output dictionary
        encoded_inputs["input_ids"] = sequence
        # check lengths
        if max_length is None or len(encoded_inputs["input_ids"]) > max_length:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum sequence length "
                "for this model ({} > {}). Running this sequence through the model will result in "
                "indexing errors".format(len(ids), max_length)
            )
        # padding
        if padding or return_attention_mask:
            encoded_inputs = self.pad(encoded_inputs=encoded_inputs,
                                      max_length=max_length,
                                      padding_strategy="MAX_LENGTH",
                                      return_attention_mask=return_attention_mask)

        return encoded_inputs

def Tokenizer():
    """ use the GPT2Tokenizer"""
    vocab_file = "./pretrain-data/gpt2-vocab.json"
    merge_file = "./pretrain-data/gpt2-merges.txt"

    tokenizer = GPT2Tokenizer(vocab_file, merge_file,add_prefix_space = False)
    return tokenizer

# if __name__=='__main__':
#     tokenizer = Tokenizer()
#     text1 = "With almost everything else to make them happy , they wanted one thing : they had no children .This vexed"
#     text2 = "This is a rather long sequence. It is at least longer than the sequence A."
#     ids = tokenizer.encode(text1)
#     print("ids: {}".format(ids))
#     pair_ids = tokenizer.encode(text2)
#     print("pair_ids: {}".format(pair_ids))
#     output = tokenizer.prepare_for_model(ids=ids,
#                                          pair_ids=pair_ids,
#                                          add_special_tokens=True,
#                                          max_length=50,
#                                          padding=True,
#                                          return_overflowing_tokens=True,
#                                          return_attention_mask=True)
#     print(output)
#     print(tokenizer.decode(output["input_ids"]))
#     print(len(output["input_ids"]))
#     # output = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
#     # print(ids)
#     # print(output)
#     # print(tokenizer.decode(output))
#
#     # print(len(tokenizer.encode(text)))
#     # print(tokenizer.pad_token)
  
"""  ******** Example ********

>>> example_text = 'How are you!'
>>> tokenizer = Tokenizer()
>>> ids = tokenizer.encode(example_text)
>>> print(ids)
    [2437, 389, 345, 0]
>>> print(tokenizer.decode(ids))
    How are you!

"""