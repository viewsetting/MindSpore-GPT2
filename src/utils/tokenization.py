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

        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k, v in self.byte_encoder.items()}
        self.unique_no_split_tokens = ["<|endoftext|>"]  # List[str]
        
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
        """
        Tokenize a string using bpe encode.

        Args:
            text (str): The sequence to be encoded.

        Returns:
            bpe_tokens (List[str]): The list of tokens.
        """
        text = self.prepare_for_tokenization(text, is_pretokenized=False)
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(bpe_token for bpe_token in self.bpe(token).split(" "))
        return bpe_tokens

    def tokenize(self, text):
        def split_on_token(tok, text):
            result = []
            split_text = text.split(tok)
            for i, sub_text in enumerate(split_text):
                if i < len(split_text) - 1:
                    sub_text = sub_text.rstrip()
                if i > 0:
                    sub_text = sub_text.lstrip()
                if i == 0 and not sub_text:
                    result += [tok]
                elif i == len(split_text) - 1:
                    if sub_text:
                        result += [sub_text]
                    else:
                        pass
                else:
                    if sub_text:
                        result += [sub_text]
                    result += [tok]
            return result

        def split_on_tokens(tok_list, text):
            if not text.strip():
                return []
            if not tok_list:
                return self._tokenize(text)

            tokenized_text = []
            text_list = [text]
            for tok in tok_list:
                tokenized_text = []
                for sub_text in text_list:
                    if sub_text not in self.unique_no_split_tokens:
                        tokenized_text += split_on_token(tok, sub_text)
                    else:
                        tokenized_text += [sub_text]
                text_list = tokenized_text

            bpe_token = []
            for token in text_list:
                if token not in self.unique_no_split_tokens:
                    bpe_token += self._tokenize(token)
                else:
                    bpe_token += [token]
            return bpe_token

        no_split_token = self.unique_no_split_tokens
        tokenized_text = split_on_tokens(no_split_token, text)
        return tokenized_text

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
        bpe_tokens = self.tokenize(text)
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
        print("here:",new_tokens)

        tokens_to_add = []
        for token in new_tokens:
            print(token)
            assert isinstance(token, str)
            if (token != self.unk_token
                and self._convert_token_to_id(token) == self._convert_token_to_id(self.unk_token)
                and token not in tokens_to_add
            ):
                tokens_to_add.append(token)
                logger.info("Adding %s to the vocabulary ! ", token)

        added_tok_encoder = dict((tok, self.vocab_size + i)for i, tok in enumerate(tokens_to_add))
        added_tok_decoder = {v: k for k, v in added_tok_encoder.items()}
        self.encoder.update(added_tok_encoder)
        self.decoder.update(added_tok_decoder)

        if special_tokens:
            self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(new_tokens)))
        else:
            self.unique_no_split_tokens = sorted(set(self.unique_no_split_tokens).union(set(tokens_to_add)))
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


class CNN_DailyMail_tokenizer(GPT2Tokenizer):
    def prepare_for_model(self,
                          ids,
                          pair_ids,
                          max_length=1024,
                          max_summary_length = 150,
                          add_special_tokens=True,
                          padding=None,
                          return_overflowing_tokens=False,
                          return_attention_mask=True):

        # pair = bool(pair_ids is not None)
        # assert (ids is None ) or  (pair_ids is None),"ids and pair_ids can not be None at the same time."

        len_ids = len(ids)
        len_pair_ids = len(pair_ids)

        encoded_inputs = {}
        # Compute the total size of the returned encodings
        total_len = len_ids + len_pair_ids

        ids_overflowing_tokens = []
        pair_overflowing_tokens = []
        # Truncation: Handle max sequence length
        if total_len > max_length-3:
            if len_pair_ids > max_summary_length:
                pair_ids, pair_overflowing_tokens = self.truncate_sequences(ids=pair_ids,
                                                                            num_tokens_to_remove=len_pair_ids - max_summary_length,
                                                                            truncation_strategy="ONLY_FIRST",
                                                                            direction="RIGHT")
                if len_ids+max_summary_length > max_length-3:
                    ids, ids_overflowing_tokens = self.truncate_sequences(ids=ids,
                                                                          num_tokens_to_remove=(len_ids+max_summary_length) - (max_length-3),
                                                                          truncation_strategy="ONLY_FIRST",
                                                                          direction="RIGHT")
            else:
                ids, ids_overflowing_tokens = self.truncate_sequences(ids=ids,
                                                                      num_tokens_to_remove=total_len - (max_length-3),
                                                                      truncation_strategy="ONLY_FIRST",
                                                                      direction="RIGHT")
            if return_overflowing_tokens:
                    encoded_inputs["article_overflowing_tokens"] = ids_overflowing_tokens
                    encoded_inputs["highlights_overflowing_tokens"] = pair_overflowing_tokens
                    encoded_inputs["num_truncated_tokens"] = total_len - (max_length-3)

        
        sequence = self.build_inputs_with_special_tokens(ids, pair_ids)
        seq_len = len(sequence)

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


class CBT_tokenizer(GPT2Tokenizer):

    def prepare_for_model(self,
                          ids,
                          pair_ids=None,
                          add_special_tokens=True,
                          max_length=None,
                          padding=None,
                          return_overflowing_tokens=False,
                          return_attention_mask=True,
                          num_choice=None):

        pair = bool(pair_ids is not None)
        len_pair_ids = len(pair_ids) if pair else 0
        input_ids = []
        attention_mask = []
        encoded_inputs = {}
        final_encoded_inputs = {}

        for i in range(num_choice):
            single_ids = ids[i]
            len_ids = len(ids)
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
            if max_length is None or len(sequence) > max_length:
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

            input_ids.append(encoded_inputs["input_ids"])
            attention_mask.append(encoded_inputs["attention_mask"])

        final_encoded_inputs["input_ids"] = input_ids
        final_encoded_inputs["attention_mask"] = attention_mask
        return final_encoded_inputs


def Tokenizer(vocab_file="./pretrain-data/gpt2-vocab.json", merge_file="./pretrain-data/gpt2-merges.txt", mode="normal"):
    """ use the GPT2Tokenizer"""
    #vocab_file = "./pretrain-data/gpt2-vocab.json"
    #merge_file = "./pretrain-data/gpt2-merges.txt"
    if mode == "normal":
        tokenizer = GPT2Tokenizer(vocab_file, merge_file, add_prefix_space=False)
    elif mode == "cnn_dailymail":
        tokenizer = CNN_DailyMail_tokenizer(vocab_file, merge_file, add_prefix_space=False)
    elif mode == "cbt":
        tokenizer = CBT_tokenizer(vocab_file, merge_file, add_prefix_space=False)
    else:
        raise ValueError("No Such Mode for {} in src.utils.tokenization.Tokenizer()".format(mode))
    return tokenizer


# if __name__=='__main__':
    # tokenizer = Tokenizer()
#     number = tokenizer.add_special_tokens({'cls_token': '[CLS]'})
#     text = "<|endoftext|>This is a rather long sequence. It is at least longer than the sequence<|endoftext|>"
    # text = "<|endoftext|>This is a rather long sequence. It is at least longer than the sequence [CLS]<|endoftext|>"
    # t = tokenizer.encode(text)
    # print(t)
    # text1 = "With almost everything else to make them happy" + " " + tokenizer.eos_token
    # ids = tokenizer.encode(text)
    # ids_copy = tokenizer.encode_copy(text)
    # print("ids: {}".format(ids))
    # print("ids copy: {}".format(ids_copy))
#     print(tokenizer.encode(text))
#     print(tokenizer.decode([1279, 91, 437, 1659, 5239, 91, 29]))
#     print(tokenizer.decode([50256]))
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
    # output = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    # print(ids)
    # print(output)
    # print(tokenizer.decode(output))

    # print(len(tokenizer.encode(text)))
    # print(tokenizer.pad_token)
  
"""  ******** Example ********

>>> example_text = 'How are you!'
>>> tokenizer = Tokenizer()
>>> ids = tokenizer.encode(example_text)
>>> print(ids)
    [2437, 389, 345, 0]
>>> print(tokenizer.decode(ids))
    How are you!

"""