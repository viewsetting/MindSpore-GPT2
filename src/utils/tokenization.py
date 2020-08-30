import json
import regex as re
from functools import lru_cache

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
        with open(merge_file,'r',encoding="utf-8") as merge_handle:
            bpe_merges = merge_handle.read().split('\n')[1:-1]

        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]

        self.bpe_ranks = dict(zip(bpe_merges,range(len(bpe_merges))))
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
        
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        self.add_prefix_space = add_prefix_space
        self.cache = {}
        self.add_prefix_space = add_prefix_space

        self.unk_token = "<|endoftext|>"

    
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
        return self.encoder.get(token,self.encoder.get(self.unk_token))


    def _convert_id_to_token(self, id):
        """ return the orgin bpe token according to id"""   
        return self.decoder.get(id)


    def _convert_tokens_to_string(self, tokens):
        """ return a string according to the list of tokens"""       
        text = "".join(tokens)
        text = bytearray([self.byte_decoder[c] for c in text]).decode("utf-8")
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


def Tokenizer():
    """ use the GPT2Tokenizer"""
    vocab_file = "./pretrain-data/gpt2-vocab.json"
    merge_file = "./pretrain-data/gpt2-merges.txt"

    tokenizer = GPT2Tokenizer(vocab_file, merge_file,add_prefix_space = False)

    return tokenizer

  
"""  ******** Example ********

>>> example_text = 'How are you!'
>>> tokenizer = Tokenizer()
>>> ids = tokenizer.encode(example_text)
>>> print(ids)
    [2437, 389, 345, 0]
>>> print(tokenizer.decode(ids))
    How are you!

"""