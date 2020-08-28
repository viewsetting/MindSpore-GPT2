# sentence->tokenized->bpe encode
"""Byte pair encoding utilities"""

import os
import json
import regex as re
from functools import lru_cache


# 生成unicode词典 {97:“a”...}
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 把words做成一对一对的 传入"ABCD" 输出（('A', 'B'), ('B', 'C'), ('C', 'D')）
def get_pairs(word):
    """Return set of symbol pairs in a word.

    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class GPT2Tokenizer:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        self.encoder = encoder   #外部embedding词典
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.errors = errors  # how to handle errors in decoding
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

         # bpe_merges 是一个类似（Ġ t）（这里的这两个元素 是未来要用的 a b ） 元组 然后在用0123..的常用频率压缩起来成一个{（Ġ t）:1}
        self.bpe_ranks = dict(
            zip(bpe_merges,
                range(len(bpe_merges)))) 
        # bpe_merges里面是各种零散词的常用程度排名
        self.cache = {}

        # Should haved added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    # 用来识别未见过的词语 把词拆成各个词源 例如:输入greenhand 输出"green hand"
    def bpe(self, token):
        # 如果dict（self.cache）中有token的key 那就返回对应的值(有缓存结果)
        if token in self.cache:
            return self.cache[token]

        word = tuple(token)  # 把list 转成 tuple
        # 下面就是把一个词，拆散了 输入（find） 输出(（f,i）,(i,n),(n,d)) 注意返回set无序
        pairs = get_pairs(word)

        # 词很短 拆不了 直接返回 token
        if not pairs:
            return token

        # 迭代所有的pairs 中的词对
        while True:  # lambda 迭代对象:对应表达式
            # 将输入的pairs 按照.bpe文件 （常用排名）排序 这里的pair 就是之前提到的a b
            # 找到最常用的哪个 pair float('inf') 表示无穷大 找不到的话 就返回无限大的值 以免被选上
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))  # MIN MAX 中key 相当于依据什么排序

            # 组合不在bpe表格中 pairs中不能再拆了 循环结束
            if bigram not in self.bpe_ranks:
                break

            # 拿到第一个词 第二个词
            first, second = bigram  # 拿到拆开的的对里 在表格里最常用的那一对
            
            new_word = []
            
            i = 0
            #  查找子串
            while i < len(word):

                try:
                    j = word.index(first, i)  # i指的是从第I个开始查找  #查找list.index(x,起始位置,终止位置) #从传入的word里 查找第一个单词
                    # 这里的意思是 因为pair 是无序的 要找到其在输入词中的顺序
                    new_word.extend(word[i:j])  # 将这个子串 first=word[i:j] 放入new_word变量中
                    i = j  # 移动指针
                except:
                    new_word.extend(word[i:])  # 当J越界时候 直接将 i: 切片放进去
                    break

                # 这里的意思是 如果first 和 second 这两个是连着的话 加到new_word时候 是一个整体
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                    # 否则的话 只加入word[i] 
                else:
                    new_word.append(word[i])
                    i += 1
                    
            #类似于递归查找
            new_word = tuple(new_word)
            word = new_word
            
            #串不能再拆了
            if len(word) == 1:
                break
            else:
                #拆开再找一遍
                pairs = get_pairs(word)
        #用空格链接所有的词 
        word = ' '.join(word)
        #增加这个词的缓存，以后再碰到就不用运算了
        self.cache[token] = word
        return word

    # Tokenize a string.  text->bpe_tokens
    def encode(self, text):
        bpe_tokens = []
        #self.pat .findall text 的意思是从text 中 把self.pat这里pattern找出来 其实就是she's 变成 she s两个单词
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            #上面一句大致等价于 token = unicode(token, "utf-8") #将文字转成utf-8后 用self.byte_encoder——bytes_to_unicode()产生的dict 转回字符形式 然后将其连城字符串
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
            #将拆完的词 在传入的embedding字典中查找，返回这个列表
        return bpe_tokens
    
    # tokens->text
    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


# 获取GPT2Tokenizer对象，参数：词典路径
def get_GPT2Tokenizer(file_path):
    with open(os.path.join(file_path, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    with open(os.path.join(file_path,  'vocab.txt'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    return GPT2Tokenizer(encoder,bpe_merges)

if __name__ == '__main__':
    unk_token="<|endoftext|>"
    errors="replace"
    vocab_file = r"F:\code\python\TJU\GPT-2\data\encoder.json"
    with open(vocab_file, encoding="utf-8") as vocab_handle:
        encoder = json.load(vocab_handle)
    decoder = {v: k for k, v in encoder.items()}
    merges_file =r"F:\code\python\TJU\GPT-2\data\gpt2-merges.txt"
    with open(merges_file, encoding="utf-8") as merges_handle:
        bpe_merges = merges_handle.read().split("\n")[1:-1]
    bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
    bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
    #创建GPT2Tokenizer对象
    tokenizer = GPT2Tokenizer(encoder,bpe_merges)
    
    res = tokenizer.encode("howareyou")  
    print(res)   #[4919, 533, 5832]

    res1 = tokenizer.decode([4919, 533, 5832])  # h
    print(res1)   #howareyou

    tokenizer1 = get_GPT2Tokenizer("F:\code\python\TJU\GPT-2\data")
    res2 = tokenizer1.encode("howareyou")
    print(res2)
