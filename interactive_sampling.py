import os
import numpy as np
from typing import TypeVar, Union
from src.finetune_eval_config import cfg, gpt2_net_cfg
from mindspore import log as logger
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.utils.tokenization import Tokenizer
from mindspore.ops import operations as P
from src.GPT2ForSummarization import GPT2SummarizationModel
from src.GPT2ForLanguageModel import GPT2LanguageModel
from src.GPT2_generation import Sample
import mindspore.nn as nn
from mindspore import context,Tensor, Model, Parameter
from mindspore import dtype as mstype
from src.GPT2_model import GPT2Config
import sys

def set_env(mode="GPU", device_id=0, ckpt_path="/datasets/pretrained_weights/ms_model_small.ckpt"):
    context.set_context(mode=context.GRAPH_MODE,
                        device_target=mode, device_id=device_id)
    context.set_auto_parallel_context(parallel_mode="stand_alone")
    print('set context as: {}, using device {}.'.format(mode, device_id))

    config =  GPT2Config(
        batch_size=1,
        seq_length=1024,
        vocab_size=50257,
        d_model=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout=0.1,
        attention_dropout=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
        input_mask_from_dataset=True,
        dtype=mstype.float32,
        compute_type=mstype.float32,
    )

    gpt2_loss = GPT2SummarizationModel(config=config,
                                   is_training=False,
                                   use_one_hot_embeddings=False)
    load_checkpoint_path = ckpt_path
    gpt2_loss.set_train(False)
    param_dict = load_checkpoint(load_checkpoint_path)

    param_dict_ = {}

    print("====process param_dict========")
    for msname in param_dict:
        param_dict_['gpt2.'+msname] = param_dict[msname]
    param_dict_[
        'lm_head.weight'] = param_dict['gpt2_embedding_lookup.embedding_table']
    print("====load params into model====")
    load_param_into_net(gpt2_loss, param_dict_)

    model = Model(gpt2_loss)
    return model, config


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


if __name__ == '__main__':
    print('*'*65)
    print('We are now in testing mode for GPT2 Interactive Generation Demo')
    print('*'*65)
    print('Set Running Env and Load Model')
    gpt2, config = set_env(mode="GPU",device_id=3)
    generate_length = 50

    tokenizer = Tokenizer(vocab_file='./src/utils/pretrain-data/gpt2-vocab.json',
                          merge_file='./src/utils/pretrain-data/gpt2-merges.txt')

    sample = Sample(gpt2, generate_length=generate_length, tokenizer=tokenizer,
                    model_config=config, topk_num=0, topp_prob=0.9, min_tokens_to_keep=1,demo_mode=True)

    official_unicorn_demo = "In a shocking finding, scientist discovered a herd of unicorns living in a remote, previously unexplored valley, in the Andes Mountains. Even more surprising to the researchers was the fact that the unicorns spoke perfect English."

    """
    In the sparse and frozen wilds of Alaska, it is not often that a tasty meal jumps almost straight into your mouth. But that was the case for one bear while hunting salmon in the\xa0Brooks River, which runs through the Katmai National Park, in southern Alaska. However, the dozy creature was unable to take advantage of his good fortune, letting the juicy fish slip away, even after it hit him on the nose. Scroll down for video . Fish supper: A bear hunting salmon in Alaska eyes up his dinner as a fish leaps straight at him while swimming up stream in order to reach its breeding grounds . Staring at defeat: This salmon's number appears to be up as it comes face to face with a hungry bear along the\xa0Brooks River in the\xa0Katmai National Park, Alaska . As close as it gets: As the two creatures come face to face, it looks as if the bear is about to enjoy the most hassle-free meal of its life . Sockeye salmon, which are native to Alaska, migrate up rivers during the spring in order to reach the breeding grounds where they were born in order to spawn. The fish, which spend the rest of the year out in the ocean, will swim against the current in order to reach the spawning grounds, leaping through waterfalls, which is where the bears wait. While the salmon are very fast and difficult to catch underwater, after they leap into the air they have no way of changing course, and so a relatively easy to pick out of the air. Husband and wife photography team Juergen and Christine Sohns captured the moment the bear let his prey get away. The salmon will not eat during their battle upstream, and will undergo a huge transformation, changing from grey to bright red, with their lower lip extending and their head turning green. Swing and a miss: However, nothing is a simple as it seems, and at the very last moment the bear makes a crucial error of judgement, and the Sockeye salmon is allowed to continue its journey . Second time unlucky: Photographer\xa0Juergen Sohns explained that once the fish are in the air they cannot change direction, which should make them easy to catch, but not for this bear, as another fish slips away . Once they reach the breeding grounds, usually a freshwater lake, they will mate, before perishing shortly afterwards. These images were captured by Juergen and Christine Sohns, who travelled to Alaska to photograph the salmon migration. Mr Sohns, 56, took the photo and said: 'The bear was just waiting at the best position in the falls to catch the fish when it was leaping.' He said that, while the bear was unlucky on this occasion, he did have more success after moving further up the river. Mr Sohns and his wife, from Germany, are veteran wildlife photographers, and over the last 20 years have travelled to every continent on Earth photographing wildlife. Fish season: Sockeye salmon spend most of the year out at sea, but during spring they attempt to swim back up rivers to breed, making them easy targets for bears and eagles3020099 . <|endoftext|>
    """

    """
["Sunderland are trailing former Wigan striker Franco di Santo. The Argentine is playing for Werder Bremen and has scored 13 goals in 22 games this season. The Bundesliga side want £8million for the 26-year-old who Sunderland sporting director Lee Congerton knows well from his time at Chelsea. Sunderland are considering a summer move for in-form Werder Bremen forward Franco Di Santo . The Argentine has been in superb form for his club this season, netting 13 goals in 22 games . Di Santo began his senior career with Chilean side Audax Italiano in 2006, before catching the Blues' eye two years later. However, he failed to make an impact at Stamford Bridge and following a similarly ineffectual loan spell at Blackburn Rovers, was sold to Wigan in 2010. He spent three seasons with the Lancashire-based outfit, scoring 13 goals in 97 appearances. Di Santo was an unused substitute during the club's FA Cup final victory over Manchester City in 2013, before being released at the end of that season. He made the move to the Bundesliga in August 2013 and has appeared to fulfil some of his early promise. Di Santo previously played for Chelsea but struggled to make an impact at Stamford Bridge and was sold . Di Santo played for Wigan for three seasons, scoring\xa013 goals in 97 appearances before being released . <|endoftext|>"]
[DEBUG INFO] len_str:[287]
[DEBUG INFO] nextword_distribution shape:(1, 50257)
[DEBUG INFO] distribution shape:(1, 50257)
[DEBUG INFO] word_index:[[25]] shape:(1, 1)
[DEBUG INFO] Sample.generate_for_CNN_DAILYMAIL debugging info:
GENERATED_SUMMARY:
 . Despite Wigan having managed just five players come out of their initial list, their numbers appear to have gone up dramatically this summer with 24 signings already
Sunderland are interested in signing former Chelsea and Wigan forward Franco Di Santo, who has recently hit form for Werder Bremen .Di Santo is currently rated at £8million by the Bundesliga side .Black Cats sporting director Lee Congerton knows Di Santo from their time together at Chelsea .
REF str:
  ['Sunderland are interested in signing former Chelsea and Wigan forward Franco Di Santo, who has recently hit form for Werder Bremen .Di Santo is currently rated at £8million\xa0by the Bundesliga side .Black Cats sporting director Lee Congerton knows Di Santo from their time together at Chelsea .']
HYPO str:
 [' . Despite Wigan having managed just five players come out of their initial list, their numbers appear to have gone up dramatically this summer with 24 signings already']
    """

    while True:
        raw_text = input("Model Prompt >>> ")
        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        if raw_text == "quit()":
            print('\n\nbye~')
            sys.exit(0)
        generate_text, full_text = sample.generate(input_str=str(raw_text))
        print("*"*100)
        print("GPT2 Generation:\n", generate_text)
        print("*"*100)
        print("Full Text Here:\n", full_text)
        # with open("generation.txt","w") as txt:
        #     txt.write('Original:\n'+raw_text+'\n')
        #     txt.write('Generation:\n'+generate_text+'\n')
        #     txt.write('Full Text:\n'+full_text+'\n')
        print("*"*100)
