from easydict import EasyDict as edict
from src.finetune_eval_config import cfg,gpt2_net_cfg
import argparse
if __name__=='__main__':
    #set argument parser
    parser = argparse.ArgumentParser(description="get settings")
    parser.add_argument('--status',type=str,help="2 status: ['train','eval']",default='eval')
    #get args
    args_opt = parser.parse_args()
    status = args_opt.status
    if status.lower() not in ['train','eval']:
        raise ValueError('train or eval')
    if status.lower() == 'train':
        print("="*35," Start CFG settings ","="*35)
        print("Model Size: {}".format(cfg['gpt2_network']))
        print("Optimizer: {}".format(cfg['optimizer']))
        opt = cfg['optimizer']
        print("learing rate: {}".format(cfg[opt]['learning_rate']))
        print("end_learning_rate': {}".format( cfg[opt]['end_learning_rate'] if 'end_learning_rate' in cfg[opt] else 'None'))
        print('power: {}'.format(cfg[opt]['power'] if 'power' in cfg[opt] else 'None' ))
        print("="*35," End of CFG settings ","="*35)
        print("\n")
    print("GPT2 settings:")
    print("Batch Size: {}".format(gpt2_net_cfg.batch_size))
    print("Seq Length: {}".format(gpt2_net_cfg.seq_length))
    print("Vocab Size: {}".format(gpt2_net_cfg.vocab_size))
    print("\n")
    
    
