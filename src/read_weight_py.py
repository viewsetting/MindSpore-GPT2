import torch
import numpy as np
from trans_dict_py import trans_dict

""" to read the parameters of the gpt-2 pretrained model from pytorch into mindspore
    and save them into npy files for mindspore to load
    
    *This script is based on gpt-2 model downloaded from huggingface.*
"""

model_path = "G:/models/117M"
gpt2_checkpoint_path = model_path + "/gpt2-pytorch_model.bin"
#model path and model name
parm_dict = torch.load(gpt2_checkpoint_path)
#load the model parameters

save_param_num=0

for key in parm_dict:
    if key not in trans_dict:
        print(key + " is not in this model")
    else:
        np.save(trans_dict[key] + ".npy", parm_dict[key].numpy())
        save_param_num=save_param_num+1
    # save the parameters by 'npy'

print("finished!")
print("save {num} parameters.".format(num=save_param_num))
