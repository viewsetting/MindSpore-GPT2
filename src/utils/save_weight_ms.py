from mindspore import Tensor
import numpy as np
from mindspore.train.serialization import save_checkpoint
from trans_dict import trans_dict_tf
import os
#In this script, dict_tf and dict_py have the same effect. So we can choose one of them.

"""to load the parameters of gpt-2 model from '.npy' file
   npy files should be in the same path with this script. Otherwise you should change the path name of the script.
"""



def trans_model_para():
    file_names = [name for name in os.listdir() if name.endswith(".npy")]
    #to find all file names with suffix '.npy' in the current path.
    new_params_list = []
    for file_name in file_names:
        var_name=file_name[:-4]
        param_dict = {"name": var_name, "data": Tensor(np.load(file_name))}
        if var_name in trans_dict.values():
            new_params_list.append(param_dict)
            print(var_name+" has been saved")

    save_checkpoint(new_params_list, "ms_model_medium.ckpt")
    #to load the parameters from npy files and save them as mindspore checkpoint

    print("Finished:the parameters have been saved into mindspore checkpoint.")

if __name__ == "__main__":
    trans_model_para()

