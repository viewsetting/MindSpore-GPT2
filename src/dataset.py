import os
import numpy as np
import mindspore.common.dtype as mstype
import mindspore.dataset as de
from .finetune_eval_config import gpt2_net_cfg
import mindspore.dataset.transforms.c_transforms as C


def create_language_model_dataset(device_num=1, repeat_count=1, rank_id=0, do_shuffle=True,
                                  dataset_path="/data/tju/src/mindspore-dataset/wikitext2-train-mindrecord"):
    type_cast_op = C.TypeCast(mstype.int32)
    ds = de.MindDataset(dataset_path,
                        columns_list=["input_ids", "input_mask", "label_ids"],
                        shuffle=do_shuffle,
                        num_shards=device_num,
                        shard_id=rank_id)
    print("batch_size: {}".format(gpt2_net_cfg.batch_size))

    ds = ds.map(input_columns="input_ids", operations=type_cast_op)
    ds = ds.map(input_columns="input_mask", operations=type_cast_op)
    ds = ds.map(input_columns="label_ids", operations=type_cast_op)

    # # apply shuffle operation
    # buffer_size = 960
    # ds = ds.shuffle(buffer_size=buffer_size)
    # apply batch operations
    ds = ds.batch(gpt2_net_cfg.batch_size, drop_remainder=True)
    ds = ds.repeat(repeat_count)
    print("dataset size: {}".format(ds.get_dataset_size()))

    print("repeat count: {}".format(ds.get_repeat_count()))
    print("output shape: {}".format(ds.output_shapes()))
    print("output type: {}".format(ds.output_types()))
    print("============== create dataset successful ===============")
    # print(ds)
    return ds

# if __name__ == "__main__":
#     create_language_model_dataset()
