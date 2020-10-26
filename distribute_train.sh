#!/bin/bash
# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================






ulimit -u unlimited
export DEVICE_NUM=8
export RANK_SIZE=8
export RANK_TABLE_FILE=/home/cm/hccl_8p_01234567_8.92.9.59.json

export SERVER_ID=0
rank_start=$((DEVICE_NUM * SERVER_ID))
#rank_start=1
for((i=0; i<${DEVICE_NUM}; i++))
do
    export DEVICE_ID=${i}
    export RANK_ID=$((rank_start + i))
    rm -rf ./train_parallel$i
    mkdir ./train_parallel$i
    cp *.py ./train_parallel$i
    cp *.sh ./train_parallel$i
    cp -r ./src ./train_parallel$i
    #cp -r ./utils ./train_parallel$i
    cd ./train_parallel$i || exit
    echo "start training for rank $RANK_ID, device $DEVICE_ID"
    env > env.log
    	    
    python run_translation_model_distributed.py --device_target=Ascend --device_num=$DEVICE_NUM --translate_direction=$1 --do_train=$2 --do_eval=$3
    
    
    

    cd ..
done

# sh distribute_train.sh en-fr true false > trainslation_record_enfr.txt