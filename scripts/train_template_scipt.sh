#activate environment
source /data/tju/env.sh &&

#output log file path
output_log="output.log"

#task info
model="GPT2"
size="medium"
task="Summarization"
dataset="CNN_Dailymail"

#device setting
device="Ascend"
device_id=0

#run setting: training
do_train=True
do_eval=False
epoch_num=5


#dataset path
datafile='/data/tju/mindspore-dataset/cnn_dailymail-test-mindrecord'

#pretrain ckpt file path
pretrain_ckpt_path="/data/tju/pretrained-weight/mindspore_model_medium.ckpt"

#finetune ckpt file path
save_ckpt_path="/data/tju/pretrained-weight/summary/medium"

#create file and head line
echo "TRAIN LOG FILE" > $output_log

#model and task
echo model: $model  model_size: $size >>$output_log
echo task: $task  dataset:$dataset >>$output_log

#record start time
echo "START_TIME" >> $output_log
echo $(date "+%Y-%m-%d %H:%M:%S") >> $output_log

#record info
echo pretrain_ckpt_path: $fintune_ckpt_path>>$output_log


#run python
nohup python run_summarization_model.py  --device_target=$device --device_id=$device_id \
    --do_eval=$do_eval --do_train=$do_train --epoch_num=$epoch_num --train_data_file_path=$datafile\
    --load_pretrain_ckpt_path=$pretrain_ckpt_path --save_finetune_ckpt_path=$save_ckpt_path >> $output_log 2>&1 &


