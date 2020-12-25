#activate environment
#source /data/tju/cm_env.sh &&
source /data/env.sh &&

#output log file path
output_log="log/summary_small_train.log"

#task info
model="GPT2"
task="Summarization"
dataset="CNN_Dailymail"
status='train' #[train or eval]

#device setting
device="Ascend"
device_id=4

#run setting
do_train=True
do_eval=False
epoch_num=1
eval_load_param_mode='zero-shot' #[zero-shot,finetune]

#dataset path
datafile="/data2/tju/mindspore-dataset/cnn/cnn_dailymail_train_10000-mindrecord"

#ckpt file path
ckpt_path="/data2/tju/gpt2_weights/ms_model_small.ckpt"

#create file and head line
echo "EVAL LOG FILE" > $output_log

#general info
echo device: $device>>$output_log
echo device_id: $device_id >>$output_log
echo model: $model  >>$output_log
echo task: $task  dataset:$dataset >>$output_log

#record run details
python ../get_run_settings.py --status $status >>$output_log

#record ckpt info
echo load_param_mode: $eval_load_param_mode >> $output_log
echo fintune_ckpt_path: $ckpt_path>>$output_log

#record start time
echo "START_TIME" >> $output_log
echo $(date "+%Y-%m-%d %H:%M:%S") >> $output_log

#run python
#nohup python ../run_summarization_model.py  --device_target=$device --device_id=$device_id \
#    --do_eval=$do_eval --do_train=$do_train --eval_load_param_mode=$eval_load_param_mode --epoch_num=$epoch_num \
#    --load_pretrain_ckpt_path=$ckpt_path --train_data_file_path=$datafile >> $output_log 2>&1 &

python ../run_summarization_model.py  --device_target=$device --device_id=$device_id \
    --do_eval=$do_eval --do_train=$do_train --eval_load_param_mode=$eval_load_param_mode --epoch_num=$epoch_num \
    --load_pretrain_ckpt_path=$ckpt_path --train_data_file_path=$datafile 

