#activate environment
source /data/tju/cm_env.sh &&

#output log file path
output_log="summary_eval_small_finetune.log"

#task info
model="GPT2"
task="Summarization"
dataset="CNN_Dailymail"
status='eval'

#device setting
device="Ascend"
device_id=0

#run setting
do_train=False
do_eval=True
epoch_num=1
eval_load_param_mode='finetune' #[zero-shot,finetune]

#dataset path
datafile="/data/tju/mindspore-dataset/cnn_dailymail-test-mindrecord"

#generate configuration path
generate_conf="scripts/generation_config_TLDR_topk2.json"

#ckpt file path
ckpt_path="/data/tju/pretrained-weight/summary/small/10000.ckpt"

#create file and head line
echo "EVAL LOG FILE" > $output_log

#general info
echo device: $device>>$output_log
echo device_id: $device_id >>$output_log
echo model: $model  >>$output_log
echo task: $task  dataset:$dataset >>$output_log

#record run details
python get_run_settings.py --status $status >>$output_log

#record ckpt info
echo load_param_mode: $eval_load_param_mode >> $output_log
echo fintune_ckpt_path: $ckpt_path>>$output_log

#record generation info
echo generate_config_path: $generate_conf>>$output_log
echo sample params: >> $output_log
echo GENERATION Settings >>$output_log
cat $generate_conf >> $output_log

#record start time
echo "START_TIME" >> $output_log
echo $(date "+%Y-%m-%d %H:%M:%S") >> $output_log

#run python
nohup python run_summarization_model.py  --device_target=$device --device_id=$device_id \
    --do_eval=$do_eval --do_train=$do_train --eval_load_param_mode=$eval_load_param_mode --epoch_num=$epoch_num \
    --generation_config_path=$generate_conf\
    --load_finetune_ckpt_path=$ckpt_path --eval_data_file_path=$datafile >> $output_log 2>&1 &


