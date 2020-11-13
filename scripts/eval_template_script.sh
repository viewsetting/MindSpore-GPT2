#activate environment
source /data/tju/env.sh &&

#output log file path
output_log="noBOS_topk2_summary.log"

#task info
model="GPT2"
size="small"
task="Summarization"
dataset="CNN_Dailymail"

#device setting
device="Ascend"
device_id=7

#run setting
do_train=False
do_eval=True
epoch_num=1
eval_load_param_mode='finetune'

#dataset path
datafile='/data/tju/mindspore-dataset/cnn_dailymail-test-mindrecord'

#sample setting
topk=2
topp=1.0
temp=1

#for CNN_DailyMail dataset modified parameter
append_eos=True

#fintune ckpt file path
fintune_ckpt_path="/data/tju/pretrained-weight/summary/gpt2_summarization_noBOS_-2_71778.ckpt"

#create file and head line
echo "EVAL LOG FILE" > $output_log

#model and task
echo model: $model  model_size: $size >>$output_log
echo task: $task  dataset:$dataset >>$output_log

#record start time
echo "START_TIME" >> $output_log
echo $(date "+%Y-%m-%d %H:%M:%S") >> $output_log

#record info
echo sample params: >> $output_log
echo top_k $topk top_p $topp temperature $temp>>$output_log
echo mode: $eval_load_param_mode >> $output_log
echo fintune_ckpt_path: $fintune_ckpt_path>>$output_log


#run python
nohup python run_summarization_model.py  --device_target=$device --device_id=$device_id \
    --do_eval=$do_eval --do_train=$do_train --eval_load_param_mode=$eval_load_param_mode --epoch_num=$epoch_num \
    --top_p=$topp --top_k=$topk --temp=$temp   --append_eos=$append_eos \
    --load_finetune_ckpt_path=$fintune_ckpt_path --eval_data_file_path=$datafile >> $output_log 2>&1 &

#record end time
echo "END_TIME" >> $output_log
echo $(date "+%Y-%m-%d %H:%M:%S") >> $output_log
