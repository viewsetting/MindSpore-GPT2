#en-fr train
python src/create_data.py --input_file /home/tju/gpt2/1M/baseline-1M_train.en-fr --output_file /home/tju/gpt2/en-fr-train-mindrecord --num_splits 1 --max_seq_length 1024  --vocab_file src/utils/pretrain-data/gpt2-vocab.json --merge_file src/utils/pretrain-data/gpt2-merges.txt &&

#en-fr test
python src/create_data.py --input_file /home/tju/gpt2/1M/newstest2014.en-fr --output_file /home/tju/gpt2/en-fr-train-mindrecord --num_splits 1 --max_seq_length 1024  --vocab_file src/utils/pretrain-data/gpt2-vocab.json --merge_file src/utils/pretrain-data/gpt2-merges.txt &&

#fr-en train
python src/create_data.py --input_file /home/tju/gpt2/1M/baseline-1M_train.en-fr --output_file /home/tju/gpt2/fr-en-test-mindrecord --num_splits 1 --max_seq_length 1024  --vocab_file src/utils/pretrain-data/gpt2-vocab.json --merge_file src/utils/pretrain-data/gpt2-merges.txt &&


#fr-en test
python src/create_data.py --input_file /home/tju/gpt2/1M/newstest2014.fr-en --output_file /home/tju/gpt2/fr-en-test-mindrecord --num_splits 1 --max_seq_length 1024  --vocab_file src/utils/pretrain-data/gpt2-vocab.json --merge_file src/utils/pretrain-data/gpt2-merges.txt &&
