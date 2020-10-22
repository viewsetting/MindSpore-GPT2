from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import logging
import numpy as np
from utils import tokenization
from mindspore.mindrecord import FileWriter

"""
WikiText:
python create_data.py --input_file /data/tju/new_wiki.test.txt --output_file /data/tju/src/mindspore-dataset/wikitext2-test-mindrecord --num_splits 1 --max_seq_length 1024

CNN_DailyMail 3.0.0:

import test set:
python create_data.py --input_file /data/tju/cnn_dailymail_test.txt --output_file /data/tju/src/mindspore-dataset/cnn_dailymail-test-mindrecord --num_splits 1 --max_seq_length 1024

import training set:
python create_data.py --input_file /data/tju/cnn_dailymail_train.txt --output_file /data/tju/src/mindspore-dataset/cnn_dailymail-train-mindrecord --num_splits 1 --max_seq_length 1024

"""

def create_instance(tokenizer, text, max_length=None):
    sentence = text.strip().split("\t")
    #print("[debug info]: ",sentence[0],"\n",sentence[1])
    # len(sentence) == 1:
    ids = tokenizer.encode(sentence[0])
    pair_ids = None
    if len(sentence) == 2:
        pair_ids = tokenizer.encode(sentence[1])
    if len(sentence)>=3:
        article = sentence[0]
        for i in range(1,len(sentence)-1):
            article+=sentence[i]
        ids = tokenizer.encode(article)
        pair_ids = tokenizer.encode(sentence[-1])

    output = tokenizer.prepare_for_model(ids=ids,
                                         pair_ids=pair_ids,
                                         add_special_tokens=True,
                                         max_length=max_length,
                                         padding=True,
                                         return_overflowing_tokens=False,
                                         return_attention_mask=True)
    return output


def write_instance_to_file(writer, instance):
    # input_ids = instance["input_ids"][:-1]  # bos text
    # input_mask = instance["attention_mask"][:-1]
    # label_ids = instance["input_ids"][1:]  # text eos
    input_ids = instance["input_ids"]
    input_mask = instance["attention_mask"]
    label_ids = instance["input_ids"]
    assert len(input_ids) == len(label_ids)

    features = collections.OrderedDict()
    features["input_ids"] = np.asarray(input_ids)
    features["input_mask"] = np.asarray(input_mask)
    features["label_ids"] = np.asarray(label_ids)

    writer.write_raw_data([features])
    return features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help='Input raw text file. ')
    parser.add_argument("--output_file", type=str, required=True, help='Output MindRecord file. ')
    parser.add_argument("--num_splits", type=int, default=1,
                        help='The MindRecord file will be split into the number of partition. ')
    parser.add_argument("--max_seq_length", type=int, required=True, help='Maximum sequence length. ')
    parser.add_argument("--vocab_file", type=str, required=False, help='url of gpt2-vocab.json ',default='./utils/pretrain-data/gpt2-vocab.json')
    parser.add_argument("--merge_file", type=str, required=False, help='url of gpt2-merges.txt ',default='./utils/pretrain-data/gpt2-merges.txt')
    parser.add_argument("--mode",type=str,required=False,default='normal',help='mode of dataset creation')
    args = parser.parse_args()

    tokenizer = tokenization.Tokenizer(vocab_file=args.vocab_file,merge_file=args.merge_file,mode=args.mode)
    input_file = args.input_file
    logging.info("***** Reading from input files *****")
    logging.info("Input File: %s", input_file)

    output_file = args.output_file
    logging.info("***** Writing to output files *****")
    logging.info("Output File: %s", output_file)

    writer = FileWriter(output_file, args.num_splits)
    data_schema = {"input_ids": {"type": "int64", "shape": [-1]},
                   "input_mask": {"type": "int64", "shape": [-1]},
                   "label_ids": {"type": "int64", "shape": [-1]}
                   }
    writer.add_schema(data_schema, "wikitext2-schema")

    total_written = 0
    total_read = 0

    logging.info("***** Reading from  %s *****", input_file)
    with open(input_file, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            total_read += 1
            if total_read % 500 == 0:
                logging.info("%d ...", total_read)

            output = create_instance(tokenizer, line, args.max_seq_length)
            features = write_instance_to_file(writer, instance=output)
            total_written += 1

            if total_written <= 20:
                logging.info("***** Example *****")
                logging.info("input tokens: %s", tokenizer.decode(output["input_ids"][:-1]))
                logging.info("label tokens: %s", tokenizer.decode(output["input_ids"][1:]))

                for feature_name in features.keys():
                    feature = features[feature_name]
                    logging.info("%s: %s", feature_name, feature)

    writer.commit()
    logging.info("Wrote %d total instances", total_written)


if __name__ == "__main__":
    main()
