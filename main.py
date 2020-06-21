import os
import re
import json
import string
import argparse
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer, TFBertModel, BertConfig
from models import CABert
import pandas as pd

max_len = 512
configuration = BertConfig()  # default paramters and configuration for BERT

# Save the slow pretrained tokenizer
slow_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
save_path = "bert_base_uncased/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
slow_tokenizer.save_pretrained(save_path)

# Load the fast tokenizer from saved file
tokenizer = BertWordPieceTokenizer("bert_base_uncased/vocab.txt", lowercase=True)

if __name__ == '__main__':
    my_parser = argparse.ArgumentParser(description='List the content of a folder')

    # Add the arguments
    my_parser.add_argument('train',
                           metavar='train',
                           type=str,
                           help='the path to train csv',
                           default='/build_dataset/src/train.csv')
    my_parser.add_argument('test',
                           metavar='test',
                           type=str,
                           help='the path to test csv',
                           default='/build_dataset/src/test.csv')
    my_parser.add_argument('model_name',
                           metavar='model_name',
                           type=str,
                           help='bert model name',
                           default='dnn')
    my_parser.add_argument('use_tpu',
                           metavar='use_tpu',
                           type=str,
                           help='use tpu or gpu',
                           default=True)
    my_parser.add_argument('epoch',
                           metavar='epoch',
                           type=str,
                           help='epoch',
                           default=3)
    my_parser.add_argument('steps',
                           metavar='steps',
                           type=str,
                           help='steps',
                           default=20)
    my_parser.add_argument('max_len',
                           metavar='max_len',
                           type=str,
                           help='max_len',
                           default=512)
    args = my_parser.parse_args()
    df=pd.read_csv(args.train)
    CaModel=CABert(args.model_name)
    print('run')
    CaModel.LoadModel(eval(args.use_tpu))
    print('run')
    CaModel.trainModel(df,eval(args.epoch),eval(args.steps))
    print('run')