import argparse
import json

import pandas as pd

from vocab import Vocab, LabelVocab
from tokenizer import Tokenizer

def gen_label_dict(data_path, out_label_vocab_path):
    data = pd.read_csv(data_path)
    label_list = data['label'].unique()
    label_dict = {}
    for key, val in zip(label_list, range(len(label_list))):
        key = str(key)
        label_dict[key] = val

    with open(out_label_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(label_dict, f, ensure_ascii=False)

def gen_vocab(data_path, out_vocab_path):
    '''
    @param
    data_path: csv文件路径，格式为id, text, text_label
    out_vocab_path: 字典路径，每一行表示一个单词，包括PAD和UNK
    '''
    data = pd.read_csv(data_path)
    token_list = []
    tokenizer = Tokenizer(lang=LANG)
    for _, row in data.iterrows():
        token_list.extend(tokenizer.tokenize(row['text']))
    vocab = Vocab.from_token_list(token_list)
    vocab.save(out_vocab_path)

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--lang', type=str, default='zh_CN')
args = parser.parse_args()

LANG = args.lang

if __name__ == "__main__":
    
    data_path = args.in_path
    out_label_vocab_path = f'{args.out_dir}/label_vocab.json'
    out_vocab_path = f'{args.out_dir}/vocab.txt'

    gen_label_dict(data_path, out_label_vocab_path)
    gen_vocab(data_path, out_vocab_path)