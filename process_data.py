import argparse
import json

import h5py
import numpy as np
import pandas as pd

from vocab import Vocab, LabelVocab
from tokenizer import Tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--in_path', type=str)
parser.add_argument('--out_dir', type=str)
parser.add_argument('--lang', type=str, default='zh_CN')
parser.add_argument('--gen_vocab', type=int, default=1)
parser.add_argument('--max_num_token', type=int, default=-1)
args = parser.parse_args()

LANG = args.lang


random_state = 2020

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


def pad_sequence(token_seq_list, pad_token):
    token_lenght_list = [len(token_seq) for token_seq in token_seq_list]
    max_length = max(token_lenght_list)

    ret_list = []
    for token_seq in token_seq_list:
        ret_list.append(token_seq + [pad_token] * (max_length - len(token_seq)))
    
    return ret_list, token_lenght_list


def process_data(data_df, vocab_path, label_vocab_path, out_path, max_num_token):
    '''
    @param
    data_df: 格式为id,text,label,clean
    vocab_path: 字典路径，每一行表示一个单词，包括PAD和UNK
    out_path: 数据集路径
    '''
    vocab = Vocab.from_file(vocab_path)
    with open(label_vocab_path) as f:
        label_vocab = json.load(f)
    tokenizer = Tokenizer(lang=LANG, max_num_token=max_num_token)

    id_list = []
    token_ids_list = []
    label_ids_list = []
    clean_list = []
    label_split = []
    cur_label = data_df.iloc[0]['label']
    pre_idx = 0
    for idx, (_, row) in enumerate(data_df.iterrows()):
        # print(idx)
        tmp_token_ids = [vocab.word2id(word) for word in tokenizer.tokenize(row['text'])]
        token_ids_list.append(tmp_token_ids)
        
        label_ids_list.append(label_vocab[str(row['label'])])
        id_list.append(row['id'])
        clean_list.append(row['clean'])
        if row['label'] != cur_label:
            label_split.append(idx - pre_idx)
            pre_idx = idx
            cur_label = row['label']
            # print(row['label'], pre_idx)
    label_split.append(len(data_df) - pre_idx)
    # print(label_split)
    # print(data_df['label'].value_counts(sort=False))
    
    token_ids_list, token_lenght_list = pad_sequence(token_ids_list, vocab.word2id('<PAD>'))

    token_ids_list = np.array(token_ids_list)
    token_lenght_list = np.array(token_lenght_list)
    label_ids_list = np.array(label_ids_list)
    id_list = np.array(id_list)
    clean_list = np.array(clean_list)
    label_split = np.array(label_split)

    f = h5py.File(out_path, 'w')
    f['token_ids'] = token_ids_list
    f['token_length'] = token_lenght_list
    f['label'] = label_ids_list
    f['id'] = id_list
    f['clean'] = clean_list
    f['label_split'] = label_split
    
    f.close()

if __name__ == "__main__":
    
    data_path = args.in_path
    out_vocab_path = f'{args.out_dir}/vocab.txt'
    out_label_vocab_path = f'{args.out_dir}/label_vocab.json'

    max_num_token = args.max_num_token if args.max_num_token > 0 else None

    if args.gen_vocab:
        gen_vocab(data_path, out_vocab_path)
    # gen_label_dict(data_path, out_label_vocab_path)

    zh2en = {
        '训练': 'train',
        '验证': 'valid',
        '测试': 'test'
    }

    data = pd.read_csv(data_path)
    for lb, df in data.groupby('数据集'):
        process_data(df,
                    out_vocab_path,
                    out_label_vocab_path,
                    f'{args.out_dir}/{zh2en[lb]}.h5',
                    max_num_token
                    )