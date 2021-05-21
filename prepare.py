import argparse
import json

import h5py

from vocab import Vocab
from tokenizer import Tokenizer

def pad_sequence(token_seq_list, pad_token):
    token_lenght_list = [len(token_seq) for token_seq in token_seq_list]
    max_length = max(token_lenght_list)

    ret_list = []
    for token_seq in token_seq_list:
        ret_list.append(token_seq + [pad_token] * (max_length - len(token_seq)))
    
    return ret_list, token_lenght_list

def process(vocab_path, keyword_path, label_vocab_path, out_path, max_num_token):
    vocab = Vocab.from_file(vocab_path)
    with open(keyword_path) as f:
        keyword_dict = json.load(f)
    with open(label_vocab_path) as f:
        label_dict = json.load(f)
    id2label = {val: key for key, val in label_dict.items()}

    tokenizer = Tokenizer(lang=LANG, max_num_token=max_num_token)

    token_ids = []
    token_length = []
    label_split = []

    # for label_name, keyword_list in keyword_dict.items():
    for i in range(len(id2label)):
        label_name = id2label[i]
        keyword_list = keyword_dict[label_name]
        print(label_name)
        for keyword in keyword_list:
            tmp_token_ids = [vocab.word2id(word) for word in tokenizer.tokenize(keyword)]
            token_ids.append(tmp_token_ids)
        label_split.append(len(keyword_list))
    
    token_ids, token_length = pad_sequence(token_ids, vocab.word2id('<PAD>'))

    f = h5py.File(out_path, 'w')
    f['token_ids'] = token_ids
    f['token_length'] = token_length
    f['label_split'] = label_split
    
    f.close()

parser = argparse.ArgumentParser()
parser.add_argument('--keywords_path', type=str)
parser.add_argument('--vocab_path', type=str)
parser.add_argument('--label_vocab_path', type=str)
parser.add_argument('--lang', type=str, default='zh_CN')
parser.add_argument('--out_path', type=str)
parser.add_argument('--max_num_token', type=int, default=-1)
args = parser.parse_args()

LANG = args.lang

if __name__ == '__main__':
    # keywords_path = '../resources/keywords_现病史.json'
    # label = '支气管扩张'
    keywords_path = args.keywords_path
    max_num_token = args.max_num_token if args.max_num_token > 0 else None
    process(args.vocab_path, keywords_path, args.label_vocab_path, args.out_path, max_num_token)