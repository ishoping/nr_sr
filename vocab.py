from collections import Counter
import json

class LabelVocab:
    def __init__(self):
        self.__word2id = {}
        self.__id2word = []
    
    def word2id(self, word):
        return self.__word2id[word]

    def id2word(self, idx):
        return self.__id2word[idx]
    
    def add_word(self, word):
        if word not in self.__word2id:
            self.__word2id[word] = len(self.__word2id)
            self.__id2word.append(word)
    
    def save(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(self.__word2id, f, ensure_ascii=False, indent=4)

class Vocab:
    def __init__(self):
        self.__word2id = {
            '<PAD>': 0,
            '<UNK>': 1
        }
        self.__id2word = ['<PAD>', '<UNK>']
    
    def __len__(self):
        return len(self.__id2word)
    
    def word2id(self, word):
        if word not in self.__word2id:
            return self.__word2id['<UNK>']
        return self.__word2id[word]

    def id2word(self, idx):
        if idx < 0 or idx >= len(self.__id2word):
            raise IndexError()
        return self.__id2word[idx]
    
    def add_word(self, word):
        if word not in self.__word2id:
            self.__word2id[word] = len(self.__word2id)
            self.__id2word.append(word)
    
    def save(self, out_path):
        with open(out_path, 'w', encoding='utf-8') as f:
            for word in self.__id2word:
                f.write(word + '\n')
    
    @classmethod
    def from_file(cls, file_path):
        vocab = cls()
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                word = line.split()[0]
                vocab.add_word(word)
        return vocab
    
    @classmethod
    def from_token_list(cls, token_list, min_freq=5):
        vocab = cls()
        counter = Counter(token_list)
        for word, freq in counter.most_common():
            if freq >= min_freq:
                vocab.add_word(word)
        return vocab

