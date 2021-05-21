import re

from nltk import word_tokenize

class Tokenizer:
    def __init__(self, lang='zh_CN', max_num_token=None):
        self.lang = lang
        self.max_num_token = max_num_token
    
    def tokenize(self, text):
        ret = None
        if self.lang == 'zh_CN':
            text = re.sub(r'\s', '', text)
            ret = list(text)
        elif self.lang == 'en':
            words = word_tokenize(text)
            ret = [w.lower() for w in words]
        else:
            raise Exception(f'no support for {self.lang}')

        if self.max_num_token is not None:
            ret = ret[:self.max_num_token]
        return ret