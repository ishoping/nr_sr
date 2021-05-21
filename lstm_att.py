import torch
import torch.nn as nn
import torch.nn.functional as F

# Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.att_vec = nn.Parameter(torch.randn(hidden_dim))
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
    
    def forward(self, seq, mask):
        '''
        @param
        seq: batch_size, seq_len, dim
        mask: batch_size, seq_len, its' value is -inf
        '''
        att_scores = torch.matmul(self.tanh(seq), self.att_vec)
        att_scores += mask
        att_scores = self.softmax(att_scores)
        batch_size, seq_len = att_scores.shape
        output = torch.matmul(att_scores.view(batch_size, 1, seq_len), seq).squeeze(dim=1)
        output = self.tanh(output)
        return output, att_scores


class lstm_att_classifier(nn.Module):

    #定义所有层
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout):

        super().__init__()          

        #embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        #lstm 层
        self.lstm = nn.LSTM(embedding_dim, 
                           hidden_dim, 
                           num_layers=n_layers, 
                           bidirectional=bidirectional, 
                           batch_first=True)

        self.attention = Attention(hidden_dim * 2)
        
        #全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(p=dropout)
    
    def extract(self, text, text_lengths, att=True):
        
        embedded = self.embedding(text)
        embedded = self.dropout(embedded)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths, batch_first=True, enforce_sorted=False)
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
        seq_unpacked, lens_unpacked = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        seq_unpacked = self.dropout(seq_unpacked)

        if att:
            mask = self.gen_mask(lens_unpacked, seq_unpacked.shape[1], seq_unpacked.device)
            hidden, att_scores  = self.attention(seq_unpacked, mask)
        else:
            hidden = torch.mean(seq_unpacked, dim=1)
            att_scores = F.softmax(torch.ones((seq_unpacked.shape[0], seq_unpacked.shape[1])), dim=1)

        hidden = self.dropout(hidden)

        return hidden, att_scores

    def forward(self, text, text_lengths, att=True):

        hidden, att_scores = self.extract(text, text_lengths, att)
        logits = self.fc(hidden)

        pred = torch.argmax(logits, dim=1)

        return logits, pred
    
    def gen_mask(self, seq_len, max_length, device):
        mask = []
        inf = -10000000
        for length in seq_len:
            length = length.item()
            mask.append([0] * length + [inf] * (max_length - length))
            
        return torch.tensor(mask).to(device)