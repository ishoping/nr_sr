import h5py
import torch
from torch.utils.data import Dataset

class PtDataset(Dataset):
    def __init__(self, data_path):
        self.lamb = 0.
        
        dataset = h5py.File(data_path, 'r')
        self.text = torch.LongTensor(dataset['token_ids'][:])
        self.text_length = torch.LongTensor(dataset['token_length'][:])
        self.label = torch.LongTensor(dataset['label'][:])
        self.pt_id = dataset['id'][:]
        self.label_split_point = tuple(dataset['label_split'][:])
        self.clean = dataset['clean'][:]

        self.weight = torch.ones(self.text_length.shape)
        # if 'weight' in dataset.keys():
        #     self.weight = torch.tensor(dataset['weight'][:])
        # else:
        #     self.weight = torch.ones(self.text_length.shape)

    def __getitem__(self, index):
        return self.text[index], self.text_length[index], self.label[index], self.weight[index], self.clean[index], index

    def __len__(self):
        return len(self.text)
    
    def update_weight(self, weight):
        self.weight = self.lamb * self.weight + (1 - self.lamb) * weight
    
    def load_weight(self, weight):
        self.weight = weight


class KwDataset(Dataset):
    def __init__(self, data_path):
        dataset = h5py.File(data_path, 'r')
        self.text = torch.LongTensor(dataset['token_ids'][:])
        self.text_length = torch.LongTensor(dataset['token_length'][:])
        self.label_split_point = tuple(dataset['label_split'][:])
    
    def __getitem__(self, index):
        return self.text[index], self.text_length[index]

    def __len__(self):
        return len(self.text)
  