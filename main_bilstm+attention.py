# 参考代码来源 http://pytorchchina.com/2020/02/15/使用pytorch建立你的第一个文本分类模型
import json
import math
import random
import os

import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import h5py
import pandas as pd

from dataset import PtDataset, KwDataset
from lstm_att import lstm_att_classifier
from vocab import Vocab

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_data_path', type=str)
parser.add_argument('--valid_data_path', type=str)
parser.add_argument('--test_data_path', type=str)
parser.add_argument('--out_model_path', type=str)
parser.add_argument('--out_weight_path', type=str)
parser.add_argument('--log_path', type=str)
parser.add_argument('--vocab_path', type=str)
parser.add_argument('--n_epochs', type=int)
parser.add_argument('--epoch_weight', type=int)
parser.add_argument('--noise_rate', type=float)
parser.add_argument('--soft', type=int)
parser.add_argument('--soft_factor', type=int)
parser.add_argument('--dim_hidden', type=int)
parser.add_argument('--num_classes', type=int)
parser.add_argument('--dropout', type=float)
parser.add_argument('--fr_type', type = str, help='forget rate type')
parser.add_argument('--construct_method', type = str)
parser.add_argument('--seed', type = int)
parser.add_argument('--batch_size', type = int)
parser.add_argument('--num_workers', type = int)
parser.add_argument('--output_model_epoch', type = int, default=0)
parser.add_argument('--device', type=str, help='cuda:0', default='cuda:0')
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
noise_rate = args.noise_rate
SOFT = args.soft
SOFT_FACTOR = args.soft_factor
OUTPUT_DIR, _ = os.path.split(args.log_path)
OUTPUT_SPECIFIED_EPOCH_MODEL = True if args.output_model_epoch else False

# define drop rate schedule
def gen_forget_rate(fr_type='type_1'):
    if fr_type=='type_1':
        rate_schedule = np.ones(1+args.n_epochs) * noise_rate
        rate_schedule[:args.epoch_weight] = np.linspace(0, noise_rate, args.epoch_weight)
    elif fr_type=='type_2':
        rate_schedule = np.ones(1+args.n_epochs) * noise_rate
        rate_schedule[:args.epoch_weight] = 0

    return rate_schedule

if args.epoch_weight < args.n_epochs:
    rate_schedule = gen_forget_rate(args.fr_type)

def cal_noise_rate(dataset):
    data = pd.DataFrame({
        'weight': dataset.weight.numpy(),
        'clean': dataset.clean,
        'label': dataset.label.numpy()
    })

    mask = data['weight'] == 1
    data = data[mask]

    ret = {}

    for lb, df in data.groupby(by=['label']):
        ret[f'n_r_{lb}'] = 1 - df['clean'].sum() / len(df)

    ret['n_r'] = 1 - data['clean'].sum() / len(data)
    return ret

def train_epoch(model, iterator, optimizer, criterion):
    #初始化
    epoch_loss = 0
    epoch_acc = 0

    eps = 1e-12

    #设置为训练模式
    model.train()  
    for text, text_lengths, label, weight, *_ in tqdm(iterator):
        
        optimizer.zero_grad()

        text = text.to(device)
        text_lengths = text_lengths.to(device)
        label = label.to(device)
        weight = weight.to(device)
        
        logits, pred = model(text, text_lengths)
        loss = criterion(logits, label)
        loss = torch.dot(weight, loss) / label.shape[0]
        acc = metrics.accuracy_score(label.detach().cpu().numpy(),
                                    pred.detach().cpu().numpy()
                                    )

        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()

        #损失和精度
        epoch_loss += loss.item()
        epoch_acc += acc

    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):

    #初始化
    epoch_loss = 0
    loss_list = []
    
    y_pred_list = []
    y_true_list = []

    clean_list = []

    model.eval()

    #取消autograd
    with torch.no_grad():
        for text, text_lengths, label, _, clean, *_ in tqdm(iterator):

            text = text.to(device)
            text_lengths = text_lengths.to(device)
            label = label.to(device)

            logits, pred = model(text, text_lengths)

            #计算损失和准确性
            loss = criterion(logits, label)
            loss_list.extend(loss.cpu().numpy())
            loss = loss.mean()
            
            pred = pred.cpu().numpy()
            label = label.cpu().numpy()

            y_pred_list.extend(pred)
            y_true_list.extend(label)

            clean_list.extend(clean.numpy())
            
            
            #跟踪损失和准确性
            epoch_loss += loss.item()
            
    y_pred = np.array(y_pred_list)
    y_true = np.array(y_true_list)
    epoch_acc = metrics.accuracy_score(y_true, y_pred)
    report = metrics.classification_report(y_true, y_pred)

    ret_df = pd.DataFrame({
        'y_true': y_true_list,
        'y_pred': y_pred_list,
        'loss': loss_list,
        'clean': clean_list
    })

    ret_dict = {
        'epoch_loss': epoch_loss / len(iterator),
        'epoch_acc': epoch_acc,
        'report': report
    }

    for grp_lb, df in ret_df.groupby(by=['y_true', 'clean']):
        suffix = '-'.join([str(i) for i in grp_lb])
        acc = metrics.accuracy_score(df['y_true'], df['y_pred'])
        
        ret_dict[f'acc-{suffix}'] = acc
        ret_dict[f'loss-{suffix}'] = df['loss'].mean()

    return ret_dict

def train(model, train_iterator, valid_iterator, test_iterator, optimizer, criterion, scheduler, n_epochs, out_model_path, log_path):
    best_valid_acc = float('-inf')
    epoch_weight = args.epoch_weight
    save_method = 'best'
    
    weight_dict = {
        'id': train_iterator.dataset.pt_id,
        'weight': train_iterator.dataset.weight.numpy()
    }

    is_first = True
    output_columns = None

    for epoch in range(1+n_epochs):

        #训练模型
        if epoch > 0:
            train_epoch(model, train_iterator, optimizer, criterion)
            # train_loss, train_acc = train_epoch(model, train_iterator, optimizer, criterion)

        #评估模型
        ret_dict = evaluate(model, valid_iterator, criterion)
        train_ret_dict = evaluate(model, train_iterator, criterion)
        train_ret_dict = {f'train_{key}': val for key, val in train_ret_dict.items()}
        test_ret_dict = evaluate(model, test_iterator, criterion)
        test_ret_dict = {f'test_{key}': val for key, val in test_ret_dict.items()}
        if not SOFT:
            train_ret_dict.update(cal_noise_rate(train_iterator.dataset))
        ret_dict.update(train_ret_dict)
        ret_dict.update(test_ret_dict)
        train_loss, train_acc = ret_dict['train_epoch_loss'], ret_dict['train_epoch_acc']
        test_loss, test_acc = ret_dict['test_epoch_loss'], ret_dict['test_epoch_acc']
        valid_loss, valid_acc, report = ret_dict['epoch_loss'], ret_dict['epoch_acc'], ret_dict['report']
        del ret_dict['report']
        del ret_dict['train_report']
        del ret_dict['test_report']
        ret_dict['epoch'] = epoch
        if is_first:
            is_first = False
            output_columns = list(ret_dict.keys())
            output_columns.remove('epoch')
            output_columns = ['epoch'] + output_columns
            pd.DataFrame({key: [val] for key, val in ret_dict.items()})[output_columns].to_csv(log_path, index=False)
        else:
            pd.DataFrame({key: [val] for key, val in ret_dict.items()})[output_columns].to_csv(log_path, mode='a', header=False, index=False)
        #保存最佳模型
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            if save_method == 'best':
                torch.save(model.state_dict(), out_model_path)
        
        scheduler.step()

        print(f'\t Epoch: {epoch} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
        print(f'\t Epoch: {epoch} | Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
        print(f'\t Epoch: {epoch} | Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')
        print(report)
        
        if epoch_weight < n_epochs:
            if rate_schedule[epoch] > 0:
                dataset, _ = update_sample_weight(model, train_iterator.dataset, epoch)
                train_iterator = DataLoader(dataset,
                        batch_size=train_iterator.batch_size,
                        shuffle=True,
                        drop_last=True,
                        num_workers=args.num_workers
                        )
                weight_dict[f'weight_{epoch}'] = train_iterator.dataset.weight.numpy()
    
        if OUTPUT_SPECIFIED_EPOCH_MODEL:
            if epoch > 0 and epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_{epoch}.pkl'))
    # if save_method == 'last':
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f'model_last.pkl'))
        # torch.save(model.state_dict(), out_model_path)

    pd.DataFrame(weight_dict).to_csv(args.out_weight_path, index=False)

def construct_prototype(train_vector_list, weight_list, clean_list, construct_method):
    
    def construct_prototype_one_cls(train_vector, weight, clean, construct_method):
        if construct_method == 'mean_over_all':
            return torch.mean(train_vector, dim=0, keepdim=True)
        elif construct_method == 'mean_over_clean':
            mask = clean == 1
            train_vector_clean = train_vector[mask]
            return torch.mean(train_vector_clean, dim=0, keepdim=True)
        elif construct_method == 'weight_mean_over_all':
            return torch.mm(weight.view(1, -1), train_vector)
        else:
            raise Exception(f'construct_method: No support for {construct_method}')

    proto = []

    for i in range(len(train_vector_list)):
        proto_one_cls = construct_prototype_one_cls(train_vector_list[i], weight_list[i], clean_list[i], construct_method)
        proto.append(proto_one_cls)
    # proto = torch.cat(proto)
    return proto

def get_weight(mat, mat_refer, epoch):
    batch_size = 512
    method = 'mean'
    k = 5
    n = math.ceil(len(mat) / batch_size)
    weight = []
    weight_per_keywords = []

    mat = F.normalize(mat, dim=1)
    mat_refer = F.normalize(mat_refer, dim=1)

    mat_refer = mat_refer.to(device)
    
    for i in range(n):
        start = i * batch_size
        mat_batch = mat[start:start+batch_size].to(device)
        similarity = torch.matmul(mat_batch, mat_refer.T)
        # print(similarity.shape)
        # weight_per_keywords.append((1 + similarity) / 2)
        if method == 'mean':
            similarity = torch.mean(similarity, dim=1)
        elif method == 'mean_topk':
            similarity = torch.mean(torch.topk(similarity, k=k, dim=1, sorted=False)[0], dim=1)
        else:
            raise Exception(f'no support method in similarity calculation: {method}')
        similarity = (1 + similarity) / 2
        weight.append(similarity)
    
    weight = torch.cat(weight)
    idx_first_good = int(len(weight) * rate_schedule[epoch]) + 1
    # idx_first_good = int(len(weight) * noise_rate) + 1
    threshold_good, _ = torch.kthvalue(weight, idx_first_good)
    
    # hard
    if SOFT:
        weight = F.sigmoid(SOFT_FACTOR * (weight - threshold_good))
    else:
        weight = torch.where(weight >= threshold_good, torch.tensor([1]).to(device), torch.tensor([0]).to(device))
    # weight_per_keywords = torch.cat(weight_per_keywords)
    
    # return weight.cpu(), weight_per_keywords.cpu()
    return weight.cpu(), None


def extract_feature(model, iterator, att=True):
    model.eval()
    all_feature_vector = []
    with torch.no_grad():
        for text, text_lengths, *_ in iterator:

            text = text.to(device)
            text_lengths = text_lengths.to(device)

            text_vector, _ = model.extract(text, text_lengths, att=att)
            all_feature_vector.append(text_vector.cpu())
    return torch.cat(all_feature_vector)

def update_sample_weight(model, dataset, epoch):
    data_iterator = DataLoader(dataset=dataset, batch_size=128 if args.batch_size < 128 else args.batch_size, num_workers=args.num_workers)

    train_vector = extract_feature(model, data_iterator)
    train_vector_list = train_vector.split(data_iterator.dataset.label_split_point)
    old_weight_list = data_iterator.dataset.weight.split(data_iterator.dataset.label_split_point)
    clean_list = torch.tensor(data_iterator.dataset.clean).split(data_iterator.dataset.label_split_point)

    proto = construct_prototype(train_vector_list, old_weight_list, clean_list, args.construct_method)
    print(len(proto), proto[0].shape, proto[1].shape)

    weight = []
    # weight_per_keywords = []

    for i in range(len(train_vector_list)):
        weight_class, weight_per_keywords_class = get_weight(train_vector_list[i], proto[i], epoch)
        weight.append(weight_class)
        # weight_per_keywords.append(weight_per_keywords_class)
    
    weight = torch.cat(weight)
    # weight_per_keywords = torch.cat(weight_per_keywords).numpy()

    dataset.update_weight(weight)

    return dataset, None


def load_data(train_data_path, train_data_weight_path, valid_data_path, test_data_path, batch_size):
    train_dataset = PtDataset(train_data_path)
    if train_data_weight_path:
        weight = pd.read_csv(train_data_weight_path)
        train_dataset.load_weight(torch.tensor(weight['weight'].values))
    valid_dataset = PtDataset(valid_data_path)
    test_dataset = PtDataset(test_data_path)

    train_iterator = DataLoader(train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    num_workers=args.num_workers
                    )
    valid_iterator = DataLoader(valid_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=args.num_workers
                    )
    test_iterator = DataLoader(test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                drop_last=False,
                                num_workers=args.num_workers
                    )
    
    return train_iterator, valid_iterator, test_iterator


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    out_model_path = args.out_model_path
    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path
    test_data_path = args.test_data_path
    vocab_path = args.vocab_path
    log_path = args.log_path
    train_data_weight_path = None

    n_epochs = args.n_epochs
    batch_size = args.batch_size
    vocab = Vocab.from_file(vocab_path)
    size_of_vocab = len(vocab)
    
    embedding_dim = args.dim_hidden
    num_hidden_nodes = args.dim_hidden
    num_output_nodes = args.num_classes
    num_layers = 1
    bidirection = True
    dropout = args.dropout

    #产生同样的结果
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    
    train_iterator, valid_iterator, test_iterator = load_data(train_data_path, train_data_weight_path, valid_data_path, test_data_path, batch_size)
    model = lstm_att_classifier(size_of_vocab,
                    embedding_dim,
                    num_hidden_nodes,
                    num_output_nodes,
                    num_layers, 
                    bidirectional=True,
                    dropout = dropout
                    )
    #模型体系
    print(SEED)
    print(model)
    print(f'The model has {count_parameters(model):,} trainable parameters')
    #初始化预训练embedding
    # pretrained_embeddings = vocab.vectors
    # model.embedding.weight.data.copy_(pretrained_embeddings)
    # print(pretrained_embeddings.shape)

    #定义优化器和损失
    optimizer = optim.Adam(model.parameters(), lr=0.001) # weight_decay=0.0003,
    criterion = nn.CrossEntropyLoss(reduction='none')
    scheduler = MultiStepLR(optimizer, milestones=[], gamma=0.5)

    #如果cuda可用
    model = model.to(device)
    criterion = criterion.to(device)

    train(model,
        train_iterator,
        valid_iterator,
        test_iterator,
        optimizer,
        criterion,
        scheduler,
        n_epochs,
        out_model_path,
        log_path)