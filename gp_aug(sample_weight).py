#%%
import warnings
from ark_nlp.model.ner.global_pointer_bert import Tokenizer
from ark_nlp.model.ner.global_pointer_bert import Dataset as Dt
import os
import jieba
import torch
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm_notebook
import time
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np
import random
from transformers import logging
from nezha import NeZhaConfig, NeZhaModel, NeZhaForMaskedLM
from tqdm import tqdm
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # GPU
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 禁止hash随机化
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

# 设置随机数种子
RANDOM_SEED = 1998
print('随机种子：', RANDOM_SEED)
setup_seed(RANDOM_SEED)

warnings.filterwarnings("ignore")
logging.set_verbosity_warning()

dropout_num = 0
start = time.time()

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
batch_size = 8
print('batch_size: ', batch_size)
EPOCHS = 6
LR = 4e-5
print('学习率：', LR)
early_stop_epochs = 2
model_path = './my_nezha_cn_base2/'
model_save_path = 'model'
print(model_path)

#%%

config = NeZhaConfig.from_json_file(model_path + 'config.json')
config.num_labels = 53
# encoder = NeZhaForSequenceClassification(config)
encoder = NeZhaModel.from_pretrained(model_path, config=config)

import os
from ark_nlp.factory.utils.conlleval import get_entity_bio
#%%
datalist = []
with open('./datasets/preliminary_contest_datasets/train_data/train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.append('\n')

    text = []
    labels = []
    label_set = set()

    for line in lines:
        if line == '\n':
            text = ''.join(text)
            entity_labels = []
            for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                entity_labels.append({
                    'start_idx': _start_idx,
                    'end_idx': _end_idx,
                    'type': _type,
                    'entity': text[_start_idx: _end_idx + 1]
                })

            if text == '':
                continue

            datalist.append({
                'text': text,
                'label': entity_labels,
                'BIO': labels
            })

            text = []
            labels = []

        elif line == '  O\n':
            text.append(' ')
            labels.append('O')
        else:
            line = line.strip('\n').split()
            if len(line) == 1:
                term = ' '
                label = line[0]
            else:
                term, label = line
            text.append(term)
            label_set.add(label.split('-')[-1])
            labels.append(label)


print(len(datalist))
# 这里随意分割了一下看指标，建议实际使用sklearn分割或者交叉验证
#datalist =  datalist[:10000]
#train_data_df = pd.DataFrame(datalist)
# print('训练集大小：', len(train_data_df))
# train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
#
#dev_data_df = pd.DataFrame(datalist[-400:])
# dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

all_data = pd.DataFrame(datalist)
all_data.info()
all_data.head(5)




#%%
#%%

#%%

#%%
# all_data[70:]
#%%

#%%
train_data_df, dev_data_df = train_test_split(all_data, test_size = 0.1, random_state=RANDOM_SEED)
train_data_df.index = list(range(len(train_data_df)))
dev_data_df.index = list(range(len(dev_data_df)))





pseudo_datalist = []
with open('./aug_train_9.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    lines.append('\n')

    text = []
    labels = []

    for line in lines:
        if line == '\n':
            text = ''.join(text)
            entity_labels = []
            for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                entity_labels.append({
                    'start_idx': _start_idx,
                    'end_idx': _end_idx,
                    'type': _type,
                    'entity': text[_start_idx: _end_idx + 1]
                })

            if text == '':
                continue

            pseudo_datalist .append({
                'text': text,
                'label': entity_labels,
                'BIO': labels
            })

            text = []
            labels = []

        elif line == '  O\n':
            text.append(' ')
            labels.append('O')
        else:
            line = line.strip('\n').split()
            if len(line) == 1:
                term = ' '
                label = line[0]
            else:
                term, label = line
            text.append(term)
            labels.append(label)

pseudo_all_data = pd.DataFrame(pseudo_datalist)
print('伪标数据大小：', len(pseudo_all_data))





#%%
# 伪标数量
train_data_df = pd.concat([train_data_df,pseudo_all_data])
train_data_df.index = list(range(len(train_data_df)))
pseudo_data_nums = len(train_data_df)-36000
print('伪标数据有：', pseudo_data_nums)
weight_train = [1]*36000
weight_pseudo = [0.5]*(pseudo_data_nums)

# 伪标数据放在训练数据后的情况
weight_all = weight_train+weight_pseudo
train_data_df['sample_weight'] = weight_all
#%%
#pseudo_data = all_data[40000:]
#all_data = all_data[:40000]





weight_dev = [1]*4000
dev_data_df['sample_weight'] = weight_dev
dev_data_df.index = list(range(len(dev_data_df)))
train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

print('训练集大小：', len(train_data_df))
print('验证集大小：', len(dev_data_df))

label_list = sorted(list(label_set))
label_list = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36',
            '37', '38', '39', '4', '40', '41', '42', '43', '44', '46', '47', '48', '49', '5', '50', '51', '52', '53', '54', '6', '7', '8', '9', 'O']


del text
del label_set
del labels
train_dataset = Dt(train_data_df, categories=label_list)
dev_dataset = Dt(dev_data_df, categories=label_list)

tokenizer = BertTokenizer.from_pretrained(model_path)
ark_tokenizer = Tokenizer(vocab=tokenizer, max_seq_len=128)

train_dataset.convert_to_ids(ark_tokenizer)
dev_dataset.convert_to_ids(ark_tokenizer)


Ent2id = train_dataset.cat2id  # 53
id2Ent = train_dataset.id2cat


#%%

train_dataset.cat2id = {'1': 0,'10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9,
 '19': 10, '2': 11, '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17, '26': 18, '28': 19,
'29': 20, '3': 21, '30': 22, '31': 23, '32': 24, '33': 25, '34': 26, '35': 27, '36': 28, '37': 29,
'38': 30, '39': 31, '4': 32, '40': 33, '41': 34, '42': 35, '43': 36, '44': 37, '46': 38, '47': 39,
'48': 40, '49': 41, '5': 42, '50': 43, '51': 44, '52': 45, '53': 46, '54': 47, '6': 48, '7': 49, '8': 50, '9': 51, 'O': 52}
Ent2id = train_dataset.cat2id


print('Ent2id len: ', len(Ent2id))

class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_len, ark_data):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ark_data = ark_data

    def __getitem__(self, index):
        row = self.data.iloc[index]
        token_ids, at_mask, sample_weight = self.get_token_ids(row)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(at_mask, dtype=torch.long), \
               torch.tensor(self.ark_data[index]['label_ids'].to_dense()), torch.tensor(self.ark_data[index]['token_type_ids'],dtype=torch.long), \
               torch.tensor(sample_weight)

    def __len__(self):
        return len(self.data)

    def get_token_ids(self, row):
        # print(type(list(row.sample_weight)))
        # print(row.sample_weight.tolist())
        sentence = row.text
        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        padding = [0] * (self.max_len - len(tokens))
        at_mask = [1] * len(tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids + padding
        at_mask = at_mask + padding

        sample_weight = row.sample_weight
        return token_ids, at_mask, sample_weight

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        at_mask = torch.stack([x[1] for x in batch])
        labels = torch.stack([x[2] for x in batch])
        token_type_ids = torch.stack([x[3] for x in batch])
        sample_weight = torch.stack([x[4] for x in batch])

        return token_ids, at_mask, labels.squeeze(), token_type_ids, sample_weight


ner_train_dataset = Dataset(train_data_df, ark_tokenizer, MAX_LEN, train_dataset)
ner_dev_dataset = Dataset(dev_data_df, ark_tokenizer, MAX_LEN, dev_dataset)

train_loader = DataLoader(ner_train_dataset,  # 1250
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=ner_train_dataset.collate_fn,
                          num_workers=8)
dev_loader = DataLoader(ner_dev_dataset,  # 13
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=ner_dev_dataset.collate_fn,
                        num_workers=8)

#%%
#%%
#%%

#%%
class FGM(object):
    def __init__(self, module):
        self.module = module
        self.backup = {}

    def attack(
            self,
            epsilon=1.,
            emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(
            self,
            emb_name='word_embeddings'
    ):
        for name, param in self.module.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model, emb_name='word_embeddings', epsilon=1., alpha=0.3):
        # emb_name这个参数要换成你模型中embedding的参数名
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class GlobalPointerCrossEntropy(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, ):
        super(GlobalPointerCrossEntropy, self).__init__()

    @staticmethod
    def multilabel_categorical_crossentropy(y_true, y_pred):
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return neg_loss + pos_loss

    def forward(self, logits, target, sample_weight=None):
        """
        logits: [N, C, L, L]
        """
        bh = logits.shape[0] * logits.shape[1]
        target = torch.reshape(target, (bh, -1))
        logits = torch.reshape(logits, (bh, -1))
        each_loss = GlobalPointerCrossEntropy.multilabel_categorical_crossentropy(target, logits)

        if sample_weight != None:
            weight = torch.tensor([[sample_weight[i]]*53 for i in range(batch_size)]).reshape((-1,)).to(device)
            each_loss = each_loss*weight
            # each_loss.shape： (bs*53, )
        return torch.mean(each_loss)


class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true):
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > 0)):
            pred.append((b, l, start, end))
        for b, l, start, end in zip(*np.where(y_true > 0)):
            true.append((b, l, start, end))

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)
        if Y != 0:
            f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        else:
            f1, precision, recall = 2 * X / (Y + Z), 0, X / Z
        return f1, precision, recall


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size*2+256, self.ent_type_size * self.inner_dim * 2)
        self.gru = nn.GRU(input_size=768,
                          hidden_size=384,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)
        self.n_gram_fc = nn.Linear(768, 256)
        self.n_gram_tanh = nn.Tanh()

        #
        # self.fc2 = nn.Linear(768, 768)
        # self.tanh2 = nn.Tanh()
        # self.last_gru = nn.GRU(input_size=self.hidden_size*2+256,
        #                   hidden_size=768,
        #                   num_layers=1,
        #                   batch_first=True,
        #                   bidirectional=True)

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings




    def get_ngram_feats(self, x, ngram_range=1):
        # ngram_range = 1表示取前后token 1个
        n_gram_feats = []
        for idx, i in enumerate(x):
            if idx - ngram_range < 0 and idx + ngram_range > len(x) - 1:
                temp = list()
                for tidx in range(idx-ngram_range, idx+ngram_range+1):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)
            elif idx - ngram_range < 0:
                temp = list()
                for tidx in range(0, idx+ngram_range+1):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)
            elif idx + ngram_range > len(x) - 1:
                temp = list()
                for tidx in range(idx-ngram_range, len(x)):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)
            else:
                temp = list()
                for tidx in range(idx-ngram_range, idx+ngram_range+1):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)

        return n_gram_feats


    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        #         print(context_outputs[0].size())  (batch_size,max_len,hidden_size)
        #         print(context_outputs[1].size())  (batch_size,hidden_size)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]
        # last_hidden_state.shape = (bs, seq_len, 768)

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        #cls_emb = context_outputs[1].unsqueeze(1)
        # cls_emb.shape = (bs, 1, 768)
        #avg_pool = torch.mean(last_hidden_state[:,1:MAX_LEN-1,:],dim=1).unsqueeze(1)
        #avg_pool = self.tanh1(self.fc1(avg_pool))

        # avg_pool.shape = (bs, 1, 768)
        #max_pool, _ = torch.max(last_hidden_state[:,1:MAX_LEN-1,:],dim=1)
        #max_pool = max_pool.unsqueeze(1)
        #max_pool = self.tanh2(self.fc2(max_pool))
        # max_pool.shape = (bs, 1, 768)


        #deep_feature = torch.cat((cls_emb, avg_pool, max_pool), dim=-1)
        # deep_feature.shape = (bs, 1, 768*3)

        #deep_feature = deep_feature.repeat(1, last_hidden_state.shape[1], 1)
        # deep_feature.shape = (bs, seq_len, 768*3)


        #last_hidden_state = torch.cat((last_hidden_state, deep_feature), dim=-1)
        # last_hidden_state.shape = (bs, seq_len, 768*4)

        #h_0 = self._init_hidden(batchs=batch_size)
        #last_hidden_state, _ = self.gru(last_hidden_state, h_0)

        # print('last_hidden_state: ', last_hidden_state.shape)

        temp_last_hidden_state = last_hidden_state[:,1:MAX_LEN-1,:]
        # temp_last_hidden_state.shape = (bs, MAX_LEN-2, 768)
        # print('temp_last_hidden_state: ', temp_last_hidden_state.shape)
        n_gram_feats_idx = self.get_ngram_feats(list(range(temp_last_hidden_state.shape[1])), ngram_range=2)
        n_gram_feats = []

        #for i, n_gram in enumerate(n_gram_feats_idx):
            #if i == 0:
             #   n_gram_feats.append(torch.cat((torch.zeros((batch_size, 768)).to(device), temp_last_hidden_state[:, n_gram[0], :], temp_last_hidden_state[:, n_gram[1], :]), dim=-1))
           # elif i == len(n_gram_feats_idx) - 1:
            #    n_gram_feats.append(torch.cat((temp_last_hidden_state[:, n_gram[0], :], temp_last_hidden_state[:, n_gram[1], :], torch.zeros((batch_size, 768)).to(device)), dim=-1))
           # else:
            #    n_gram_feats.append(torch.cat((temp_last_hidden_state[:, n_gram[0], :], temp_last_hidden_state[:, n_gram[1], :], temp_last_hidden_state[:, n_gram[2], :]), dim=-1))

       # n_gram_feats = torch.stack(n_gram_feats, dim=1)
        # n_gram_feats.shape = (bs, MAX_LEN-2, 768*3)




        for n_gram in n_gram_feats_idx:

            temp = temp_last_hidden_state[:, n_gram[0], :]
            for i in range(1, len(n_gram)):
                # temp += temp_last_hidden_state[:, n_gram[i], :]
                temp = torch.add(temp, temp_last_hidden_state[:, n_gram[i], :])
            n_gram_feats.append(temp)


        n_gram_feats = torch.stack(n_gram_feats, dim=1)


        n_gram_feats = self.n_gram_tanh(self.n_gram_fc(n_gram_feats))

        # (bs, MAX_LEN-2, 256)


        # last_hidden_state = torch.cat((last_hidden_state[:,0,:].unsqueeze(1), n_gram_feats, last_hidden_state[:,MAX_LEN-1,:].unsqueeze(1)), dim=1)
        n_gram_feats = torch.cat((torch.zeros((batch_size,1, 256)).to(device), n_gram_feats, torch.zeros((batch_size,1, 256)).to(device)),dim=1)
        last_hidden_state = torch.cat((last_hidden_state, n_gram_feats), dim=-1)

        cls_emb = context_outputs[1].unsqueeze(1)
        h_0 = torch.randn(2, batch_size, 384).to(device)
        #cls_emb, _ = self.gru(cls_emb, h_0)
        cls_emb, _ = self.gru(cls_emb, h_0)
        cls_emb = cls_emb.repeat(1, last_hidden_state.shape[1], 1)
        last_hidden_state = torch.cat((last_hidden_state, cls_emb), dim=-1)


        # h_1 = torch.randn(2, batch_size, 768).to(device)
        # last_hidden_state, _ = self.last_gru(last_hidden_state, h_1)



        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = attention_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5
#%%







model = GlobalPointer(encoder, len(Ent2id), 64)  # (encoder, ent_type_size, inner_dim)
model = model.to(device)

#----------------------------------加载伪标训练模型-----------------------------------
#path = 'model/best_model(5w伪标训练).bin'
#print('使用伪标训练模型：', path)
#model.load_state_dict(torch.load(path))


def global_pointer_f1_score(y_true, y_pred):
    y_pred = torch.gt(y_pred, 0)
    return torch.sum(y_true * y_pred).item(), torch.sum(y_true + y_pred).item()


def validation_fn(model, dev_loader, loss_fn):
    model.eval()
    ema.apply_shadow()
    total_loss = []
    cnt = 0
    total_f1_, total_precision_, total_recall_ = 0., 0., 0.
    with torch.no_grad():
        for token_id, at_mask, label_id, token_type_ids, sample_weight in tqdm(dev_loader):
            outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device))
            loss = loss_fn(outputs, label_id.to(device), sample_weight.to(device))
            total_loss.append(loss.item())
            cnt += 1
            f1, p, r = metrics.get_evaluate_fpr(outputs, label_id.to(device))
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / cnt
        avg_precision = total_precision_ / cnt
        avg_recall = total_recall_ / cnt
        # t_loss = np.array(total_loss).mean()
    # ema.restore()
    return avg_f1, avg_precision, avg_recall


def train_model(model, train_loader, dev_loader, model_save_path='../outputs',
                early_stop_epochs=2, is_fgm=True, is_pgd=False):
    no_improve_epochs = 0

    ########优化器 学习率
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
         {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
     ]

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)



    loss_fn = GlobalPointerCrossEntropy().to(device)
    best_vmetric = 0
    if is_fgm:
        fgm = FGM(module=model)

    if is_pgd:
        pgd = PGD(model=model)
        K = 3
    for epoch in range(EPOCHS):
        total_loss = []
        total_f1 = []
        model.train()
        #bar = tqdm_notebook(train_loader)
        
        idx = -1
        for (token_id, at_mask, label_id, token_type_ids, sample_weight) in tqdm(train_loader):
            idx += 1
            outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device))
            loss = loss_fn(outputs, label_id.to(device), sample_weight.to(device))
            total_loss.append(loss.item())
            loss.backward()

            if is_fgm:
                fgm.attack()
                outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device))
                loss_fgm = loss_fn(outputs, label_id.to(device), sample_weight.to(device))
                loss_fgm.backward()
                fgm.restore()
            if is_pgd:
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device))
                    loss_pgd = loss_fn(outputs, label_id.to(device), sample_weight.to(device))
                    loss_pgd.backward()
                pgd.restore()
            f1 = metrics.get_sample_f1(outputs, label_id.to(device))
            total_f1.append(f1.item())
            if ((idx + 1) % 4) == 0:
                # optimizer the net
                optimizer.step()  # update parameters of net
                # ema.update()
                optimizer.zero_grad()  # reset gradient
                ema.update()
        ema.apply_shadow()
        t_loss = np.array(total_loss).mean()
        t_f1 = np.array(total_f1).mean()
        avg_f1, avg_precision, avg_recall = validation_fn(model, dev_loader, loss_fn)
        print('epoch:{},训练集损失t_loss:{:.6f},准确率pre:{:.6f},召回率:{:.6f},F1_eval:{:.6f}'.format(epoch, t_loss,
                                                                                           avg_precision, avg_recall,
                                                                                           avg_f1))

        model_save_path = 'model/model(9:1).bin'
        if avg_f1 > best_vmetric:
            torch.save(model.state_dict(), model_save_path)
            best_vmetric = avg_f1
            no_improve = 0
            print('improve save model!!!')
        else:
            no_improve_epochs += 1

        if no_improve_epochs == early_stop_epochs:
            print('no improve score !!! stop train !!!')
            break
        ema.restore()

#%%
ema = EMA(model, 0.995)
ema.register()
metrics = MetricsCalculator()
train_model(model, train_loader, dev_loader, early_stop_epochs=early_stop_epochs)


end = time.time()
print('共用时:', (end - start) / 60, '分钟')