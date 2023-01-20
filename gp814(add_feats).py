# %%

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
from make_feature.extract_word_features import *


# %%

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
batch_size = 8
print('batch_size: ', batch_size)
EPOCHS = 6
LR = 2e-5
early_stop_epochs = 2
model_path = './my_nezha_cn_base2/'

print(model_path)

config = NeZhaConfig.from_json_file(model_path + 'config.json')
config.num_labels = 53
config.output_hidden_states = True
# encoder = NeZhaForSequenceClassification(config)
encoder = NeZhaModel.from_pretrained(model_path, config=config)

import os
from ark_nlp.factory.utils.conlleval import get_entity_bio

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

# 这里随意分割了一下看指标，建议实际使用sklearn分割或者交叉验证
# datalist =  datalist[:10000]
# train_data_df = pd.DataFrame(datalist)
# print('训练集大小：', len(train_data_df))
# train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
#
# dev_data_df = pd.DataFrame(datalist[-400:])
# dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

all_data = pd.DataFrame(datalist)
# all_data = all_data[:10]

train_data_df, dev_data_df = train_test_split(all_data, test_size=0.1, random_state=RANDOM_SEED)
train_data_df.index = list(range(len(train_data_df)))
dev_data_df.index = list(range(len(dev_data_df)))
train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

print('训练集大小：', len(train_data_df))
print('验证集大小：', len(dev_data_df))

label_list = sorted(list(label_set))
del text
del label_set
del labels
# %%


# %%

train_dataset = Dt(train_data_df, categories=label_list)
dev_dataset = Dt(dev_data_df, categories=label_list)

tokenizer = BertTokenizer.from_pretrained(model_path)
ark_tokenizer = Tokenizer(vocab=tokenizer, max_seq_len=128)

train_dataset.convert_to_ids(ark_tokenizer)
dev_dataset.convert_to_ids(ark_tokenizer)

Ent2id = train_dataset.cat2id  # 53
id2Ent = train_dataset.id2cat


bound_dict = pd.read_pickle('./make_feature/feature_dict/word_bound_dict.pkl')
Bound_dict_len = len(bound_dict)

flag_dict = pd.read_pickle('./make_feature/feature_dict/word_flag_dict.pkl')
Flag_dict_len = len(flag_dict)

radical_dict = pd.read_pickle('./make_feature/feature_dict/word_radical_dict.pkl')
Radical_dict_len = len(radical_dict)

pinyin_dict = pd.read_pickle('./make_feature/feature_dict/word_pinyin_dict.pkl')
Pinyin_dict_len = len(pinyin_dict)





class Dataset(torch.utils.data.Dataset):
    def __init__(self, dataframe, tokenizer, max_len, ark_data):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.ark_data = ark_data

    def __getitem__(self, index):
        row = self.data.iloc[index]
        token_ids, at_mask, bound_ids, flag_ids, radical_ids, pinyin_ids = self.get_token_ids(row)
        return torch.tensor(token_ids, dtype=torch.long), torch.tensor(at_mask, dtype=torch.long), \
               torch.tensor(self.ark_data[index]['label_ids'].to_dense()), torch.tensor(
            self.ark_data[index]['token_type_ids'], dtype=torch.long), \
               torch.tensor(bound_ids), torch.tensor(flag_ids), torch.tensor(radical_ids), torch.tensor(pinyin_ids)

    def __len__(self):
        return len(self.data)

    def return_pinyin_ids(self, x):
        pinyin_list = extract_pinyin(x, MAX_LEN)
        pinyin_ids = [pinyin_dict[i] if i in pinyin_dict.keys() else pinyin_dict['<UNK>'] for i in pinyin_list]
        return pinyin_ids
    def return_radical_ids(self, x):
        radical_list = extract_radical(x, MAX_LEN)
        radical_ids = [radical_dict[i] if i in radical_dict.keys() else radical_dict['<UNK>'] for i in radical_list]
        return radical_ids

    def return_bound_flag_ids(self, x):
        flag_list, bound_list = extract_word_flags_bounds(x, MAX_LEN)
        flag_ids = [flag_dict[i] for i in flag_list]
        bound_ids = [bound_dict[i] for i in bound_list]
        return flag_ids, bound_ids


    def get_token_ids(self, row):
        sentence = row.text
        flag_ids, bound_ids= self.return_bound_flag_ids(sentence)
        radical_ids = self.return_radical_ids(sentence)
        pinyin_ids = self.return_pinyin_ids(sentence)

        tokens = self.tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        padding = [0] * (self.max_len - len(tokens))
        at_mask = [1] * len(tokens)
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_ids = token_ids + padding
        at_mask = at_mask + padding
        return token_ids, at_mask, bound_ids, flag_ids, radical_ids, pinyin_ids

    def collate_fn(self, batch):
        token_ids = torch.stack([x[0] for x in batch])
        at_mask = torch.stack([x[1] for x in batch])
        labels = torch.stack([x[2] for x in batch])
        token_type_ids = torch.stack([x[3] for x in batch])

        bound_ids = torch.stack([x[4] for x in batch])
        flag_ids = torch.stack([x[5] for x in batch])
        radical_ids = torch.stack([x[6] for x in batch])
        pinyin_ids = torch.stack([x[7] for x in batch])

        return token_ids, at_mask, labels.squeeze(), token_type_ids, bound_ids, flag_ids, radical_ids, pinyin_ids


ner_train_dataset = Dataset(train_data_df, ark_tokenizer, MAX_LEN, train_dataset)
ner_dev_dataset = Dataset(dev_data_df, ark_tokenizer, MAX_LEN, dev_dataset)

train_loader = DataLoader(ner_train_dataset,  # 1250
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=ner_train_dataset.collate_fn)
dev_loader = DataLoader(ner_dev_dataset,  # 13
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=ner_dev_dataset.collate_fn)


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

    def forward(self, logits, target):
        """
        logits: [N, C, L, L]
        """
        bh = logits.shape[0] * logits.shape[1]
        target = torch.reshape(target, (bh, -1))
        logits = torch.reshape(logits, (bh, -1))
        return torch.mean(GlobalPointerCrossEntropy.multilabel_categorical_crossentropy(target, logits))


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
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.lstm = nn.LSTM(input_size=768,
                            hidden_size=384,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        self.bound_embedding = nn.Embedding(Bound_dict_len, 768)
        self.flag_embedding = nn.Embedding(Flag_dict_len, 768)
        self.radical_embedding = nn.Embedding(Radical_dict_len, 768)
        self.pinyin_embedding = nn.Embedding(Pinyin_dict_len, 768)

        # self.gru = nn.GRU(input_size=768,
        #         hidden_size=384,
        #        num_layers=1,
        #       batch_first=True,
        #      bidirectional=True)
        # self.n_gram_fc = nn.Linear(768, 256)
        # self.n_gram_tanh = nn.Tanh()

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


    def forward(self, input_ids, attention_mask, token_type_ids,
                bound_ids, flag_ids, radical_ids, pinyin_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        last_hidden_state = context_outputs[0]
        # last_hidden_state.shape = (bs, seq_len, 768)

        emb_hidden_state = context_outputs[2][0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # 特征融合
        # radical_feat = self.radical_embedding(radical_ids)
        bound_feat = self.bound_embedding(bound_ids)
        flag_feat = self.flag_embedding(flag_ids)
        radical_feat = self.radical_embedding(radical_ids)
        pinyin_feat = self.pinyin_embedding(pinyin_ids)

        feats = emb_hidden_state + bound_feat + flag_feat + radical_feat + pinyin_feat

        h_0 = torch.randn(2, batch_size, 384).to(device)
        c_0 = torch.randn(2, batch_size, 384).to(device)
        feats, (_, _) = self.lstm(feats, (h_0, c_0))

        # last_hidden_state = torch.cat((last_hidden_state, feats), dim=-1)
        last_hidden_state = last_hidden_state + feats

        # last_hidden_state = torch.cat((last_hidden_state, feats), dim=-1)
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


model = GlobalPointer(encoder, len(Ent2id), 64)  # (encoder, ent_type_size, inner_dim)
model = model.to(device)


# %%
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
        for token_id, at_mask, label_id, token_type_ids, bound_ids, flag_ids, radical_ids, pinyin_ids in dev_loader:
            outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device),
                            bound_ids.to(device), flag_ids.to(device), radical_ids.to(device), pinyin_ids.to(device))
            loss = loss_fn(outputs, label_id.to(device))
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
        bar = tqdm_notebook(train_loader)

        for idx, (token_id, at_mask, label_id, token_type_ids, bound_ids, flag_ids, radical_ids, pinyin_ids) in enumerate(bar):
            outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device),
                            bound_ids.to(device), flag_ids.to(device), radical_ids.to(device), pinyin_ids.to(device))
            loss = loss_fn(outputs, label_id.to(device))
            total_loss.append(loss.item())
            loss.backward()
            bar.set_postfix(loss=loss.item())
            if is_fgm:
                fgm.attack()
                outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device),
                                bound_ids.to(device), flag_ids.to(device), radical_ids.to(device), pinyin_ids.to(device))
                loss_fgm = loss_fn(outputs, label_id.to(device))
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
                    outputs = model(token_id.to(device), at_mask.to(device), token_type_ids.to(device),
                                    bound_ids.to(device), flag_ids.to(device), radical_ids.to(device), pinyin_ids.to(device))
                    loss_pgd = loss_fn(outputs, label_id.to(device))
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

        model_save_path = 'model/best_model(add_pinyin).bin'
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


ema = EMA(model, 0.995)
ema.register()
metrics = MetricsCalculator()
train_model(model, train_loader, dev_loader, early_stop_epochs=early_stop_epochs)