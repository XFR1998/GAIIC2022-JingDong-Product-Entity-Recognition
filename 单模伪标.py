import warnings

warnings.filterwarnings("ignore")
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

logging.set_verbosity_warning()

from nezha import NeZhaConfig, NeZhaModel, NeZhaForMaskedLM

dropout_num = 0
start = time.time()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
batch_size = 8
EPOCHS = 6
LR = 4e-5
early_stop_epochs = 2
model_path = './my_nezha_cn_base2/'
#model_path = './nezha-cn-base-wsl/'
model_save_path = 'model'
print(model_path)

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
setup_seed(1998)


config = NeZhaConfig.from_json_file(model_path + 'config.json')
config.num_labels = 53
# encoder = NeZhaForSequenceClassification(config)
# encoder = NeZhaModel.from_pretrained(model_path, config=config)
tokenizer = BertTokenizer.from_pretrained(model_path)
ark_tokenizer = Tokenizer(vocab=tokenizer, max_seq_len=128)
import os
from ark_nlp.factory.utils.conlleval import get_entity_bio


Ent2id = {'1': 0,'10': 1, '11': 2, '12': 3, '13': 4, '14': 5, '15': 6, '16': 7, '17': 8, '18': 9,
 '19': 10, '2': 11, '20': 12, '21': 13, '22': 14, '23': 15, '24': 16, '25': 17, '26': 18, '28': 19,
'29': 20, '3': 21, '30': 22, '31': 23, '32': 24, '33': 25, '34': 26, '35': 27, '36': 28, '37': 29,
'38': 30, '39': 31, '4': 32, '40': 33, '41': 34, '42': 35, '43': 36, '44': 37, '46': 38, '47': 39,
'48': 40, '49': 41, '5': 42, '50': 43, '51': 44, '52': 45, '53': 46, '54': 47, '6': 48, '7': 49, '8': 50, '9': 51, 'O': 52}

print('Ent2id len: ', len(Ent2id))


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size * 2 + 256, self.ent_type_size * self.inner_dim * 2)
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
                for tidx in range(idx - ngram_range, idx + ngram_range + 1):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)
            elif idx - ngram_range < 0:
                temp = list()
                for tidx in range(0, idx + ngram_range + 1):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)
            elif idx + ngram_range > len(x) - 1:
                temp = list()
                for tidx in range(idx - ngram_range, len(x)):
                    temp.append(x[tidx])
                n_gram_feats.append(temp)
            else:
                temp = list()
                for tidx in range(idx - ngram_range, idx + ngram_range + 1):
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

        # cls_emb = context_outputs[1].unsqueeze(1)
        # cls_emb.shape = (bs, 1, 768)
        # avg_pool = torch.mean(last_hidden_state[:,1:MAX_LEN-1,:],dim=1).unsqueeze(1)
        # avg_pool = self.tanh1(self.fc1(avg_pool))

        # avg_pool.shape = (bs, 1, 768)
        # max_pool, _ = torch.max(last_hidden_state[:,1:MAX_LEN-1,:],dim=1)
        # max_pool = max_pool.unsqueeze(1)
        # max_pool = self.tanh2(self.fc2(max_pool))
        # max_pool.shape = (bs, 1, 768)

        # deep_feature = torch.cat((cls_emb, avg_pool, max_pool), dim=-1)
        # deep_feature.shape = (bs, 1, 768*3)

        # deep_feature = deep_feature.repeat(1, last_hidden_state.shape[1], 1)
        # deep_feature.shape = (bs, seq_len, 768*3)

        # last_hidden_state = torch.cat((last_hidden_state, deep_feature), dim=-1)
        # last_hidden_state.shape = (bs, seq_len, 768*4)

        # h_0 = self._init_hidden(batchs=batch_size)
        # last_hidden_state, _ = self.gru(last_hidden_state, h_0)

        # print('last_hidden_state: ', last_hidden_state.shape)

        temp_last_hidden_state = last_hidden_state[:, 1:MAX_LEN - 1, :]
        # temp_last_hidden_state.shape = (bs, MAX_LEN-2, 768)
        # print('temp_last_hidden_state: ', temp_last_hidden_state.shape)
        n_gram_feats_idx = self.get_ngram_feats(list(range(temp_last_hidden_state.shape[1])), ngram_range=2)
        n_gram_feats = []

        # for i, n_gram in enumerate(n_gram_feats_idx):
        # if i == 0:
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
        n_gram_feats = torch.cat(
            (torch.zeros((batch_size, 1, 256)).to(device), n_gram_feats, torch.zeros((batch_size, 1, 256)).to(device)),
            dim=1)
        last_hidden_state = torch.cat((last_hidden_state, n_gram_feats), dim=-1)

        cls_emb = context_outputs[1].unsqueeze(1)
        h_0 = torch.randn(2, batch_size, 384).to(device)
        # cls_emb, _ = self.gru(cls_emb, h_0)
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


import json
import torch
import numpy as np


# ark-nlp提供该函数：from ark_nlp.model.ner.global_pointer_bert import Predictor
# 这里主要是为了可以比较清晰地看到解码过程，所以将代码copy到这
class GlobalPointerNERPredictor(object):
    """
    GlobalPointer命名实体识别的预测器

    Args:
        module: 深度学习模型
        tokernizer: 分词器
        cat2id (:obj:`dict`): 标签映射
    """  # noqa: ignore flake8"

    def __init__(
            self,
            module,
            tokernizer,
            cat2id
    ):
        self.module = module
        self.module.task = 'TokenLevel'

        self.cat2id = cat2id
        self.tokenizer = tokernizer
        self.device = list(self.module.parameters())[0].device

        self.id2cat = {}
        for cat_, idx_ in self.cat2id.items():
            self.id2cat[idx_] = cat_

    def _convert_to_transfomer_ids(
            self,
            text
    ):

        tokens = self.tokenizer.tokenize(text)
        token_mapping = self.tokenizer.get_token_mapping(text, tokens)

        tokens = ['[CLS]'] + tokens + ['[SEP]']

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        padding = [0] * (self.tokenizer.max_seq_len - len(tokens))
        segment_ids = segment_ids + padding
        input_mask = [1] * len(tokens) + padding
        input_ids = input_ids + padding

        zero = [0 for i in range(self.tokenizer.max_seq_len)]
        span_mask = [input_mask for i in range(sum(input_mask))]
        span_mask.extend([zero for i in range(sum(input_mask), self.tokenizer.max_seq_len)])
        span_mask = np.array(span_mask)

        features = {
            'input_ids': input_ids,
            'attention_mask': input_mask,
            'token_type_ids': segment_ids,
            'span_mask': span_mask
        }

        return features, token_mapping

    def _get_input_ids(
            self,
            text
    ):
        if self.tokenizer.tokenizer_type == 'vanilla':
            return self._convert_to_vanilla_ids(text)
        elif self.tokenizer.tokenizer_type == 'transfomer':
            return self._convert_to_transfomer_ids(text)
        elif self.tokenizer.tokenizer_type == 'customized':
            return self._convert_to_customized_ids(text)
        else:
            raise ValueError("The tokenizer type does not exist")

    def _get_module_one_sample_inputs(
            self,
            features
    ):
        return {col: torch.Tensor(features[col]).type(torch.long).unsqueeze(0).to(self.device) for col in features}

    def predict_one_sample(
            self,
            text='',
            threshold=0
    ):
        """
        单样本预测

        Args:
            text (:obj:`string`): 输入文本
            threshold (:obj:`float`, optional, defaults to 0): 预测的阈值
        """  # noqa: ignore flake8"

        features, token_mapping = self._get_input_ids(text)
        self.module.eval()

        with torch.no_grad():
            inputs = self._get_module_one_sample_inputs(features)
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            token_type_ids = inputs['token_type_ids']

            scores = self.module(input_ids, attention_mask, token_type_ids)[0].cpu()

        scores[:, [0, -1]] -= np.inf
        scores[:, :, [0, -1]] -= np.inf

        entities = []

        for category, start, end in zip(*np.where(scores > threshold)):
            if end - 1 > token_mapping[-1][-1]:
                break
            if token_mapping[start - 1][0] <= token_mapping[end - 1][-1]:
                entitie_ = {
                    "start_idx": token_mapping[start - 1][0],
                    "end_idx": token_mapping[end - 1][-1],
                    "entity": text[token_mapping[start - 1][0]: token_mapping[end - 1][-1] + 1],
                    "type": self.id2cat[category]
                }

                if entitie_['entity'] == '':
                    continue

                entities.append(entitie_)

        return entities




# 加载最优模型
print('*' * 20, 'Starting Predicting........', '*' * 20)
encoder = NeZhaModel.from_pretrained(model_path, config=config)
model = GlobalPointer(encoder, len(Ent2id), 64)  # (encoder, ent_type_size, inner_dim)
model.to(device)
path = 'model/best_model(9:1).bin'
print('使用最牛模型：', path)
model.load_state_dict(torch.load(path))
model.eval()
ner_predictor_instance = GlobalPointerNERPredictor(model, ark_tokenizer, Ent2id)

from tqdm import tqdm

predict_results = []

with open('./datasets/preliminary_contest_datasets/train_data/unlabeled_train_data.txt', 'r',
          encoding='utf-8') as f:
    lines = f.readlines()
    lines = lines[:50000]
    print('取{}数据伪标'.format(len(lines)))
    for _line in tqdm(lines):
        label = len(_line) * ['O']
        for _preditc in ner_predictor_instance.predict_one_sample(_line[:-1]):
            if 'I' in label[_preditc['start_idx']]:
                continue
            if 'B' in label[_preditc['start_idx']] and 'O' not in label[_preditc['end_idx']]:
                continue
            if 'O' in label[_preditc['start_idx']] and 'B' in label[_preditc['end_idx']]:
                continue

            label[_preditc['start_idx']] = 'B-' + _preditc['type']
            label[_preditc['start_idx'] + 1: _preditc['end_idx'] + 1] = (_preditc['end_idx'] - _preditc[
                'start_idx']) * [('I-' + _preditc['type'])]

        predict_results.append([_line, label])

with open('./单模伪标数据/pseudo_5w(单模).txt', 'w', encoding='utf-8') as f:
    for _result in predict_results:
        for word, tag in zip(_result[0], _result[1]):
            if word == '\n':
                continue
            f.writelines(f'{word} {tag}\n')
        f.writelines('\n')
end = time.time()
print('共用时:', (end - start) / 60, '分钟')
