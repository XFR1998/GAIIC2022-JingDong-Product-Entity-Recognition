# %%

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

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
from tqdm import tqdm

# %%

from ark_nlp.factory.utils.conlleval import get_entity_bio

datalist = []
with open('../data/contest_data/train_data/train.txt', 'r', encoding='utf-8') as f:
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
datalist = datalist
# 这里随意分割了一下看指标，建议实际使用sklearn分割或者交叉验证
train_data_df = pd.DataFrame(datalist)
train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))

# dev_data_df = pd.DataFrame(datalist[-400:])
# dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

# train_data_df, dev_data_df = train_test_split(pd.DataFrame(datalist), test_size=0.1, random_state=42)
# train_data_df['label'] = train_data_df['label'].apply(lambda x: str(x))
# dev_data_df['label'] = dev_data_df['label'].apply(lambda x: str(x))

label_list = sorted(list(label_set))

train_dataset = Dt(train_data_df, categories=label_list)
# dev_dataset = Dt(dev_data_df, categories=label_list)
original_model_path = '../data/pretrain_model/my_nezha_cn_base/'
tokenizer = BertTokenizer.from_pretrained(original_model_path)
ark_tokenizer = Tokenizer(vocab=tokenizer, max_seq_len=128)

train_dataset.convert_to_ids(ark_tokenizer)
# dev_dataset.convert_to_ids(ark_tokenizer)

train_labels_id = []
for i in range(len(train_dataset)):
    train_labels_id.append(train_dataset[i]['label_ids'])
dev_labels_id = []
# for i in range(len(dev_dataset)):
#     dev_labels_id.append(dev_dataset[i]['label_ids'])
Ent2id = train_dataset.cat2id  # 53
id2Ent = train_dataset.id2cat


# %%


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


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size*2, self.ent_type_size * self.inner_dim * 2)
        self.gru = nn.GRU(input_size=768,
                          hidden_size=384,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=True)

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

    def _init_hidden(self, batchs):
        h_0 = torch.randn(2, batchs, 384).to(device)
        return h_0

    def forward(self, input_ids, attention_mask, token_type_ids):
        self.device = input_ids.device

        context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)

        #         print(context_outputs[0].size())  (batch_size,max_len,hidden_size)
        #         print(context_outputs[1].size())  (batch_size,hidden_size)
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        cls_emb = context_outputs[1].unsqueeze(1)
        h_0 = self._init_hidden(batchs=batch_size)
        cls_emb, _ = self.gru(cls_emb, h_0)
        cls_emb = cls_emb.repeat(1, last_hidden_state.shape[1], 1)
        last_hidden_state = torch.cat((last_hidden_state, cls_emb), dim=-1)

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


# %% md

# KF = 5 代表你使用的是5折

# %%

KF = 5
# 5折模型
for i in tqdm(range(1, KF + 1)):
    print('{}fold开始预测啦！！！'.format(i))
    config = NeZhaConfig.from_json_file(original_model_path + 'config.json')
    config.num_labels = 53
    encoder = NeZhaModel.from_pretrained(original_model_path, config=config)
    model = GlobalPointer(encoder, len(Ent2id), 64)  # (encoder, ent_type_size, inner_dim)
    model.to(device)
    model_path = '../data/model_data/nezha_train_model_{}fold.bin'.format(i)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    ner_predictor_instance = GlobalPointerNERPredictor(model, ark_tokenizer, Ent2id)

    predict_results = []
    # try.txt改成你需要伪标的数据
    with open('../data/tmp_data/26w_unlabeled_train_data.txt', 'r',
          encoding='utf-8') as f:
    #with open('./datasets/preliminary_contest_datasets/train_data/80-95w_half_unlabeled_train_data.txt', 'r',
              #encoding='utf-8') as f:
        lines = f.readlines()
        for _line in lines:
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

    # new_df = pd.DataFrame({'line': [], 'label': []})
    # for line in predict_results:
    #     new_df.loc[len(new_df)] = line
    # new_df.to_excel('./伪标数据/labels_{}fold.xlsx'.format(i), index=False)
    with open('../data/tmp_data/labels_{}fold.txt'.format(i), 'w', encoding='UTF-8') as f:
        f.write('line')
        f.write('\t')
        f.write('label')
        f.write('\n')
        for line in predict_results:
            l = line[0].rstrip('\n')
            f.write(l)
            r = str(line[1])
            f.write('\t')
            f.write(r)
            f.write('\n')




df_list = []
for i in range(1, KF+1):
    temp_df = pd.read_csv('../data/tmp_data/labels_{}fold.txt'.format(i),sep='\t')
    df_list.append(temp_df)
df1, df2, df3, df4, df5 = df_list

#%%
with open('../data/tmp_data/best_label.txt', 'w', encoding='UTF-8') as f:
    f.write('line')
    f.write('\t')
    f.write('label')
    f.write('\n')
    for i in range(len(df1)):
        # 一致投票才保存
        if df1['label'][i]==df2['label'][i]==df3['label'][i]==df4['label'][i]==df5['label'][i]:
            # new_df.loc[len(new_df)] = df1.loc[i]
            f.write(df1.loc[i]['line'])
            f.write('\t')
            f.write(df1.loc[i]['label'])
            f.write('\n')



#%%

new_df = pd.read_csv('../data/tmp_data/best_label.txt',sep='\t')
new_df['label'] = new_df['label'].apply(lambda x: eval(x))
new_array = np.array(new_df)
new_list = new_array.tolist()

#%%

with open('../data/tmp_data/pseudo_data_3w.txt', 'w', encoding='utf-8') as f:
    for _result in new_list:
        for word, tag in zip(_result[0], _result[1]):
            if word == '\n':
                continue
            f.writelines(f'{word} {tag}\n')
        f.writelines('\n')


