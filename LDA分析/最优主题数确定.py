from datetime import datetime
import time
import os
from tqdm import tqdm
import pandas as pd
import re
import matplotlib.pyplot as plt

import numpy as np
import nltk
from nltk.corpus import stopwords
import pyLDAvis.gensim_models
import jieba.posseg as jp,jieba
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import matplotlib
import jieba
from ark_nlp.factory.utils.conlleval import get_entity_bio
#%%
total_septext_list = []
total_bio_list = []
t = []
# with open('../datasets/sample_datasets/train_500.txt', encoding='UTF-8') as f:
with open('./train_9.txt', encoding='UTF-8') as f:
    line_lists = f.readlines()
    septext_list = []
    bio_list = []
    for line in tqdm(line_lists):

        l = line.split(' ')

        # 判断是否读完一条数据
        if len(l) == 1:
            total_septext_list.append(''.join(septext_list))
            total_bio_list.append(' '.join(bio_list))
            septext_list = []
            bio_list = []
        # 防止遇到这种情况：'  O'
        elif len(l) == 3:
            septext_list.append(' ')
            bio_list.append(l[2].rstrip('\n'))
        elif len(l) == 2:
            septext_list.append(l[0])
            bio_list.append(l[1].rstrip('\n'))
        else:
            print('异常数据')

#%%

len(total_septext_list),len(total_bio_list)


#%%

df = pd.DataFrame({'text': total_septext_list,
                   'bio_label': total_bio_list})


#%%


#%%
text = df['text'].to_list()
label_list = df['bio_label'].apply(lambda x: x.split(' '))
len(text)
print(text[:5])
print(label_list[:5])
#%%
# jieba分词
stopwords = [line.strip() for line in open('./hit_stopwords.txt',encoding='UTF-8').readlines()]
stopwords.append(' ')
for i in range(101):
    stopwords.append(i)
def jieba_separate_sentence(text):
    text = text.lower()
    sentence_depart = jieba.cut(text.strip())
    # 输出结果为outstr
    tokens = []
    # 去停用词
    for word in sentence_depart:
        if word not in stopwords:
            if word != '\t':
                tokens.append(word)
    return tokens



# 按实体分词
def entity_separate_sentence(text,labels):
    entity_labels = []
    for _type, _start_idx, _end_idx in get_entity_bio(labels, id2label=None):
                    entity_labels.append({
                        'start_idx': _start_idx,
                        'end_idx': _end_idx,
                        'type': _type,
                        'entity': text[_start_idx: _end_idx + 1]
                    })
    seq_list = []
    for info in entity_labels:
        seq_list.append(info['entity'])

    return seq_list
#%%

#%%

#%%
#%%
#%%
# text = [entity_separate_sentence(text=i,labels=j) for i,j in tqdm(zip(text, label_list))]
# print(len(text))
text = [jieba_separate_sentence(text=i) for i in tqdm(text)]
print(len(text))
#%%
# x = text[0]
# x
#%%
# entity_separate_sentence(x, label_list[0])
#%%
# jieba_separate_sentence(x)
#%%
#%%

#%%
# 构造词典
dictionary = Dictionary(text)
# 基于词典，使【词】→【稀疏向量】，并将向量放入列表，形成【稀疏向量集】
corpus = [dictionary.doc2bow(words) for words in text]
#计算困惑度
def perplexity(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=15))
    print(ldamodel.log_perplexity(corpus))
    return ldamodel.log_perplexity(corpus)
#计算coherence
def coherence(num_topics):
    ldamodel = LdaModel(corpus, num_topics=num_topics, id2word = dictionary, passes=30,random_state = 1)
    print(ldamodel.print_topics(num_topics=num_topics, num_words=10))
    ldacm = CoherenceModel(model=ldamodel, texts=text, dictionary=dictionary, coherence='c_v')
    print(ldacm.get_coherence())
    return ldacm.get_coherence()



#%%
x = range(1,30)
# z = [perplexity(i) for i in x]  #如果想用困惑度就选这个
y = [coherence(i) for i in tqdm(x)]

#%%
print(y)
plt.plot(x, y)
plt.xlabel('主题数目')
plt.ylabel('coherence大小')
plt.rcParams['font.sans-serif']=['SimHei']
matplotlib.rcParams['axes.unicode_minus']=False
plt.title('主题-coherence变化情况')
plt.savefig('./best topics.jpg')