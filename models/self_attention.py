from math import sqrt

import torch
import torch.nn as nn
import numpy as np
class SelfAttention(nn.Module):
    def __init__(self, dim_q, dim_k, dim_v):
        super(SelfAttention, self).__init__()
        self.dim_q = dim_q
        self.dim_k = dim_k
        self.dim_v = dim_v

        #定义线性变换函数
        self.linear_q = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_k = nn.Linear(dim_q, dim_k, bias=False)
        self.linear_v = nn.Linear(dim_q, dim_v, bias=False)
        self._norm_fact = 1 / np.sqrt(dim_k)

    def forward(self, x):
        # x: batch, max_len, dim_q
        #根据文本获得相应的维度

        batch, n, dim_q = x.shape
        assert dim_q == self.dim_q

        q = self.linear_q(x)  # batch, max_len, dim_k
        k = self.linear_k(x)  # batch, max_len, dim_k
        v = self.linear_v(x)  # batch, max_len, dim_v
        #q*k的转置 并*开根号后的dk
        dist = torch.bmm(q, k.transpose(1, 2)) * self._norm_fact  # batch, max_len, max_len
        #归一化获得attention的相关系数
        dist = torch.softmax(dist, dim=-1)  # batch, max_len, max_len
        #attention系数和v相乘，获得最终的得分
        att = torch.bmm(dist, v) # batch, max_len, dim_v
        return att

