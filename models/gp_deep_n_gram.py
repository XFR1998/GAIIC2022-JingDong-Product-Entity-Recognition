import torch
import torch.nn as nn
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

        temp_last_hidden_state = last_hidden_state[:, 1:MAX_LEN - 1, :]
        # temp_last_hidden_state.shape = (bs, MAX_LEN-2, 768)
        # print('temp_last_hidden_state: ', temp_last_hidden_state.shape)
        n_gram_feats_idx = self.get_ngram_feats(list(range(temp_last_hidden_state.shape[1])), ngram_range=2)
        n_gram_feats = []

        for n_gram in n_gram_feats_idx:

            temp = temp_last_hidden_state[:, n_gram[0], :]
            for i in range(1, len(n_gram)):
                # temp += temp_last_hidden_state[:, n_gram[i], :]
                temp = torch.add(temp, temp_last_hidden_state[:, n_gram[i], :])
            n_gram_feats.append(temp)

        n_gram_feats = torch.stack(n_gram_feats, dim=1)


        n_gram_feats = self.n_gram_tanh(self.n_gram_fc(n_gram_feats))

        # (bs, MAX_LEN-2, 256)

        n_gram_feats = torch.cat(
            (torch.zeros((batch_size, 1, 256)).to(device), n_gram_feats, torch.zeros((batch_size, 1, 256)).to(device)),
            dim=1)
        last_hidden_state = torch.cat((last_hidden_state, n_gram_feats), dim=-1)

        cls_emb = context_outputs[1].unsqueeze(1)
        h_0 = torch.randn(2, batch_size, 384).to(device)

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