import torch
import numpy as np
import torch.nn as nn

class MetricsCalculator(object):
    def __init__(self):
        super().__init__()

    def get_sample_f1(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return 2 * torch.sum(y_true * y_pred) / torch.sum(y_true + y_pred)

    def get_sample_precision(self, y_pred, y_true):
        y_pred = torch.gt(y_pred, 0).float()
        return torch.sum(y_pred[y_true == 1]) / (y_pred.sum() + 1)

    def get_evaluate_fpr(self, y_pred, y_true, length_info, text=None):
        y_pred = y_pred.cpu().numpy()
        length = torch.sum(length_info > 0, dim=-1)
        pred = []
        true = []
        for b, l, start, end in zip(*np.where(y_pred > -1)):
            if end >= length[b]-1 or end <= start: continue
            if text and text[b][end] == '[SEP]': continue
            pred.append((b, l, start, end))
        if type(y_true) != type([]):
            y_true = y_true.cpu().numpy()
            for b, l, start, end in zip(*np.where(y_true > 0)):
                true.append((b, l, start, end))
        else:
            true = y_true

        R = set(pred)
        T = set(true)
        X = len(R & T)
        Y = len(R)
        Z = len(T)

        if text is not None and X < Z:
            misses = T - R
            errors = R - T
            if misses:
                for i in range(len(text)):
                    f = 0
                    for miss in misses:
                        if miss[0] == i:
                            f += 1
                            if f == 1:
                                print(text[i][:length[i]])
                                print('missed:')
                            print(text[miss[0]][miss[2]:miss[3]+1])

                    if f > 0:
                        print('errors:')
                        for error in errors:
                            if error[0] == i:
                                print(text[error[0]][error[2]:error[3] + 1])


        # f1, precision, recall = 2 * X / (Y + Z + 1e-8), X / (Y + 1e-8), X / (Z + 1e-8)
        # return f1, precision, recall        
        return X, Y, Z


class GlobalPointer(nn.Module):
    def __init__(self, encoder, ent_type_size, inner_dim, RoPE=True):
        super().__init__()
        self.encoder = encoder
        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = encoder.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)  # 10 * 64 * 2 = 1280

        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)  # [seq, 1]

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)  # [32, ]
        indices = torch.pow(10000, -2 * indices / output_dim)  # [32, ]
        embeddings = position_ids * indices  # [seq, 32]
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def forward(self, input_ids, attention_mask, token_type_ids, append_params=None):
        self.device = input_ids.device

        if append_params:
            context_outputs = self.encoder(input_ids, attention_mask, token_type_ids, append_params=append_params)
        else:
            context_outputs = self.encoder(input_ids, attention_mask, token_type_ids)
        last_hidden_state = context_outputs[0]

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)  # 分割列表，中间的参数可以是分割数、或每一组的维长度；
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
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)  # [bat, seq, 10, 32, 2]
            qw2 = qw2.reshape(qw.shape)  # [bat, seq, 10, 64]
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
