import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_in, d_model, nhead, dim_feedforward=256, dropout=0.1, activation="gelu", class_num=1):
        super().__init__()

        self.nhead = nhead
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.class_num = class_num
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.k = nn.Linear(d_in, d_model)
        self.v = nn.Linear(d_in, d_model)
        self.q = nn.Linear(d_in, d_model)
        self.proj = nn.Linear(d_model, d_model)

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _get_activation_fn(self, activation):
        """Return an activation function given a string"""
        if activation == "relu":
            return F.relu
        if activation == "gelu":
            return F.gelu
        if activation == "glu":
            return F.glu
        raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

    def key_masking(self, token, key):  # token, key: (N, C)
        key_padding_mask = (key == 0.).all(dim=-1)  # (N,)
        valid_indices = (~key_padding_mask).nonzero(as_tuple=False).squeeze(1)

        if valid_indices.numel() == 0:
            return None, None, None, None

        return token, key, key_padding_mask, valid_indices


    def forward(self, token, key):

        valid_token, valid_key, valid_key_padding_mask, valid_batch_indices = self.key_masking(token, key)

        if valid_key_padding_mask is None:
            return token

        N, C = valid_token.shape

        q = self.q(valid_token)
        k = self.k(valid_key)
        v = self.v(valid_key)

        q = rearrange(q, 'n (h d) -> h n d', h=self.nhead)
        k = rearrange(k, 'n (h d) -> h n d', h=self.nhead)
        v = rearrange(v, 'n (h d) -> h n d', h=self.nhead)

        attn_mask = ~valid_key_padding_mask  # (S)
        attn_mask = attn_mask[None, None, :]  # (1,1,S)
        attn_mask = attn_mask.expand(1, self.nhead, N, -1)

        # cross attn
        src2 = F.scaled_dot_product_attention(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), attn_mask=attn_mask, scale=(C // self.nhead) ** -0.5, dropout_p=self.dropout_p)
        src2 = rearrange(src2, 'b h n d -> (b n) (h d)')
        src2 = self.proj(src2)
        q = rearrange(q, 'h n d -> n (h d)')
        src = q + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = src + self.dropout(self.activation(self.linear1(src)))

        src = src.squeeze(0)

        # skip
        token = token + src

        return token

class SceneMemory(nn.Module):
    def __init__(self, n_e, e_dim):
        super(SceneMemory, self).__init__()
        self.n_e = n_e
        self.e_dim = e_dim

        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

        self.module = TransformerEncoderLayer(e_dim, e_dim, 4) #4

    def forward(self, z):

        B, K, C = z.shape
        z_flattened = z.view(-1, C)

        out = self.module(self.embedding.weight, z_flattened)

        out = out.unsqueeze(0).repeat(B, 1, 1)

        return out