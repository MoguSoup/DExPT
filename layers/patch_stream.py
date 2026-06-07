import math

import torch
import torch.nn.functional as F
from torch import nn


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class FullAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        embed_dim = queries.shape[-1]
        scale = 1.0 / math.sqrt(embed_dim)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        weights = self.dropout(torch.softmax(scale * scores, dim=-1))
        return torch.einsum("bhls,bshd->blhd", weights, values).contiguous()


class AttentionLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        head_dim = d_model // n_heads
        self.inner_attention = FullAttention(dropout)
        self.query_projection = nn.Linear(d_model, head_dim * n_heads)
        self.key_projection = nn.Linear(d_model, head_dim * n_heads)
        self.value_projection = nn.Linear(d_model, head_dim * n_heads)
        self.out_projection = nn.Linear(head_dim * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values):
        batch, query_len, _ = queries.shape
        _, key_len, _ = keys.shape
        heads = self.n_heads
        queries = self.query_projection(queries).view(batch, query_len, heads, -1)
        keys = self.key_projection(keys).view(batch, key_len, heads, -1)
        values = self.value_projection(values).view(batch, key_len, heads, -1)
        out = self.inner_attention(queries, keys, values)
        return self.out_projection(out.view(batch, query_len, -1))


class EndogenousPatchEmbedding(nn.Module):
    def __init__(self, n_vars, d_model, seq_len, patch_len, dropout):
        super().__init__()
        if patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if seq_len % patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")

        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n_vars = x.shape[1]
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        patches = patches.reshape(x.shape[0] * n_vars, self.patch_num, self.patch_len)
        patches = self.value_embedding(patches) + self.position_embedding(patches)
        patches = patches.reshape(x.shape[0], n_vars, self.patch_num, -1)

        global_token = self.global_token[:, :n_vars].repeat(x.shape[0], 1, 1, 1)
        patches = torch.cat([patches, global_token], dim=2)
        patches = patches.reshape(x.shape[0] * n_vars, self.patch_num + 1, -1)
        return self.dropout(patches), n_vars


class ExogenousContextEmbedding(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_mark):
        tokens = x.permute(0, 2, 1)
        if x_mark is not None:
            tokens = torch.cat([tokens, x_mark.permute(0, 2, 1)], dim=1)
        return self.dropout(self.value_embedding(tokens))


class PatchEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.self_attention = AttentionLayer(d_model, n_heads, dropout)
        self.cross_attention = AttentionLayer(d_model, n_heads, dropout)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross=None):
        x = x + self.dropout(self.self_attention(x, x, x))
        x = self.norm1(x)

        global_token = x[:, -1, :].unsqueeze(1)
        if cross is None:
            global_token = self.norm2(global_token)
        else:
            batch, _, d_model = cross.shape
            query = global_token.reshape(batch, -1, d_model)
            query = self.dropout(self.cross_attention(query, cross, cross))
            query = query.reshape(query.shape[0] * query.shape[1], query.shape[2]).unsqueeze(1)
            global_token = self.norm2(global_token + query)

        y = torch.cat([x[:, :-1, :], global_token], dim=1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)


class PatchEncoder(nn.Module):
    def __init__(self, d_model, n_heads, e_layers, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        if e_layers <= 0:
            raise ValueError("e_layers must be positive")
        self.layers = nn.ModuleList(
            [
                PatchEncoderLayer(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, cross=None):
        for layer in self.layers:
            x = layer(x, cross)
        return self.norm(x)


class ForecastHead(nn.Module):
    def __init__(self, head_dim, pred_len, dropout=0.1):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(head_dim, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear(self.flatten(x)))


class SeasonalPatchStream(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.enc_in <= 0:
            raise ValueError("enc_in must be positive")

        patch_num = config.seq_len // config.patch_len
        self.en_embedding = EndogenousPatchEmbedding(
            config.enc_in,
            config.d_model,
            config.seq_len,
            config.patch_len,
            config.dropout,
        )
        self.ex_embedding = ExogenousContextEmbedding(config.seq_len, config.d_model, config.dropout)
        self.encoder = PatchEncoder(
            config.d_model,
            config.n_heads,
            config.e_layers,
            config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.head = ForecastHead(config.d_model * (patch_num + 1), config.pred_len, config.dropout)

    def forward(self, seasonal, context=None, x_mark=None):
        en_embed, n_vars = self.en_embedding(seasonal.permute(0, 2, 1))
        ex_embed = self.ex_embedding(seasonal if context is None else context, x_mark)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        return self.head(enc_out).reshape(-1, self.head.linear.out_features)


class HorizonResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        patch_num = config.seq_len // config.patch_len
        self.pred_len = config.pred_len
        self.enc_in = config.enc_in
        self.en_embedding = EndogenousPatchEmbedding(
            config.enc_in,
            config.d_model,
            config.seq_len,
            config.patch_len,
            config.dropout,
        )
        self.ex_embedding = ExogenousContextEmbedding(config.seq_len, config.d_model, config.dropout)
        self.encoder = PatchEncoder(
            config.d_model,
            config.n_heads,
            config.e_layers,
            config.d_ff,
            dropout=config.dropout,
            activation=config.activation,
        )
        self.head = ForecastHead(config.d_model * (patch_num + 1), config.pred_len, config.dropout)
        self.gamma = nn.Parameter(torch.full((1, config.pred_len, 1), float(config.horizon_gamma)))

    def forward(self, x, x_mark=None):
        en_embed, n_vars = self.en_embedding(x.permute(0, 2, 1))
        ex_embed = self.ex_embedding(x, x_mark)
        enc_out = self.encoder(en_embed, ex_embed)
        enc_out = enc_out.reshape(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)
        residual = self.head(enc_out).permute(0, 2, 1)
        return self.gamma * residual
