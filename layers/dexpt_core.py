import torch
from torch import nn

from layers.patch_stream import SeasonalPatchStream


def parse_int_list(value):
    if isinstance(value, str):
        value = [item.strip() for item in value.split(",") if item.strip()]
    return [int(item) for item in value]


class MultiScalePatchLayer(nn.Module):
    def __init__(self, seq_len, patch_len, scale_dim):
        super().__init__()
        if patch_len <= 0:
            raise ValueError("multi-scale patch lengths must be positive")
        if patch_len > seq_len:
            raise ValueError("multi-scale patch length cannot exceed seq_len")

        self.patch_len = patch_len
        self.patch_step = max(1, patch_len // 2)
        self.patch_num = (seq_len - patch_len) // self.patch_step + 1
        patch_dim = scale_dim // self.patch_num
        if patch_dim <= 0:
            raise ValueError("rmpe_dim is too small for the requested patch lengths")

        self.ff = nn.Linear(patch_len, patch_dim)
        self.flatten = nn.Flatten(start_dim=-2)
        self.ff_out = nn.Linear(patch_dim * self.patch_num, scale_dim)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_step)
        x = self.ff(x)
        x = self.flatten(x)
        return self.ff_out(x)


class MultiScalePatchEmbedding(nn.Module):
    def __init__(self, seq_len, patch_lens, d_model):
        super().__init__()
        patch_lens = parse_int_list(patch_lens)
        if not patch_lens:
            raise ValueError("rmpe_patch_lens must contain at least one patch length")

        scale_dim = d_model // len(patch_lens)
        if scale_dim <= 0:
            raise ValueError("rmpe_dim must be at least the number of patch lengths")

        self.output_dim = scale_dim * len(patch_lens)
        self.layers = nn.ModuleList(
            [MultiScalePatchLayer(seq_len, patch_len, scale_dim) for patch_len in patch_lens]
        )

    def forward(self, x):
        return torch.cat([layer(x) for layer in self.layers], dim=-1)


class TrendStream(nn.Module):
    def __init__(self, seq_len, pred_len):
        super().__init__()
        self.fc1 = nn.Linear(seq_len, pred_len * 4)
        self.pool1 = nn.AvgPool1d(kernel_size=2)
        self.norm1 = nn.LayerNorm(pred_len * 2)
        self.fc2 = nn.Linear(pred_len * 2, pred_len)
        self.pool2 = nn.AvgPool1d(kernel_size=2)
        self.norm2 = nn.LayerNorm(pred_len // 2)
        self.fc3 = nn.Linear(pred_len // 2, pred_len)

    def forward(self, x):
        x = self.fc1(x)
        x = self.pool1(x)
        x = self.norm1(x)
        x = self.fc2(x)
        x = self.pool2(x)
        x = self.norm2(x)
        return self.fc3(x)

    def head_parameters(self):
        return self.fc3.parameters()


class DExPTCore(nn.Module):
    def __init__(self, config):
        super().__init__()
        if not 0.0 <= config.stream_alpha <= 1.0:
            raise ValueError("stream_alpha must be in [0, 1]")

        self.pred_len = config.pred_len
        self.stream_alpha = float(config.stream_alpha)
        self.rmpe_gamma = nn.Parameter(torch.tensor(float(config.rmpe_gamma)))
        self.seasonal_stream = SeasonalPatchStream(config)
        self.trend_stream = TrendStream(config.seq_len, config.pred_len)
        self.rmpe_embedding = MultiScalePatchEmbedding(
            config.seq_len,
            config.rmpe_patch_lens,
            config.rmpe_dim,
        )
        self.rmpe_head = nn.Sequential(
            nn.Linear(self.rmpe_embedding.output_dim, config.pred_len * 2),
            nn.GELU(),
            nn.Linear(config.pred_len * 2, config.pred_len),
        )

    def forward(self, seasonal, trend, x_mark=None, context=None):
        seasonal_series = seasonal
        batch_size, seq_len, channels = seasonal.shape

        seasonal = seasonal.permute(0, 2, 1).reshape(batch_size * channels, seq_len)
        trend = trend.permute(0, 2, 1).reshape(batch_size * channels, seq_len)

        seasonal_out = self.seasonal_stream(seasonal_series, context, x_mark)
        seasonal_out = seasonal_out + self.rmpe_gamma * self.rmpe_head(self.rmpe_embedding(seasonal))
        trend_out = self.trend_stream(trend)

        out = self.stream_alpha * trend_out + (1.0 - self.stream_alpha) * seasonal_out
        out = out.reshape(batch_size, channels, self.pred_len)
        return out.permute(0, 2, 1)

    def structural_head_parameters(self):
        params = list(self.seasonal_stream.head.parameters())
        params.extend(self.trend_stream.head_parameters())
        return params
