from torch import nn

from layers.ema import EMA


class SeriesDecomposition(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.smoother = EMA(alpha)

    def forward(self, x):
        trend = self.smoother(x)
        seasonal = x - trend
        return seasonal, trend
