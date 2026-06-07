from dataclasses import dataclass

from torch import nn

from layers.decomp import SeriesDecomposition
from layers.dexpt_core import DExPTCore
from layers.patch_stream import HorizonResidual
from layers.revin import RevIN


def parse_horizon_gamma(value, pred_len):
    if isinstance(value, (int, float)):
        return float(value)

    value = str(value).strip().lower()
    if value == "auto":
        return {192: 0.003, 720: 0.001}.get(int(pred_len), 0.0)

    gamma = float(value)
    if gamma < 0:
        raise ValueError("horizon_gamma must be non-negative or auto")
    return gamma


@dataclass(frozen=True)
class DExPTConfig:
    seq_len: int
    pred_len: int
    enc_in: int
    patch_len: int
    d_model: int
    n_heads: int
    e_layers: int
    d_ff: int
    dropout: float
    activation: str
    ema_alpha: float
    stream_alpha: float
    rmpe_gamma: float
    rmpe_patch_lens: str
    rmpe_dim: int
    horizon_gamma: float
    revin: bool

    @classmethod
    def from_args(cls, args):
        config = cls(
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            enc_in=args.enc_in,
            patch_len=args.patch_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            e_layers=args.e_layers,
            d_ff=args.d_ff,
            dropout=args.dropout,
            activation=args.activation,
            ema_alpha=args.ema_alpha,
            stream_alpha=args.stream_alpha,
            rmpe_gamma=args.rmpe_gamma,
            rmpe_patch_lens=args.rmpe_patch_lens,
            rmpe_dim=args.rmpe_dim,
            horizon_gamma=parse_horizon_gamma(args.horizon_gamma, args.pred_len),
            revin=bool(args.revin),
        )
        config.validate()
        return config

    def validate(self):
        if self.seq_len <= 0 or self.pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive")
        if self.enc_in <= 0:
            raise ValueError("enc_in must be positive")
        if self.patch_len <= 0:
            raise ValueError("patch_len must be positive")
        if self.seq_len % self.patch_len != 0:
            raise ValueError("seq_len must be divisible by patch_len")
        if self.n_heads <= 0 or self.e_layers <= 0 or self.d_ff <= 0:
            raise ValueError("n_heads, e_layers, and d_ff must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.pred_len < 2:
            raise ValueError("pred_len must be at least 2")
        if not 0.0 < self.ema_alpha < 1.0:
            raise ValueError("ema_alpha must be in (0, 1)")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = DExPTConfig.from_args(args)
        self.revin = self.config.revin
        self.revin_layer = RevIN(self.config.enc_in, affine=True, subtract_last=False)
        self.decomp = SeriesDecomposition(self.config.ema_alpha)
        self.net = DExPTCore(self.config)
        self.horizon_residual = (
            HorizonResidual(self.config) if self.config.horizon_gamma > 0.0 else None
        )

    def forward(self, x, x_mark_enc=None):
        if self.revin:
            x = self.revin_layer(x, "norm")

        residual_context = x
        seasonal, trend = self.decomp(x)
        x = self.net(seasonal, trend, x_mark_enc, residual_context)

        if self.horizon_residual is not None:
            x = x + self.horizon_residual(residual_context, x_mark_enc)

        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x
