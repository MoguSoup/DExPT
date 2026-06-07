# DExPT

Official source code for **Dual-Stream Exogenous Patch Transformer with Exponential Decomposition and Patch-wise Structural Loss (DExPT)**.

DExPT follows the long-term time-series forecasting project layout popularized by the Tsinghua time-series forecasting library: `run.py`, `data_provider/`, `exp/`, `models/`, `layers/`, `utils/`, and `scripts/`. This repository contains only the DExPT implementation and the code needed to train and evaluate it.

## Model Overview

DExPT applies reversible instance normalization, decomposes the input sequence with exponential moving average smoothing, and forecasts with two streams:

- an exogenous patch seasonal stream for local and contextual variation,
- a lightweight trend stream for smooth low-frequency movement,
- a fixed trend-biased fusion layer,
- residual multi-scale patch enhancement,
- patch-wise structural loss with selective low-uncertainty masking,
- horizon residual correction for long prediction lengths.

## Installation

```bash
conda create -n dexpt python=3.9 -y
conda activate dexpt
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version if the default pip package is not suitable for your machine.

## Data

Place datasets under `./dataset` by default. Each CSV file should contain a `date` column followed by one or more numeric variables. For a custom multivariate dataset:

```text
dataset/
  weather/
    weather.csv
```

## Training

Example command:

```bash
python -u run.py \
  --is_training 1 \
  --model_id weather_96_96 \
  --model DExPT \
  --data custom \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --train_epochs 100 \
  --patience 10
```

The default DExPT settings match the paper configuration. The main exposed model controls are:

- `--d_model`, `--n_heads`, `--e_layers`, `--d_ff`, `--dropout`
- `--stream_alpha`
- `--rmpe_gamma`, `--rmpe_patch_lens`, `--rmpe_dim`
- `--ps_lambda`, `--selective_ratio`
- `--horizon_gamma`

`--horizon_gamma auto` enables the paper default horizon residual schedule for prediction lengths 192 and 720.

## Testing

After training, test a saved setting with:

```bash
python -u run.py \
  --is_training 0 \
  --model_id weather_96_96 \
  --model DExPT \
  --data custom \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --features M \
  --enc_in 21 \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96
```

Metrics are printed in the form:

```text
mse:<value>, mae:<value>
```

The same line is appended to `result.txt` for script-friendly collection.
