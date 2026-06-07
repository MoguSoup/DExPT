#!/usr/bin/env bash
set -euo pipefail

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
  --patience 10 \
  --des release
