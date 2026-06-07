import argparse
import random

import numpy as np
import torch

from exp.exp_main import Exp_Main


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser():
    parser = argparse.ArgumentParser(description="DExPT long-term forecasting")

    parser.add_argument("--is_training", type=int, required=True, help="1 for train+test, 0 for test only")
    parser.add_argument("--train_only", type=str2bool, default=False, help="use all available data for training")
    parser.add_argument("--model_id", type=str, required=True, help="experiment identifier")
    parser.add_argument("--model", type=str, default="DExPT", choices=["DExPT"], help="model name")

    parser.add_argument("--data", type=str, required=True, help="dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset", help="dataset root directory")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv", help="dataset file name")
    parser.add_argument("--features", type=str, default="M", choices=["M", "S", "MS"], help="forecasting task")
    parser.add_argument("--target", type=str, default="OT", help="target column for S or MS tasks")
    parser.add_argument("--freq", type=str, default="h", help="time feature frequency")
    parser.add_argument("--embed", type=str, default="timeF", help="time feature encoding")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="checkpoint directory")

    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="label length used by the data loader")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction length")
    parser.add_argument("--enc_in", type=int, default=7, help="number of input variables")

    parser.add_argument("--patch_len", type=int, default=16, help="non-overlapping patch length")
    parser.add_argument("--d_model", type=int, default=128, help="hidden dimension")
    parser.add_argument("--n_heads", type=int, default=4, help="attention heads")
    parser.add_argument("--e_layers", type=int, default=1, help="encoder layers")
    parser.add_argument("--d_ff", type=int, default=256, help="feed-forward dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu"], help="activation")
    parser.add_argument("--ema_alpha", type=float, default=0.3, help="EMA smoothing factor")
    parser.add_argument("--stream_alpha", type=float, default=0.70, help="trend stream fusion weight")
    parser.add_argument("--rmpe_gamma", type=float, default=0.01, help="residual multi-scale patch scale")
    parser.add_argument("--rmpe_patch_lens", type=str, default="48,24,12,6", help="multi-scale patch lengths")
    parser.add_argument("--rmpe_dim", type=int, default=1024, help="multi-scale patch hidden dimension")
    parser.add_argument(
        "--horizon_gamma",
        type=str,
        default="auto",
        help="horizon residual scale; use auto for the paper default or a non-negative float",
    )

    parser.add_argument("--ps_lambda", type=float, default=1.0, help="patch-wise structural loss weight")
    parser.add_argument("--patch_len_threshold", type=int, default=24, help="maximum adaptive structural patch length")
    parser.add_argument("--selective_ratio", type=float, default=0.8, help="low-uncertainty step ratio")
    parser.add_argument("--selective_warmup", type=int, default=1, help="warmup epochs for selective masking")

    parser.add_argument("--num_workers", type=int, default=0, help="data loader workers")
    parser.add_argument("--itr", type=int, default=1, help="number of repeated runs")
    parser.add_argument("--train_epochs", type=int, default=100, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=0.0005, help="optimizer learning rate")
    parser.add_argument("--des", type=str, default="Exp", help="experiment description")
    parser.add_argument("--lradj", type=str, default="sigmoid", help="learning-rate schedule")
    parser.add_argument("--revin", type=str2bool, default=True, help="use reversible instance normalization")

    parser.add_argument("--use_gpu", type=str2bool, default=True, help="use GPU when available")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--use_multi_gpu", action="store_true", default=False, help="use multiple GPUs")
    parser.add_argument("--devices", type=str, default="0,1,2,3", help="multi-GPU device ids")
    return parser


def set_seed(seed=2021):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def experiment_setting(args, iteration):
    return "{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}_{}".format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.label_len,
        args.pred_len,
        args.des,
        iteration,
    )


def main():
    set_seed()
    args = build_parser().parse_args()
    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(device) for device in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("Args in experiment:")
    print(args)

    if args.is_training:
        for iteration in range(args.itr):
            setting = experiment_setting(args, iteration)
            exp = Exp_Main(args)
            print(">>>>>>> start training: {} >>>>>>>>>>>>>>>>>>>>>>>>>>".format(setting))
            exp.train(setting)
            print(">>>>>>> testing: {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
            exp.test(setting)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        setting = experiment_setting(args, 0)
        exp = Exp_Main(args)
        print(">>>>>>> testing: {} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<".format(setting))
        exp.test(setting, test=1)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
