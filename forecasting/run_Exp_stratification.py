import argparse
import os
import sys
import math

# Configure CUDA_VISIBLE_DEVICES before importing torch.
# Parse the device and privacy-budget options before importing torch.
temp_parser = argparse.ArgumentParser()
temp_parser.add_argument('--privacy_budget_limit', type=float, default=10.0)
temp_parser.add_argument('--devices', type=str, default='0')
temp_parser.add_argument('--gpu_reserve_mb', type=int, default=0)
temp_args, _ = temp_parser.parse_known_args()
# Select the first device supplied by the launcher.
# devices = '3'
# devices = '0'
privacy_budgets = list(range(1, 21))
device_ids = [int(id_) for id_ in temp_args.devices.split(',')]
id_gpu = int(temp_args.privacy_budget_limit)
selected_gpu = device_ids[0]
# selected_gpu = device_ids[0]

# Set CUDA_VISIBLE_DEVICES before importing torch.
os.environ["CUDA_VISIBLE_DEVICES"] = str(selected_gpu)
print(f"Set CUDA_VISIBLE_DEVICES to: {selected_gpu} (privacy_budget_limit: {temp_args.privacy_budget_limit})")

# It is now safe to import torch.
import torch
from utils.gpu_reservation import reserve_gpu_memory

# Establish the visible GPU reservation before importing the experiment/model
# stack, minimizing the window between GPU selection and memory ownership.
early_gpu_reservation_mb = 0
if temp_args.gpu_reserve_mb > 0 and torch.cuda.is_available():
    early_gpu_reservation_mb = reserve_gpu_memory(temp_args.gpu_reserve_mb)

from exp.exp_main_spaced_sampling import Exp_Main
from utils.seed_utils import set_seed
import random
import numpy as np

parser = argparse.ArgumentParser(description='SparseTSF & other models for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='SparseTSF', help='model name')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
parser.add_argument('--sampling_stride', type=int, default=None,
                    help='stride between time-series window starts (default: seq_len + pred_len)')

# SparseTSF
parser.add_argument('--period_len', type=int, default=24, help='period length')
parser.add_argument('--model_type', default='mlp', help='model type: linear/mlp')


# PatchTST
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
parser.add_argument('--norm', type=str, default='LayerNorm', help='PatchTST normalization: LayerNorm or BatchNorm')

# Formers 
parser.add_argument('--embed_type', type=int, default=0, help='0: default 1: value embedding + temporal embedding + positional embedding 2: value embedding + temporal embedding 3: value embedding + positional embedding 4: value embedding')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size') # DLinear with --individual, use this hyperparameter as the number of channels
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--distil', action='store_false',
                    help='whether to use distilling in encoder, using this argument means not using distilling',
                    default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--embed', type=str, default='learned',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--micro_batch_size', type=int, default=0, help='micro batch size for per-sample grad computation (0=use full batch)')
parser.add_argument('--patience', type=int, default=100, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type3', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, help='use multiple gpus', default=0)
parser.add_argument('--devices', type=str, default='0,1', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

# DP parameters
parser.add_argument('--dp_sigma', type=float, default=1.0, help='DP sigma')
parser.add_argument('--dp_delta', type=float, default=1e-5, help='DP delta')
# parser.add_argument('--dp_sampling_rate', type=float, default=0.1, help='DP sampling rate')
# parser.add_argument('--sensitivity', type=float, default=5, help='DP sensitivity')
parser.add_argument('--privacy_budget_limit', type=float, default=10.0, help='DP privacy budget limit')
parser.add_argument('--w', type=float, default=0.05, help='private rate')
parser.add_argument('--clipping_norm', type=float, default=0.1, help='per-sample gradient clipping norm (DP mode)')
parser.add_argument('--result_file', type=str, default=None, help='optional path to append (epsilon, mean_test_loss) results')
parser.add_argument('--gpu_reserve_mb', type=int, default=0, help='keep this many MiB in the PyTorch CUDA cache until process exit')

# Reproducibility
parser.add_argument('--seed', type=int, default=None, help='random seed for reproducibility (overrides fix_seed_list if provided)')

args = parser.parse_args()
if args.sampling_stride is None:
    args.sampling_stride = args.seq_len + args.pred_len

# random seed
fix_seed_list = range(42, 50)


args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

# The GPU was configured before importing torch; record the selected device here.
args.gpu = selected_gpu
print(f"Using GPU: {args.gpu}")
if args.use_gpu and early_gpu_reservation_mb <= 0:
    reserve_gpu_memory(args.gpu_reserve_mb)

# Auto-fix SparseTSF period_len to be compatible with seq_len and pred_len
if args.model == 'SparseTSF':
    if args.period_len <= 0 or (args.seq_len % args.period_len != 0) or (args.pred_len % args.period_len != 0):
        new_pl = math.gcd(args.seq_len, args.pred_len)
        new_pl = new_pl if new_pl > 0 else args.pred_len
        print(f"[Auto-config][SparseTSF] Adjust period_len from {args.period_len} to {new_pl} for seq_len={args.seq_len}, pred_len={args.pred_len}")
        args.period_len = new_pl

# Unless --dec_in is explicit, match dec_in to enc_in for multivariate tasks; use 1 for univariate tasks.
if not any(arg.startswith('--dec_in') for arg in sys.argv):
    if args.features in ['M', 'MS']:
        args.dec_in = args.enc_in
        print(f"[Auto-config] Set dec_in to enc_in ({args.dec_in}) for features={args.features}")
    elif args.features == 'S':
        args.enc_in = 1
        args.dec_in = 1
        if not any(arg.startswith('--c_out') for arg in sys.argv):
            args.c_out = 1
        print(f"[Auto-config] Set enc_in=dec_in=c_out=1 for features=S")

print('Args in experiment:')
# print(args)

# Use --seed if provided, otherwise use fix_seed_list
if args.seed is not None:
    fix_seed_list = [args.seed]
    args.itr = 1  # Override iterations to 1 when specific seed is provided
else:
    fix_seed_list = [42+i for i in range(args.itr)]

Exp = Exp_Main
test_losses = []


if args.is_training:
    for ii in range(args.itr):
        current_seed = fix_seed_list[ii]
        # Use unified set_seed for complete reproducibility
        set_seed(current_seed, deterministic=True)
        # Store seed in args for data loaders to use
        args.seed = current_seed
        
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_pl{}_ss{}_{}_{}_lr{}_iter{}_seed{}_strata_privacy{}_bz{}_w{}_clip{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.pred_len,
            args.sampling_stride,
            args.model_type,
            args.des,
            args.learning_rate,
            ii,
            current_seed,
            args.privacy_budget_limit,
            args.batch_size,
            args.w,
            args.clipping_norm
        )

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        test_loss = exp.test(setting)
        test_losses.append(test_loss)

        if args.do_predict:
            print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.predict(setting, True)

        if args.gpu_reserve_mb <= 0:
            torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_pl{}_ss{}_{}_{}_{}_seed{}'.format(
        args.model_id,
        args.model,
        args.data,
        args.features,
        args.seq_len,
        args.pred_len,
        args.sampling_stride,
        args.model_type,
        args.des,
        ii,
        fix_seed_list[ii])

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    if args.gpu_reserve_mb <= 0:
        torch.cuda.empty_cache()

print('>>>>>>>all experiments done<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
mean_loss = np.mean(test_losses)
print(test_losses)
print(f'\nMean Test Loss: {mean_loss:.6f}')
print('Args in experiment:')
print(args)

# Optionally append results to file for sweeping privacy budgets
if args.result_file is not None:
    try:
        os.makedirs(os.path.dirname(os.path.abspath(args.result_file)), exist_ok=True)
        with open(args.result_file, 'w') as f:
            eps_val = int(args.privacy_budget_limit) if float(args.privacy_budget_limit).is_integer() else args.privacy_budget_limit
            f.write(f"{eps_val}\t{mean_loss:.6f}\n")
        print(f"Appended result to {args.result_file}")
    except Exception as e:
        print(f"Failed to write result file {args.result_file}: {e}")
