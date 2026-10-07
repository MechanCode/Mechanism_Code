import argparse
import os
import sys
import random
import torch
import numpy as np
from exp.baseline_TSC import Exp_Baseline_TSC as Exp


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Time Series Classification with Rocket (DP-SGD)')

# basic config
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model', type=str, default='Rocket', help='model name')

# data loader
# parser.add_argument('--root_path', type=str, default='./dataset/WISDM/', help='root path of the data file')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/baseline/', help='location of model checkpoints')

# forecasting task (reused for classification window)
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--sampling_stride', type=int, default=None,
                    help='stride between training-window starts (default: seq_len)')

# Rocket
parser.add_argument('--num_kernels', type=int, default=1000, help='number of kernels for Rocket')

# Model parameters
parser.add_argument('--enc_in', type=int, default=3, help='encoder input size (channels)') 
parser.add_argument('--num_classes', type=int, default=3, help='number of classes')
parser.add_argument('--individual', type=int, default=0, help='individual model head; True 1 False 0')
parser.add_argument('--classification_dim', type=int, default=None)
parser.add_argument('--classifier_hidden', type=int, default=256)
parser.add_argument('--classifier_dropout', type=float, default=0.2)
parser.add_argument('--dlinear_features', choices=('linear', 'quadratic', 'moments'), default='linear')
parser.add_argument('--dlinear_hidden', type=int, default=0)
parser.add_argument('--dlinear_moment_scale', type=float, default=1.0)

# PatchTST (kept identical to forecasting defaults)
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=16, help='patch length')
parser.add_argument('--stride', type=int, default=8, help='PatchTST patch stride')
parser.add_argument('--padding_patch', default='end', help='PatchTST end padding')
parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='subtract mean 0; subtract last 1')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition kernel')
parser.add_argument('--norm', type=str, default='LayerNorm', help='PatchTST normalization')
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of feed-forward network')
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=3, help='train epochs')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of train input data')
parser.add_argument('--learning_rate', type=float, default=1e-4, help='optimizer learning rate')
parser.add_argument('--lr_step_size', type=int, default=10, help='learning rate scheduler step size')
parser.add_argument('--lr_gamma', type=float, default=0.9, help='learning rate scheduler gamma')
parser.add_argument('--des', type=str, default='test', help='exp description')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', type=int, default=0, help='use multiple gpus')
parser.add_argument('--devices', type=str, default='0', help='device ids of multile gpus')

# DP parameters
parser.add_argument('--dp_sigma', type=float, default=1.0, help='DP sigma')
parser.add_argument('--dp_delta', type=float, default=1e-5, help='DP delta')
parser.add_argument('--clipping_norm', type=float, default=1.0, help='per-sample gradient clipping norm')
parser.add_argument('--privacy_budget_limit', type=float, default=10.0, help='DP privacy budget limit')
parser.add_argument('--w', type=float, default=0.01, help='private ratio')

parser.add_argument('--train_path', type=str, required=True, help='training data csv path')
parser.add_argument('--test_path', type=str, required=True, help='testing data csv path')
parser.add_argument('--result_file', type=str, default='./results/baseline_results.txt', help='file to save results')
parser.add_argument('--seed', type=int, default=None,
                    help='random seed (default: 42 + iteration index)')

# Label mode
parser.add_argument('--label_mode', type=str, default='sequence', choices=['sequence', 'point'], 
                    help='label mode: sequence (one label per sequence) or point (one label per data point)')

args = parser.parse_args()
if args.sampling_stride is None:
    args.sampling_stride = args.seq_len
if args.sampling_stride <= 0:
    parser.error('--sampling_stride must be positive')

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

seed_base = 42 if args.seed is None else args.seed
fix_seed_list = [seed_base + i for i in range(args.itr)]
test_result = []
if args.is_training:
    for ii in range(args.itr):
        # Set random seed for reproducibility
        args.seed = fix_seed_list[ii]  
        set_seed(fix_seed_list[ii])
        
        # setting record of experiments
        setting = '{}_{}_sl{}_nk{}_lr{}_step{}_gamma{}_dps{}_cn{}_eps{}_itr{}_bz{}'.format(
            args.model_id,
            args.model,
            args.seq_len,
            args.num_kernels,
            args.learning_rate,
            args.lr_step_size,
            args.lr_gamma,
            args.dp_sigma,
            args.clipping_norm,
            args.privacy_budget_limit,
            ii,
            args.batch_size)
        setting += f'_ss{args.sampling_stride}_seed{fix_seed_list[ii]}'

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        test_acc, test_f1 = exp.test(setting)
        test_result.append((test_acc, test_f1))

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_sl{}_nk{}_lr{}_dp{}_{}_eps{}_itr{}'.format(
        args.model_id,
        args.model,
        args.seq_len,
        args.num_kernels,
        args.learning_rate,
        args.dp_sigma,
        args.clipping_norm,
        args.privacy_budget_limit,
        ii)
    setting += f'_ss{args.sampling_stride}_seed{seed_base}'

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting, test=1)
    torch.cuda.empty_cache()

print('>>>>>>>all experiments done<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
if test_result:
    mean_acc = np.mean([r[0] for r in test_result])
    std_acc = np.std([r[0] for r in test_result])
    mean_f1 = np.mean([r[1] for r in test_result])
    std_f1 = np.std([r[1] for r in test_result])
    print(f'\nMean Test Accuracy: {mean_acc:.6f} ± {std_acc:.6f}')
    print(f'Mean Test F1: {mean_f1:.6f} ± {std_f1:.6f}')
    print(f'Individual results: {test_result}')
else:
    mean_acc, std_acc, mean_f1, std_f1 = 0.0, 0.0, 0.0, 0.0
print('Args in experiment:')
if args.result_file:
    result_dir = os.path.dirname(os.path.abspath(args.result_file))
    os.makedirs(result_dir, exist_ok=True)
    with open(args.result_file, 'w') as f:
        eps = args.privacy_budget_limit
        f.write(f"{eps}\t{mean_acc:.6f}\t{mean_f1:.6f}\t{std_acc:.6f}\t{std_f1:.6f}\n")
