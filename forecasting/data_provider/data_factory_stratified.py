"""data_factory_stratified.py

Data provider factory for stratified spaced sampling with differential privacy.

Creates DataLoader instances with SpacedSamplingWithFixedSize for
differentially private time series forecasting experiments using stratified sampling.

DP notes:
- Uses stratified spaced sampling for privacy amplification during training.
- Actual noise injection (sigma = 2*C) occurs in exp/exp_main_spaced_sampling.py
  and exp/exp_main_poisson_spaced_sampling.py.
- This module only handles data loading and sampling, not DP noise.

Reproducibility:
- Accepts an optional `seed` parameter via args to ensure consistent
  sampling across different experiments with the same seed.
"""

# from torch.utils.data import Dataloader
from data_provider.data_loader import Dataset_Custom, Dataset_Pred, custom_collate_fn
from torch.utils.data import DataLoader
from data_provider.data_sampler import SpacedSamplingWithFixedSize
from utils.seed_utils import get_worker_init_fn, get_dataloader_generator

data_dict = {
    # 'ETTh1': Dataset_ETT_hour,
    # 'ETTh2': Dataset_ETT_hour, 
    # 'ETTm1': Dataset_ETT_minute, 
    # 'ETTm2': Dataset_ETT_minute, 
    'custom': Dataset_Custom
    # 'Solar': Dataset_Solar
}

def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    elif flag == 'train':
        shuffle_flag = True
        drop_last = False
        # batch_size = args.batch_size
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False  
        drop_last = False    
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq


    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq
    )
    
    # Get seed from args for reproducible sampling (None if not set)
    seed = getattr(args, 'seed', None)
    
    if flag == 'train':
        size=[args.seq_len, args.label_len, args.pred_len]
        batch_sampler = SpacedSamplingWithFixedSize(data_set, args.batch_size, args.lam, size, seed=seed)
        data_loader = DataLoader(
            data_set,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            worker_init_fn=get_worker_init_fn(seed) if seed is not None else None,
            generator=get_dataloader_generator(seed) if seed is not None else None)
        return data_set, data_loader, len(data_set)
    else:
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            worker_init_fn=get_worker_init_fn(seed) if seed is not None else None,
            generator=get_dataloader_generator(seed) if seed is not None else None)
        return data_set, data_loader