# from torch.utils.data import Dataloader
from data_provider.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
from data_provider.data_sampler import SpacedSamplingWithFixedSize
import torch
import numpy as np
import random


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _batch_collate(batch):
    """Collate function to stack individual samples into a batch.
    batch: list of (data, label) tuples
    Returns: (stacked_data, stacked_labels)
    """
    data_list = [item[0] for item in batch]
    label_list = [item[1] for item in batch]
    
    # Stack data: [batch_size, seq_len, features]
    stacked_data = torch.stack(data_list, dim=0)
    
    # Stack labels: [batch_size] or [batch_size, seq_len]
    # Handle scalar labels (0-dim tensors)
    if label_list[0].dim() == 0:
        stacked_labels = torch.stack([l.unsqueeze(0) for l in label_list], dim=0).squeeze(1)
    else:
        stacked_labels = torch.stack(label_list, dim=0)
    
    return stacked_data, stacked_labels


def data_provider(args, flag):
    Data = Dataset_Custom
    
    # Get label_mode from args, default to 'sequence'
    label_mode = getattr(args, 'label_mode', 'sequence')
    
    if flag == 'test':
        shuffle_flag = False
        drop_last = False  # fix bug
        batch_size = args.batch_size
    elif flag == 'train':
        shuffle_flag = True
        drop_last = False
    elif flag == 'val':
        shuffle_flag = False  
        drop_last = False    
        batch_size = args.batch_size


    data_set = Data(
        train_path=args.train_path,
        test_path=args.test_path,
        flag=flag,
        size=args.seq_len,
        label_mode=label_mode,
    )
    if flag == 'train':
        size=args.seq_len
        batch_sampler = SpacedSamplingWithFixedSize(data_set, args.batch_size, args.lam, size)
        g = torch.Generator()
        seed = getattr(args, 'seed', 42)
        g.manual_seed(seed)
        # print(flag, len(data_set), len(data_set[0]), len(data_set[0][0]))
        data_loader = DataLoader(
            data_set,
            batch_sampler=batch_sampler,
            num_workers=args.num_workers,
            collate_fn=_batch_collate,
            worker_init_fn=seed_worker,
            generator=g)
        return data_set, data_loader, len(data_set)
    else:
        g = torch.Generator()
        seed = getattr(args, 'seed', 42)
        g.manual_seed(seed)
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=g)
        return data_set, data_loader


