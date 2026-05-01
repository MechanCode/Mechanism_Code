"""data_loader.py

Dataset classes for time series forecasting with differential privacy support.

Provides PyTorch Dataset implementations for loading and preprocessing
time series data for DP-SGD training.

DP notes:
- This module handles data loading only, not DP noise injection.
- Supports batch index lists from DP samplers (Poisson, spaced sampling).
- Data normalization: scaler is fit on training data, then applied to all splits.
"""

import os 
import numpy as np 
import pandas as pd 
import os 
import torch 
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import StandardScaler 
from utils.timefeatures import time_features 
import warnings
from collections import Counter

warnings.filterwarnings('ignore')

    
class Dataset_Custom(Dataset):
    """Custom dataset for multivariate time series forecasting.

    Supports both single-index and batch-index access patterns for
    compatibility with DP samplers (PoissonSampler, SpacedSampler).

    Attributes:
        seq_len: Input sequence length.
        label_len: Label sequence length (for decoder).
        pred_len: Prediction sequence length.
        scaler: StandardScaler fitted on training data.
    """

    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        """Initialize dataset.

        Parameters:
            root_path: Root directory containing data files.
            flag: Dataset split ('train', 'val', 'test').
            size: Tuple of (seq_len, label_len, pred_len).
            features: Feature mode ('S', 'M', 'MS').
            data_path: CSV filename.
            target: Target column name for 'S' and 'MS' modes.
            scale: Whether to apply standardization.
            timeenc: Time encoding type (0: manual, 1: learned).
            freq: Time frequency for time features.
        """
        
        if size == None: 
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        self.features = features 
        self.target = target 
        self.scale = scale 
        self.timeenc = timeenc 
        self.freq = freq 
        
        self.root_path = root_path
        self.data_path = data_path 
        
        self.data_shape = (0, 0)
        
        self.__read_data__() 
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        self.data_shape = df_raw.shape

        # Reorder columns: date first, then features
        cols = list(df_raw.columns)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols]

        # Split sizes (70% train, 10% val, 20% test)
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # Borders for each split
        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Feature columns (exclude date)
        cols_data = df_raw.columns[1:]
        df_data = df_raw[cols_data]

        # Z-score normalization using the entire dataset
        if self.scale:
            self.scaler.fit(df_data.values[:num_train])  # fit on training data
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # Debug prints
        dataset_names = ['train', 'val', 'test']
        print(
            f"Dataset {dataset_names[self.set_type]}: total_data={len(df_raw)}, "
            f"border=[{border1}, {border2}], size={border2 - border1}"
        )

        # Timestamps for current split
        df_stamp = df_raw[['date']][border1:border2].copy()
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])

        # Time features
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute)
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        else:
            # Fallback
            data_stamp = None

        # Slice data for current split
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        if isinstance(index, (list, tuple)):
            # print("batch index:", index)
            batch_data = []
            for idx in index:
                batch_data.append(self._get_single_item(idx))
            
            if batch_data:
                seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list, seq_x_id_list = zip(*batch_data)
                return list(seq_x_list), list(seq_y_list), list(seq_x_mark_list), list(seq_y_mark_list), list(seq_x_id_list)
            else:
                return [], [], [], [], []
        else:
            # print("single index:", index)
            return self._get_single_item(index)

    def _get_single_item(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end 
        r_end = r_begin + self.pred_len

        seq_x = self.data_x[s_begin:s_end].astype(np.float32)
        seq_y = self.data_y[r_begin:r_end].astype(np.float32)
        seq_x_mark = self.data_stamp[s_begin:s_end].astype(np.float32)
        seq_y_mark = self.data_stamp[r_begin:r_end].astype(np.float32)
        

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    
class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            
            data_stamp = df_stamp.drop(['date'], axis=1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
           

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_id

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def custom_collate_fn(batch):
    
    if not batch:
        return [], [], [], [], []
    
    if isinstance(batch[0], list):
        seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list = batch[0]
        
        seq_x_tensor = torch.stack([torch.tensor(x, dtype=torch.float32) for x in seq_x_list])
        seq_y_tensor = torch.stack([torch.tensor(y, dtype=torch.float32) for y in seq_y_list])
        seq_x_mark_tensor = torch.stack([torch.tensor(x_mark, dtype=torch.float32) for x_mark in seq_x_mark_list])
        seq_y_mark_tensor = torch.stack([torch.tensor(y_mark, dtype=torch.float32) for y_mark in seq_y_mark_list])
        
        return seq_x_tensor, seq_y_tensor, seq_x_mark_tensor, seq_y_mark_tensor
    else:
        seq_x_list, seq_y_list, seq_x_mark_list, seq_y_mark_list = zip(*batch)
        
        seq_x_tensor = torch.stack([torch.tensor(x, dtype=torch.float32) for x in seq_x_list])
        seq_y_tensor = torch.stack([torch.tensor(y, dtype=torch.float32) for y in seq_y_list])
        seq_x_mark_tensor = torch.stack([torch.tensor(x_mark, dtype=torch.float32) for x_mark in seq_x_mark_list])
        seq_y_mark_tensor = torch.stack([torch.tensor(y_mark, dtype=torch.float32) for y_mark in seq_y_mark_list])
        
        return seq_x_tensor, seq_y_tensor, seq_x_mark_tensor, seq_y_mark_tensor