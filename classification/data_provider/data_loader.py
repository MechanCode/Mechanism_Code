import os 
import numpy as np 
import pandas as pd 
import os 
import torch 
from torch.utils.data import Dataset, DataLoader 
from sklearn.preprocessing import StandardScaler
import warnings
from collections import Counter

warnings.filterwarnings('ignore')


def get_stratum_id(date, dataset_name):
    stratum_id = date.weekday()
    return stratum_id

    
class Dataset_Custom(Dataset):
    def __init__(self, train_path, test_path, flag='train', size=None, stratification=None,
                 val_ratio=0.2, label_mode='sequence', val_path=None,
                 window_stride=None):
        """
        Args:
            train_path: Path to the training data CSV file
            test_path: Path to the test data CSV file
            flag: 'train', 'val', or 'test'
            size: Sequence length (seq_len)
            label_mode: 'sequence' for one label per sequence (default),
                        'point' for one label per data point
            val_path: Path to the validation data CSV file (if None, it will be inferred from train_path)
        """
        self.seq_len = size
        self.train_path = train_path
        self.test_path = test_path
        self.flag = flag
        self.val_ratio = val_ratio
        self.label_mode = label_mode  # 'sequence' or 'point'
        self.window_stride = self.seq_len if window_stride is None else window_stride
        if self.seq_len <= 0:
            raise ValueError(f'seq_len must be positive, got {self.seq_len}')
        if self.window_stride <= 0:
            raise ValueError(f'window_stride must be positive, got {self.window_stride}')
        
        if val_path is None:
            train_dir = os.path.dirname(train_path)
            self.val_path = os.path.join(train_dir, 'validation_data.csv')
        else:
            self.val_path = val_path
        
        self.__read_data__() 
    
    def __read_data__(self):
        if self.flag == 'test':
            # Test uses separate file
            path = self.test_path
            df_raw = pd.read_csv(path)
            
            # Features and labels
            self.data = df_raw.iloc[:, :-1].values
            labels_raw = df_raw.iloc[:, -1].values.astype(np.int64)
            if labels_raw.min() >= 1:
                self.label = labels_raw - 1  # Convert 1-based to 0-based
            else:
                self.label = labels_raw  # Already 0-based
            print(f"Test data loaded: {len(self.data)} samples, labels range: {self.label.min()}-{self.label.max()}")
                
        elif self.flag == 'val':
            # Validation uses separate file
            path = self.val_path
            if os.path.exists(path):
                df_raw = pd.read_csv(path)
                self.data = df_raw.iloc[:, :-1].values
                labels_raw = df_raw.iloc[:, -1].values.astype(np.int64)
                if labels_raw.min() >= 1:
                    self.label = labels_raw - 1  # Convert 1-based to 0-based
                else:
                    self.label = labels_raw  # Already 0-based
                print(f"Validation data loaded from {path}: {len(self.data)} samples, labels range: {self.label.min()}-{self.label.max()}")
            else:
                raise FileNotFoundError(f"Validation file not found: {path}")
                
        else:  # train
            path = self.train_path
            df_raw = pd.read_csv(path)
            
            # Features and labels
            self.data = df_raw.iloc[:, :-1].values
            labels_raw = df_raw.iloc[:, -1].values.astype(np.int64)
            if labels_raw.min() >= 1:
                self.label = labels_raw - 1  # Convert 1-based to 0-based
            else:
                self.label = labels_raw  # Already 0-based
            print(f"Training data loaded: {len(self.data)} samples, labels range: {self.label.min()}-{self.label.max()}")
        
        self.data_shape = (len(self.data), self.data.shape[1] if len(self.data) > 0 else 0)
        self.raw_data_size = len(self.data)

    def __getitem__(self, index):
        return self._get_single_item(index)

    def _get_single_item(self, index):
        # Map a window id to its raw row offset.  Training uses unit-stride
        # window ids and lets the batch sampler apply ``sampling_stride``;
        # validation/test retain the historical non-overlapping default.
        s_begin = int(index) * self.window_stride
        s_end = s_begin + self.seq_len
    
        seq_data = self.data[s_begin:s_end].astype(np.float32)
        seq_label_array = self.label[s_begin:s_end]
        
        if self.label_mode == 'point':
            # Return all labels for the sequence [seq_len]
            seq_label = torch.from_numpy(seq_label_array.astype(np.int64))
        else:
            # Return single label for the entire sequence (mode of labels in the window)
            from scipy import stats
            seq_label = torch.tensor(stats.mode(seq_label_array, keepdims=False)[0], dtype=torch.int64)
        
        seq_data = torch.from_numpy(np.ascontiguousarray(seq_data))
        return seq_data, seq_label

    def __len__(self):
        if len(self.data) < self.seq_len:
            return 0
        return (len(self.data) - self.seq_len) // self.window_stride + 1
