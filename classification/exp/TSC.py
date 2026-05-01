import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
from models.rocket import RocketClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

class WISDMDataset(Dataset):
    def __init__(self, root_path, flag='train', seq_len=120):
        self.root_path = root_path
        self.flag = flag
        self.seq_len = seq_len
        self.__read_data__()

    def __read_data__(self):
        if self.flag == 'train':
            file_name = 'train.csv'
        elif self.flag == 'val':
            file_name = 'val.csv'
        else:
            file_name = 'test.csv'
            
        file_path = os.path.join(self.root_path, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        df = pd.read_csv(file_path)
        # Assuming columns are x, y, z, label
        self.data = df.iloc[:, :-1].values # x, y, z
        self.labels = df.iloc[:, -1].values # label

    def __getitem__(self, index):
        s_begin = index * self.seq_len
        s_end = s_begin + self.seq_len
        
        seq_x = self.data[s_begin:s_end]
        # Transpose to [Channel, Length] for Rocket
        seq_x = seq_x.T 
        
        # Label determination: Majority vote
        window_labels = self.labels[s_begin:s_end].astype(np.int64)
        counts = np.bincount(window_labels)
        seq_y = np.argmax(counts)
        
        return torch.FloatTensor(seq_x), torch.LongTensor([seq_y])

    def __len__(self):
        return len(self.data) // self.seq_len

class Exp_TSC:
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.learning_rate)

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = RocketClassifier(
            input_length=self.args.seq_len,
            num_classes=self.args.num_classes,
            num_kernels=self.args.num_kernels,
            in_channels=self.args.enc_in
        )
        return model

    def _get_data(self, flag):
        dataset = WISDMDataset(
            root_path=self.args.root_path,
            flag=flag,
            seq_len=self.args.seq_len
        )
        
        shuffle_flag = (flag == 'train')
        drop_last = True
        batch_size = self.args.batch_size

        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=self.args.num_workers,
            drop_last=drop_last
        )
        return data_loader

    def train(self, setting):
        train_loader = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_loss = []
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device).squeeze()

                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                train_loss.append(loss.item())
                
                loss.backward()
                self.optimizer.step()
                
                vali_loss, vali_acc = self.valid(vali_loader)
                test_loss, test_acc = self.valid(test_loader)

                print(f"Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item():.7f} | Vali Loss: {vali_loss:.7f} Vali Acc: {vali_acc:.7f} Test Loss: {test_loss:.7f} Test Acc: {test_acc:.7f}")

            train_loss = np.average(train_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))

        return self.model

    def valid(self, vali_loader):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device).squeeze()

                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                total_loss.append(loss.item())
                
                pred = outputs.argmax(dim=1).detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        accuracy = accuracy_score(trues, preds)
        
        self.model.train()
        return total_loss, accuracy

    def test(self, setting, test=0):
        test_loader = self._get_data(flag='test')
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device).squeeze()

                outputs = self.model(batch_x)
                pred = outputs.argmax(dim=1).detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        accuracy = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='macro')
        
        print('Test Accuracy: {:.6f}, Test F1: {:.6f}'.format(accuracy, f1))
        print("\nClassification Report:")
        print(classification_report(trues, preds, zero_division=0))
        print("\nConfusion Matrix:")
        print(confusion_matrix(trues, preds))
        
        return accuracy
