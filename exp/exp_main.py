from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, DLinear, Linear, PatchTST, SparseTSF
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score, f1_score, classification_report

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'DLinear': DLinear,
            'Linear': Linear,
            'PatchTST': PatchTST,
            'SparseTSF': SparseTSF
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        all_preds = []
        all_labels = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.long().to(self.device, non_blocking=True)

                # Forward pass
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        total_loss = np.average(total_loss)
        acc = accuracy_score(all_labels, all_preds)
        self.model.train()
        return total_loss, acc

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        if self.args.lradj == 'TST':
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=self.args.pct_start,
                                                epochs=self.args.train_epochs,
                                                max_lr=self.args.learning_rate)
        else:
            scheduler = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            correct = 0
            total = 0
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad(set_to_none=True)
                batch_x = batch_x.float().to(self.device, non_blocking=True)
                batch_y = batch_y.long().to(self.device, non_blocking=True)

                # Forward pass
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                # Track accuracy
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f} | acc: {3:.4f}".format(
                        i + 1, epoch + 1, loss.item(), correct / total))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'TST':
                    adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                    scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            train_acc = correct / total
            vali_loss, vali_acc = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_acc = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Train Acc: {3:.4f} | "
                  "Vali Loss: {4:.7f} Vali Acc: {5:.4f} | Test Loss: {6:.7f} Test Acc: {7:.4f}".format(
                epoch + 1, train_steps, train_loss, train_acc, vali_loss, vali_acc, test_loss, test_acc))

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj != 'TST':
                adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
            else:
                print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path, map_location="cuda:0"))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location="cuda:0"))

        all_preds = []
        all_labels = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.long().to(self.device)

                # Forward pass
                outputs = self.model(batch_x)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        acc = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        f1_weighted = f1_score(all_labels, all_preds, average='weighted')

        print('Test Accuracy: {:.4f}, F1 Macro: {:.4f}, F1 Weighted: {:.4f}'.format(acc, f1_macro, f1_weighted))
        print('\nClassification Report:')
        print(classification_report(all_labels, all_preds))

        from pathlib import Path
        eps_value = Path(self.args.root_path).name
        train_value = Path(self.args.data_path).name
        result_dir = './classification_results/'
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        file_name = result_dir + self.args.model + '_' + eps_value + '_' + train_value + '.txt'
        with open(file_name, 'a') as f:
            f.write('Accuracy: {:.4f}, F1_macro: {:.4f}, F1_weighted: {:.4f}\n'.format(acc, f1_macro, f1_weighted))

        # Save predictions
        np.save(folder_path + 'pred.npy', all_preds)
        np.save(folder_path + 'true.npy', all_labels)

        return {'accuracy': acc, 'f1_macro': f1_macro, 'f1_weighted': f1_weighted}

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                outputs = self.model(batch_x)
                pred = torch.argmax(outputs, dim=1)
                preds.append(pred.cpu().numpy())

        preds = np.concatenate(preds, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return
