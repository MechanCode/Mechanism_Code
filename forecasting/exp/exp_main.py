"""exp_main.py

Main experiment runner implementing DP-SGD (Differentially Private SGD) with Poisson sampling.

DP notes:
- Uses Poisson subsampling strategy for privacy amplification.
- Gradient clipping is applied per-sample before aggregation.
- Gaussian noise is injected with sigma = C (where C = clipping_norm),
  following the standard DP-SGD convention.
- Privacy budget is tracked via RDP (Rényi Differential Privacy) accounting.
"""

import warnings
import logging
import math
import sys
from exp.exp_basic import Exp_Basic
# Import models dynamically to support different model types
import torch.nn as nn
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping
import os
import torch.optim as optim
import numpy as np
import torch
from torch.func import functional_call, vmap, grad
from typing import Dict, List, Tuple, Optional, Any
from scipy.special import comb, logsumexp


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    """Main experiment class for DP-SGD with Poisson sampling.

    This class implements standard DP-SGD training with Poisson subsampling.
    Noise is scaled by C (sigma = clipping_norm), following DP-SGD convention.

    Attributes:
        dp_sigma: Noise multiplier for DP (sourced from args.dp_sigma).
        dp_delta: Delta parameter for (ε,δ)-DP (sourced from args.dp_delta).
        clipping_norm: Per-sample gradient clipping bound C (sourced from args.clipping_norm).
        sampling_rate: Poisson sampling rate for privacy amplification.
        privacy_budget_limit: Maximum epsilon before stopping training.
    """

    def __init__(self, args):
        """Initialize experiment with DP parameters.

        Parameters:
            args: Argument namespace containing model and DP configuration.
                Required DP args: dp_sigma, dp_delta, clipping_norm, privacy_budget_limit.

        Side effects:
            Sets up DP tracking variables (total_steps, should_stop_training).
        """
        super(Exp_Main, self).__init__(args)
        # DP: Differential Privacy parameters (sourced from command-line args)
        self.dp_sigma = getattr(args, 'dp_sigma', 1.0)  # DP: noise multiplier σ
        self.dp_delta = getattr(args, 'dp_delta', 1e-5)  # DP: delta parameter δ
        self.sampling_rate = getattr(args, 'dp_sampling_rate', 0.01)  # DP: Poisson sampling rate
        self.sensitivity = getattr(args, 'sensitivity', 1.0)  # DP: sensitivity parameter
        self.clipping_norm = getattr(args, 'clipping_norm', 0.5)  # DP: gradient clipping bound C
        self.privacy_budget_limit = getattr(args, 'privacy_budget_limit', 10.0)  # DP: max epsilon budget
        self.total_steps = 0  # DP: total training steps for privacy accounting
        self.should_stop_training = False  # DP: flag to stop when budget exhausted
        self.data_size = 0  # to be set when data is loaded
        self.w = 0  # DP: window parameter for privacy analysis

    def _build_model(self):
        from models import DLinear, PatchTST
        
        model_dict = {
            'DLinear': DLinear,
            'PatchTST': PatchTST
        }
        
        model = model_dict[self.args.model].Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        if flag == 'train':
            data_set, data_loader, sampling_rate, data_size = data_provider(self.args, flag)
            self.data_size = data_size
            self.w = int(data_size * self.args.w)
            # self.w = int(self.args.w)
            self.sampling_rate = sampling_rate
            print("Data size:", data_size, "Window w:", self.w, "Sampling rate:", sampling_rate)
            # exit()
            return data_set, data_loader, sampling_rate
        else:
            data_set, data_loader = data_provider(self.args, flag)
            return data_set, data_loader


    def _select_optimizer(self):
        # optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        optimizer = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
        return optimizer


    def _select_criterion(self):
        if self.args.loss == "mae":
            criterion = nn.L1Loss()
        elif self.args.loss == "mse":
            criterion = nn.MSELoss()
        elif self.args.loss == "smooth":
            criterion = nn.SmoothL1Loss()
        elif self.args.loss == "NLL":
            criterion = nn.NLLLoss()
        else:
            criterion = nn.MSELoss()
        return criterion


    def _update_privacy_accountant(self, steps):
        """
        Compute privacy budget ε using optimized RDP accounting with T-privacy loss
        This provides tighter privacy bounds compared to simple approximation
        """
        if self.dp_sigma == 0:
            return float('inf')
        L = self.args.seq_len + self.args.pred_len
        stride = getattr(self.args, 'sampling_stride', L)
        sensitivity_int = math.ceil((self.w + L - 1) / stride)
        # print(sensitivity_int)
        def T_privacy_loss(alpha, steps):
            """Compute T-privacy loss function for RDP accounting"""
            loss = 0
            q = self.sampling_rate
            # for k in range(sensitivity_int + 1):
            #     comb = math.comb(sensitivity_int, k)
            #     term1 = comb * math.pow(1 - self.sampling_rate, sensitivity_int-k)
            #     term2 = math.pow(self.sampling_rate, k)
            #     term3 = np.exp(((alpha - 1) * alpha * k * k) / (2 * self.dp_sigma * self.dp_sigma))
            #     loss += term1 * term2 * term3
            #     if alpha < 5:
            #         print("terms", alpha, term1, term2, term3, loss)
            # step_loss =  steps / (alpha - 1) * math.log(loss)
            m = sensitivity_int
            sigma = self.dp_sigma
            log_terms = []
            for k in range(0, m+1):
                log_term = (np.log(comb(m, k, exact=False)) + 
                        (m-k)*np.log(1-q) + 
                        k*np.log(q) + 
                        (alpha-1)*alpha*k*k/2/sigma/sigma)
                log_terms.append(log_term)
            
            step_loss = steps / (alpha - 1) * logsumexp(log_terms)

            
            para_loss = (math.log(1 / self.dp_delta) + (alpha - 1) * math.log(1 - 1 / alpha) - math.log(alpha)) / (alpha - 1)
            # para_loss = math.log(1 / self.dp_delta)/(alpha-1)
            return step_loss+para_loss, step_loss, para_loss
    
        # Find optimal alpha that minimizes privacy loss
        min_loss = sys.maxsize
        min_step_loss = 0
        min_para_loss = 0
        min_alpha = 0
        
        # Search over range of alpha values to find minimum
        for alpha_value in range(2, 64):  # Start from 2 to avoid division by 
            loss, step_loss, para_loss = T_privacy_loss(alpha_value, steps)
            if loss < min_loss:
                min_loss = loss
                min_step_loss = step_loss
                min_para_loss = para_loss
                min_alpha = alpha_value
        # Convert RDP to (ε, δ)-DP using optimal conversion
        eps = min_loss
        # print("alpha value", min_alpha, "Min ε:", eps, "Step loss:", min_step_loss, "Para loss:", min_para_loss)
        # exit()
        
        # Check if privacy budget limit is exceeded
        if eps >= self.privacy_budget_limit:
            print(f"Privacy budget limit reached! Current ε={eps:.4f}, limit={self.privacy_budget_limit}")
            self.should_stop_training = True
        return eps

    def _compute_dp_gradients(self, batch_x, batch_y, criterion, batch_x_mark=None, batch_y_mark=None):
        """Compute differentially private gradients with per-sample clipping and noise injection.

        Implements DP-SGD for Poisson sampling using vectorized per-sample gradients (vmap):
        1. Compute per-sample gradients using torch.func.vmap for efficiency
           (optionally in micro-batches to reduce GPU memory)
        2. Clip each sample's gradient by clipping_norm (C)
        3. Aggregate clipped gradients
        4. Add Gaussian noise with sigma = 2 * C (standard DP-SGD convention)

        Parameters:
            batch_x: Input batch tensor [batch_size, seq_len, features].
            batch_y: Target batch tensor [batch_size, pred_len, features].
            criterion: Loss function.
            batch_x_mark: Optional time features for input.
            batch_y_mark: Optional time features for target.

        Returns:
            Tuple of (noisy_gradients, batch_size) where noisy_gradients is a list
            of privatized gradient tensors for each model parameter.

        Side effects:
            Modifies model gradients during computation.
        """
        batch_size = batch_x.size(0)
        micro_bs = getattr(self.args, 'micro_batch_size', 0)
        if micro_bs <= 0 or micro_bs >= batch_size:
            micro_bs = batch_size

        param_names = [k for k, _ in self.model.named_parameters()]
        num_params = len(param_names)

        # Accumulator for clipped gradient sums across micro-batches
        aggregated_grads = None

        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            mb_x = batch_x[start:end]
            mb_y = batch_y[start:end]
            mb_size = end - start

            # Compute per-sample gradients for this micro-batch
            per_sample_grads = self._compute_per_sample_grads_vmap(mb_x, mb_y)

            # Compute per-sample gradient norms for clipping
            grad_norms = torch.zeros(mb_size, device=batch_x.device)
            for param_grads in per_sample_grads:
                grad_norms += param_grads.view(mb_size, -1).pow(2).sum(dim=1)
            grad_norms = grad_norms.sqrt()

            # DP: Compute clipping factors for each sample
            clip_factors = torch.clamp(self.clipping_norm / (grad_norms + 1e-8), max=1.0)

            # DP: Clip and aggregate gradients for this micro-batch
            for idx, param_grads in enumerate(per_sample_grads):
                clip_shape = [mb_size] + [1] * (param_grads.dim() - 1)
                clipped_grads = param_grads * clip_factors.view(*clip_shape)
                summed_grad = clipped_grads.sum(dim=0)

                del clipped_grads
                per_sample_grads[idx] = None

                if aggregated_grads is None:
                    aggregated_grads = [None] * num_params
                if aggregated_grads[idx] is None:
                    aggregated_grads[idx] = summed_grad
                else:
                    aggregated_grads[idx] = aggregated_grads[idx] + summed_grad
                del summed_grad

            del per_sample_grads, grad_norms, clip_factors
            if batch_x.is_cuda:
                torch.cuda.empty_cache()

        # DP: Add Gaussian noise once after all micro-batches are aggregated
        noisy_grad = []
        for idx in range(num_params):
            noise = torch.randn_like(aggregated_grads[idx]) * self.dp_sigma * self.clipping_norm * 2
            noisy_param_grad = aggregated_grads[idx] + noise
            del noise
            aggregated_grads[idx] = None
            noisy_grad.append(noisy_param_grad)

        del aggregated_grads
        if batch_x.is_cuda:
            torch.cuda.empty_cache()

        return noisy_grad, batch_size
    
    def _compute_per_sample_grads_vmap(self, batch_x, batch_y):
        """Compute per-sample gradients using vmap with independent randomness per sample."""
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = dict(self.model.named_buffers())
        
        def compute_loss(params, buffers, x, y):
            x_input = x.unsqueeze(0)
            y_target = y.unsqueeze(0)
            
            output = functional_call(self.model, (params, buffers), (x_input,))
            
            f_dim = -1 if self.args.features == 'MS' else 0
            output = output[:, -self.args.pred_len:, f_dim:]
            target = y_target[:, -self.args.pred_len:, f_dim:]
            
            return torch.nn.functional.mse_loss(output, target, reduction='mean')
        
        ft_compute_grad = grad(compute_loss)
        # randomness='different': each sample in the batch gets independent random numbers
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='different')
        
        per_sample_grads_dict = ft_compute_sample_grad(params, buffers, batch_x, batch_y)
        
        # Convert dict to list in parameter order
        return [per_sample_grads_dict[name] for name in params.keys()]

    def _evaluate_loader(self, loader, criterion):
        """Shared evaluation for validation and test to ensure identical logic (MSE only)."""
        self.model.eval()
        total_loss_sum = 0.0
        total_num = 0

        # Use provided criterion or create new one
        if criterion is None:
            criterion = self._select_criterion()

        with torch.no_grad():
            for batch in loader:
                # Expecting 4 elements from dataset
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                # Move to device
                if self.args.use_gpu:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    if batch_x_mark is not None:
                        batch_x_mark = batch_x_mark.to(self.device)
                    if batch_y_mark is not None:
                        batch_y_mark = batch_y_mark.to(self.device)

                # Decoder input (for seq2seq-style models)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(batch_x.device)

                # Forward
                if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                    pred = self.model(batch_x)
                else:
                    if self.args.output_attention:
                        pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                # Slice prediction window and features
                f_dim = -1 if self.args.features == 'MS' else 0
                pred = pred[:, -self.args.pred_len:, f_dim:]
                target = batch_y[:, -self.args.pred_len:, f_dim:]

                # Compute loss on GPU to avoid CPU-GPU sync overhead
                loss = criterion(pred, target)

                bs = batch_x.size(0)
                total_loss_sum += loss.item() * bs
                total_num += bs

        self.model.train()
        return (total_loss_sum / total_num) if total_num > 0 else float('inf')

    def vali(self, vali_loader, criterion):
        """Validation method for multivariate time series"""
        return self._evaluate_loader(vali_loader, criterion)

    def test(self, setting=None, test=0, criterion=None):
        """Test method for multivariate time series"""
        test_data, test_loader = self._get_data(flag='test')

        # If setting is provided, try to load the best model from that setting
        if setting is not None:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            if os.path.exists(best_model_path):
                self.model.load_state_dict(torch.load(best_model_path))
                print(f"Loaded model from {best_model_path}")

        avg_loss = self._evaluate_loader(test_loader, criterion)
        print(f"Test Loss: {avg_loss:.4f}")
        return avg_loss

    def train(self, setting):
        max_steps = getattr(self.args, 'max_train_steps', None)
        if max_steps is not None and max_steps <= 0:
            raise ValueError('max_train_steps must be positive')
        _, train_loader, sampling_rate = self._get_data(flag='train')
        self.sampling_rate = sampling_rate
        _, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)
        
        optimizer = self._select_optimizer()
        criterion = self._select_criterion()
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=getattr(self.args, 'patience', 7),
            verbose=True,
            delta=getattr(self.args, 'delta', 0.0)
        )
        
        # Log and set validation interval (default 10; configurable via args.val_interval_batches)
        print(f"Starting training with Early stopping patience: {early_stopping.patience}")
        val_interval_batches = getattr(self.args, 'val_interval_batches', 10)
        
        for epoch in range(self.args.train_epochs):
            if self.should_stop_training:
                print(f"Training stopped early at epoch {epoch} due to privacy budget limit")
                break
                
            self.model.train()
            # Save model state at start of epoch so we can restore it if privacy budget is exceeded mid-epoch
            prev_epoch_ckpt = os.path.join(path, 'model_before_epoch.pth')
            try:
                torch.save(self.model.state_dict(), prev_epoch_ckpt)
            except Exception:
                # If saving fails for some reason (e.g., distributed wrappers), ignore and continue
                prev_epoch_ckpt = None
            epoch_loss_sum = 0.0
            epoch_num = 0
            print(f"Epoch [{epoch+1}/{self.args.train_epochs}] started")
            
            for i, batch in enumerate(train_loader):
                if max_steps is not None and self.total_steps >= max_steps:
                    break
                if self.should_stop_training:
                    print(f"Training stopped at epoch {epoch+1}, batch {i+1} due to privacy budget limit")
                    break

                # Pre-check: if the next step would exceed privacy budget, stop before optimizer step
                eps_next = self._update_privacy_accountant(self.total_steps + 1) if max_steps is None else None
                if max_steps is None and eps_next >= self.privacy_budget_limit:
                    print(f"Upcoming step would exceed privacy budget (ε={eps_next:.4f} >= {self.privacy_budget_limit}). Stopping before optimizer step.")
                    # One last validation try before stopping to capture a potential best
                    val_loss_batch = self.vali(vali_loader, criterion)
                    if val_loss_batch < early_stopping.val_loss_min - early_stopping.delta:
                        torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                        early_stopping.val_loss_min = val_loss_batch
                        early_stopping.best_score = -val_loss_batch
                        early_stopping.counter = 0
                        print(f"  [Budget Stop Save] Epoch {epoch+1}, Batch {i+1}: val improved to {val_loss_batch:.4f}. Saved checkpoint.pth")
                    self.should_stop_training = True
                    break

                batch_x, batch_y, batch_x_mark, batch_y_mark = batch

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # Compute differentially private gradients using vectorized vmap
                dp_grads, _ = self._compute_dp_gradients(
                    batch_x, batch_y, criterion, batch_x_mark, batch_y_mark
                )

                if dp_grads is not None:
                    optimizer.zero_grad()
                    # Apply DP gradients - dp_grads is aligned with named_parameters order
                    for (name, param), dp_grad in zip(self.model.named_parameters(), dp_grads):
                        if param.requires_grad and dp_grad is not None:
                            param.grad = dp_grad / self.args.batch_size
                    optimizer.step()
                    
                    # Compute loss for logging (without affecting gradients)
                    with torch.no_grad():
                        # SparseTSF style forward pass
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(batch_x.device)
                        
                        if any(substr in self.args.model for substr in {'Linear', 'TST', 'SparseTSF'}):
                            pred = self.model(batch_x)
                        else:
                            if self.args.output_attention:
                                pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                            else:
                                pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        
                        # Extract prediction part
                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred = pred[:, -self.args.pred_len:, f_dim:]
                        target = batch_y[:, -self.args.pred_len:, f_dim:]
                        
                        loss = criterion(pred, target)
                        bs = batch_x.size(0)
                        epoch_loss_sum += loss.item() * bs
                        epoch_num += bs

                    # Batch-level validation every val_interval_batches
                    if val_interval_batches and (i + 1) % val_interval_batches == 0:
                        # Clear GPU memory before validation to prevent OOM
                        if batch_x.is_cuda:
                            torch.cuda.empty_cache()
                        val_loss_batch = self.vali(vali_loader, criterion)
                        if val_loss_batch < early_stopping.val_loss_min - early_stopping.delta:
                            torch.save(self.model.state_dict(), path + '/' + 'checkpoint.pth')
                            early_stopping.val_loss_min = val_loss_batch
                            early_stopping.best_score = -val_loss_batch
                            early_stopping.counter = 0
                            print(f"  [Batch Save] Epoch {epoch+1}, Batch {i+1}: val improved to {val_loss_batch:.4f}. Saved checkpoint.pth")
                
                # Update total steps counter
                self.total_steps += 1
                
                # Check privacy budget limit
                eps = self._update_privacy_accountant(self.total_steps) if max_steps is None else None
                # Privacy budget check is done inside _update_privacy_accountant
                
                # Show batch progress every 10 batches
                if (i + 1) % 10 == 0:
                    print(f"  Batch [{i+1}/{len(train_loader)}] processed")
                    print("Current privacy budget: ", eps, "Total steps:", self.total_steps)
                    
            avg_epoch_loss = (epoch_loss_sum / epoch_num) if epoch_num > 0 else 0
            print(f"Epoch [{epoch+1}/{self.args.train_epochs}] completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Clear GPU memory before validation/test to prevent OOM
            if self.args.use_gpu:
                torch.cuda.empty_cache()
            
            # Validation and early stopping using EarlyStopping class
            val_loss = self.vali(vali_loader, criterion)
            
            # # Test evaluation using the SAME criterion as validation
            # test_loss = self.test(criterion=criterion)
            # print(f"Epoch [{epoch+1}/{self.args.train_epochs}] - Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            early_stopping(val_loss, self.model, path)
            
            # # Print test loss after model is saved
            # if early_stopping.counter == 0:  # Model was saved
            #     print(f"  Current Test Loss: {test_loss:.4f}")
            
            if max_steps is not None and self.total_steps >= max_steps:
                print(f'Fixed training steps completed: {self.total_steps}')
                break
            if max_steps is None and early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
        if max_steps is not None and self.total_steps != max_steps:
            raise RuntimeError(f'Expected {max_steps} updates, completed {self.total_steps}; increase train_epochs')
        # Load the best model from early stopping
        best_model_path = path + '/' + 'checkpoint.pth'
        # If budget stop occurred and no best checkpoint exists yet, save the current last-safe model.
        if self.should_stop_training and (not os.path.exists(best_model_path)):
            torch.save(self.model.state_dict(), best_model_path)
            print("Privacy budget stop: saved current last-safe model to checkpoint.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))
            print(f"Loaded best model with validation loss: {early_stopping.val_loss_min:.4f}")
        
        # Final test evaluation with the best model
        final_test_results = self.test()
        print(f"Final Test Loss: {final_test_results:.4f}")
        return self.model
