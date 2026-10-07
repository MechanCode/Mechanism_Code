"""exp_main_spaced_sampling.py

Experiment runner implementing stratified spaced sampling with differential privacy.

DP notes:
- Uses stratified spaced sampling strategy for privacy amplification.
- Gradient clipping is applied per-sample before aggregation.
- Gaussian noise is injected with sigma = 2 * C (where C = clipping_norm),
  following the spaced sampling DP convention.
- Privacy budget is tracked via RDP (Rényi Differential Privacy) accounting.
"""

import warnings
import logging
import math
import sys
from exp.exp_basic import Exp_Basic
# Import models dynamically to support different model types
import torch.nn as nn
from data_provider.data_factory_stratified import data_provider
from utils.tools import EarlyStopping
import os
import torch.optim as optim
import numpy as np
import torch
from torch.func import functional_call, vmap, grad
from typing import Dict, List, Tuple, Optional, Any


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    """Main experiment class for stratified spaced sampling with differential privacy.

    This class implements DP-SGD training with stratified spaced sampling strategy.
    For spaced sampling, noise is scaled by 2*C (sigma = 2 * clipping_norm).

    Attributes:
        dp_sigma: Noise multiplier for DP (sourced from args.dp_sigma).
        dp_delta: Delta parameter for (ε,δ)-DP (sourced from args.dp_delta).
        clipping_norm: Per-sample gradient clipping bound C (sourced from args.clipping_norm).
        privacy_budget_limit: Maximum epsilon before stopping training.
        lam: Lambda parameter for sampling rate control.
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
        self.args = args
        # DP: Ensure lambda parameter exists; default to 1 if not provided
        if not hasattr(self.args, 'lam'):
            setattr(self.args, 'lam', 1.0)
        self.dp_sigma = getattr(args, 'dp_sigma', 1.0)  # DP: noise multiplier σ
        self.dp_delta = getattr(args, 'dp_delta', 1e-5)  # DP: delta parameter δ
        self.sensitivity = getattr(args, 'sensitivity', 5)  # DP: sensitivity parameter
        self.clipping_norm = getattr(args, 'clipping_norm', 0.5)  # DP: gradient clipping bound C
        self.privacy_budget_limit = getattr(args, 'privacy_budget_limit', 10.0)  # DP: max epsilon budget
        self.total_steps = 0  # DP: total training steps for privacy accounting
        self.should_stop_training = False  # DP: flag to stop when budget exhausted
        self.data_size = 0  # to be set when data is loaded
        self.batch_size = args.batch_size  # batch size for training
        self.w = 0  # DP: window parameter for privacy analysis
        self._privacy_cache = {}  # DP: cache for privacy accountant results (step -> eps)

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
            data_set, data_loader, data_size = data_provider(self.args, flag)
            self.data_size = data_size
            self.w = int(data_size * self.args.w)
            # self.w = int(self.args.w)
            return data_set, data_loader
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


    def _compute_dp_gradients(self, batch_x, batch_y, criterion, batch_x_mark=None, batch_y_mark=None):
        """Compute differentially private gradients with per-sample clipping and noise injection.

        Implements DP-SGD for stratified spaced sampling using vectorized per-sample gradients (vmap):
        1. Compute per-sample gradients using torch.func.vmap for efficiency
           (optionally in micro-batches to reduce GPU memory)
        2. Clip each sample's gradient by clipping_norm (C)
        3. Aggregate clipped gradients
        4. Add Gaussian noise with sigma = 2*C (spaced sampling convention)

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

        # Get model parameters and buffers for functional_call
        params = {k: v.detach() for k, v in self.model.named_parameters()}
        buffers = dict(self.model.named_buffers())
        param_names = list(params.keys())
        num_params = len(param_names)

        # Define the loss function for a single sample
        def compute_loss(params, buffers, x, y):
            """Compute loss for a single sample using functional_call."""
            x_input = x.unsqueeze(0)
            y_target = y.unsqueeze(0)
            output = functional_call(self.model, (params, buffers), (x_input,))
            f_dim = -1 if self.args.features == 'MS' else 0
            output = output[:, -self.args.pred_len:, f_dim:]
            target = y_target[:, -self.args.pred_len:, f_dim:]
            loss = torch.nn.functional.mse_loss(output, target, reduction='mean')
            return loss

        ft_compute_grad = grad(compute_loss)
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='different')

        # Accumulator for clipped gradient sums across micro-batches
        aggregated_grads = None

        for start in range(0, batch_size, micro_bs):
            end = min(start + micro_bs, batch_size)
            mb_x = batch_x[start:end]
            mb_y = batch_y[start:end]
            mb_size = end - start

            # Compute per-sample gradients for this micro-batch
            per_sample_grads = ft_compute_sample_grad(params, buffers, mb_x, mb_y)

            # Compute per-sample gradient norms for clipping
            grad_norms = torch.zeros(mb_size, device=batch_x.device)
            for param_name in per_sample_grads:
                param_grads = per_sample_grads[param_name]
                grad_norms += param_grads.view(mb_size, -1).pow(2).sum(dim=1)
            grad_norms = grad_norms.sqrt()

            # DP: Compute clipping factors for each sample
            clip_factors = torch.clamp(self.clipping_norm / (grad_norms + 1e-8), max=1.0)

            # DP: Clip and aggregate gradients for this micro-batch
            for idx, param_name in enumerate(param_names):
                param_grads = per_sample_grads[param_name]
                clip_shape = [mb_size] + [1] * (param_grads.dim() - 1)
                clipped_grads = param_grads * clip_factors.view(*clip_shape)
                summed_grad = clipped_grads.sum(dim=0)

                del clipped_grads
                per_sample_grads[param_name] = None

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


    @staticmethod
    def rho_stable_b0_from_c(rho, c: np.ndarray, valid=1) -> np.ndarray:
        """Stable computation of the quadratic root b0.
        In the paper, h(b) = A b^2 + B b + C with
            A = 1 - 3 e^c + 3 e^{4c} - e^{9c}
            B = (2rho-1) + (2-4rho) e^c +(2rho-1) e^{4c}
            C = (rho - 1/2)^2 - (2rho^2+1/2) e^c + (rho+1/2)^2 e^{4c}
        where c = alpha(alpha-1)/(2*sigma^2).

        Directly forming e^{kc} overflows quickly. We scale coefficients by e^{-9c}
        (roots unchanged) so only exp(-kc) is used.

        Returns 0 for invalid/degenerate roots.
        """
        if valid == 0:
            return np.zeros_like(c, dtype=np.float64)+0.5
        c = np.asarray(c, dtype=np.float64)

        e5 = np.exp(-5.0 * c)
        e8 = np.exp(-8.0 * c)
        e9 = np.exp(-9.0 * c)

        # Scaled coefficients (divide A,B,C by e^{9c})
        A = e9 - 3.0 * e8 + 3.0 * e5 - 1.0
        B = e9 * (2*rho - 1) + (2 - 4 * rho) * e8 + (2*rho - 1) * e5
        C = e9 * (rho - 0.5)**2 - (2 * rho**2 + 0.5) * e8 + (rho + 0.5)**2 * e5

        delta = B * B - 4.0 * A * C
        neg_delta = delta < 0
        safe_delta = np.where(neg_delta, 0.0, delta)

        denom = 2.0 * A
        b0 = np.where(denom == 0.0, np.nan, (-B - np.sqrt(safe_delta)) / denom)
        invalid = (~np.isfinite(b0)) | neg_delta | (b0 < 0.0)
        return np.where(invalid, 0.0, b0)


    def _update_privacy_accountant(self, steps):
        """
        Compute privacy budget ε using optimized RDP accounting with T-privacy loss.
        This provides tighter privacy bounds compared to simple approximation.
        Uses caching to avoid redundant computation for the same step count.
        """
        # Check cache first to avoid redundant computation
        if steps in self._privacy_cache:
            eps = self._privacy_cache[steps]
            if eps >= self.privacy_budget_limit:
                self.should_stop_training = True
            return eps
        
        L = int(self.args.seq_len + self.args.pred_len)
        stride = getattr(self.args, 'sampling_stride', L)
        w = math.ceil((self.w + L - 1) / stride)
        sample_num = math.floor((self.data_size - L) / stride) + 1
        t = int(math.floor(sample_num / max(1, self.args.batch_size)))

        def T_privacy_loss(alpha, steps, w, t):
            """Compute T-privacy loss function for RDP accounting"""
            sigma2 = self.dp_sigma * self.dp_sigma
            # When tau values overflow, return inf to skip this alpha
            c = (alpha - 1) * alpha / (2 * sigma2)
            a_value = math.ceil(w/2)/t
            b_value = math.ceil((w-t)/2)/t
            if t > w:
                _valid = 0
                rho = None
            else:
                _valid = 1
                rho = a_value - b_value - 1/2
            b0 = self.rho_stable_b0_from_c(rho, np.array([c]), valid=_valid)
            
            e_ = (2 * math.ceil(b0 * t) -1)
            m = max(np.floor((w - e_) / t), 0)

            exp1_arg = (alpha - 1) * alpha * (m * m) / sigma2 / 2
            exp2_arg = (alpha - 1) * alpha * ((m + 1) * (m + 1)) / sigma2 / 2
            exp3_arg = (alpha - 1) * alpha * ((m + 2) * (m + 2)) / sigma2 / 2

            c1 = ((m + 2) * t - w) ** 2 / (4 * t * t)
            c2 = ((m + 2) * t - w) * (w - m * t) / (2 * t * t)
            c3 = (w - m * t) ** 2 / (4 * t * t)
            
           
            # log(c1*exp(a1) + c2*exp(a2) + c3*exp(a3)) 
            # = max_a + log(c1*exp(a1-max_a) + c2*exp(a2-max_a) + c3*exp(a3-max_a))
            max_arg = max(exp1_arg, exp2_arg, exp3_arg)
            base_inner = (c1 * np.exp(exp1_arg - max_arg) + 
                          c2 * np.exp(exp2_arg - max_arg) + 
                          c3 * np.exp(exp3_arg - max_arg))
            
            if base_inner <= 0:
                return float('inf'), float('inf'), float('inf')
            
            log_base = max_arg + np.log(base_inner)
            step_loss = steps / (alpha - 1) * log_base
            para_loss = (np.log(1/self.dp_delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
            return step_loss+para_loss, step_loss, para_loss

        min_alpha = 2  # alpha must be > 1
        min_eps = sys.maxsize
        min_step_loss = 0 
        min_para_loss = 0
        
        # Search over range of alpha values to find minimum
        for alpha_value in range(2, 64):  # Start from 2 to avoid division by zero
            loss, step_loss, para_loss = T_privacy_loss(alpha_value, steps, w, t)
            if loss < min_eps:
                min_eps = loss
                min_alpha = alpha_value
                min_step_loss = step_loss
                min_para_loss = para_loss
        # print("Optimal alpha:", min_alpha, "Min ε:", min_eps, "Step loss:", min_step_loss, "Para loss:", min_para_loss)
        # Convert RDP to (ε, δ)-DP using optimal conversion
        eps = min_eps
        
        # Cache the result for future lookups
        self._privacy_cache[steps] = eps
        
        # Check if privacy budget limit is exceeded
        if eps >= self.privacy_budget_limit:
            print(f"Privacy budget limit reached! Current ε={eps:.4f}, limit={self.privacy_budget_limit}")
            self.should_stop_training = True
        return eps


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
        _, train_loader = self._get_data(flag='train')
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
        # Log and set validation interval (default 10; can be overridden via args.val_interval_batches)
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
                prev_epoch_ckpt = None
            
            epoch_loss_sum = 0.0
            epoch_num = 0
            print(f"Epoch [{epoch+1}/{self.args.train_epochs}] started")
            
            for i, batch in enumerate(train_loader):
                if self.should_stop_training:
                    print(f"Training stopped at epoch {epoch+1}, batch {i+1} due to privacy budget limit")
                    break

                # Pre-check: if the next step would exceed privacy budget, stop before optimizer step
                eps_next = self._update_privacy_accountant(self.total_steps + 1)
                if eps_next >= self.privacy_budget_limit:
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
                
                # Compute differentially private gradients (using vectorized vmap)
                dp_grads, _ = self._compute_dp_gradients(
                    batch_x, batch_y, criterion, batch_x_mark, batch_y_mark
                )
                
                if dp_grads is not None:
                    optimizer.zero_grad()
                    # Apply DP gradients - dp_grads is aligned with named_parameters order
                    for (name, param), dp_grad in zip(self.model.named_parameters(), dp_grads):
                        if param.requires_grad and dp_grad is not None:
                            param.grad = dp_grad / self.batch_size
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
                        # Accumulate the loss weighted by the current batch size.
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
                eps = self._update_privacy_accountant(self.total_steps)
                # Privacy budget check is done inside _update_privacy_accountant
                
                # Show batch progress every 10 batches
                if (i + 1) % 5 == 0:
                    print(f"  Batch [{i+1}/{len(train_loader)}] processed")
                    print("Current privacy budget: ", eps, "Total steps: ", self.total_steps)
            
            # Log epoch statistics
            avg_epoch_loss = (epoch_loss_sum / epoch_num) if epoch_num > 0 else 0
            print(f"Epoch [{epoch+1}/{self.args.train_epochs}] completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Clear GPU memory before validation/test to prevent OOM
            if self.args.use_gpu:
                torch.cuda.empty_cache()
            
            # Validation and early stopping using EarlyStopping class
            val_loss = self.vali(vali_loader, criterion)
            
            # Test evaluation using the SAME criterion as validation
            # test_loss = self.test(setting=None, criterion=criterion)
            # print(f"Epoch [{epoch+1}/{self.args.train_epochs}] - Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            early_stopping(val_loss, self.model, path)
            
            # Removed best_meta.json writes per user request; checkpoint.pth reflects latest best
            
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
        
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
        final_test_results = self.test(setting=None)
        print(f"Final Test Loss: {final_test_results:.4f}")
        
        return self.model
