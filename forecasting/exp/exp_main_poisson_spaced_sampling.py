"""exp_main_poisson_spaced_sampling.py

Experiment runner implementing Poisson spaced sampling with differential privacy.

DP notes:
- Uses Poisson spaced sampling strategy for privacy amplification.
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
from scipy.special import logsumexp, comb


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    """Main experiment class for Poisson spaced sampling with differential privacy.

    This class implements DP-SGD training with Poisson spaced sampling strategy.
    For spaced sampling, noise is scaled by 2*C (sigma = 2 * clipping_norm).

    Attributes:
        dp_sigma: Noise multiplier for DP (sourced from args.dp_sigma).
        dp_delta: Delta parameter for (ε,δ)-DP (sourced from args.dp_delta).
        clipping_norm: Per-sample gradient clipping bound C (sourced from args.clipping_norm).
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
        self.args = args
        self.dp_sigma = getattr(args, 'dp_sigma', 1.0)  # DP: noise multiplier σ
        self.dp_delta = getattr(args, 'dp_delta', 1e-5)  # DP: delta parameter δ
        self.sensitivity = getattr(args, 'sensitivity', 5)  # DP: sensitivity parameter
        self.privacy_budget_limit = getattr(args, 'privacy_budget_limit', 10.0)  # DP: max epsilon budget
        self.clipping_norm = getattr(args, 'clipping_norm', 0.5)  # DP: gradient clipping bound C
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

        Implements DP-SGD for Poisson spaced sampling using vectorized per-sample gradients (vmap):
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
        # This allows dropout to work correctly - different mask for each sample
        ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0), randomness='different')
        
        per_sample_grads_dict = ft_compute_sample_grad(params, buffers, batch_x, batch_y)
        
        # Convert dict to list in parameter order
        return [per_sample_grads_dict[name] for name in params.keys()]


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


    def select_space_distance(self):
        print("before select lambda, data_size:", self.data_size, "w:", self.w)
        """Select optimal lambda (sampling rate) using vectorized computation.
        
        This function finds the lambda value that maximizes the number of training
        steps while staying within the privacy budget. Uses numpy vectorization
        over the alpha dimension for efficiency, but linear search over steps
        since the privacy loss is not monotonic in step count.
        """
        epoch = 100
        L = int(self.args.seq_len + self.args.pred_len)
        w = math.ceil((self.w + L -1) / L)
        t = int(math.floor(self.data_size / L / max(1, self.args.batch_size)))
        sample_length = math.floor(self.data_size / L)
        steps = epoch * t
        
        sigma2 = self.dp_sigma * self.dp_sigma
        
        # Pre-compute alpha-dependent constants (vectorized over alpha)
        # Alpha ranges from 2 to 64 (inclusive), alpha_m1 = alpha - 1 for RDP formula
        alphas = np.arange(2, 65)  # 2 to 64 inclusive
        alpha_m1 = alphas - 1  # (alpha - 1) term in RDP accounting
        
        # tau values for all alphas (some may overflow to inf, resulting in nan for b0)
        # This is OK because we use nanmin later to filter out invalid alphas
        tau_exp = alpha_m1 * alphas / (2 * sigma2)
    
        # para_loss for all alphas (constant term independent of step)
        para_loss = (np.log(1/self.dp_delta) + alpha_m1 * np.log(1 - 1/alphas) - np.log(alphas)) / alpha_m1
        
        lams = np.arange(1/t, 1, 0.01)
        max_step = 0
        min_lam = 1.0/t
        
        for lam in lams:
            batch_per_lam = max(1, math.floor(self.args.batch_size / lam))
            if batch_per_lam > sample_length:
                continue
            zeta = max(np.floor(sample_length / batch_per_lam), 1)
            if zeta < 7:
                continue
            xi = self.args.batch_size / batch_per_lam
            
            # Process each alpha separately since m depends on alpha
            all_log_bases = []
            for alpha_idx in range(len(alphas)):
                a_value = math.ceil(w/2)/zeta
                b_value = math.ceil((w-zeta)/2)/zeta
                if zeta > w:
                    _valid = 0
                    rho = None
                else:
                    _valid = 1
                    rho = a_value - b_value - 1/2
                b0 = self.rho_stable_b0_from_c(rho, tau_exp, valid=_valid)
                if not np.isfinite(b0[alpha_idx]):
                    all_log_bases.append(np.nan)
                    continue
                e_single = 2 * math.ceil(b0[alpha_idx] * zeta) -1
                m_single = max(np.floor((w - e_single) / zeta), 0)

                # Skip if m is invalid
                if np.isnan(m_single) or m_single < 0:
                    all_log_bases.append(np.nan)
                    continue

                # Weight calculations for this alpha
                w_p1 = (zeta - np.floor((w - m_single * zeta) / 2)) / zeta
                w_p2 = (zeta - np.ceil((w - m_single * zeta) / 2)) / zeta
                w_m1 = np.floor((w - m_single * zeta) / 2) / zeta
                w_m2 = np.ceil((w - m_single * zeta) / 2) / zeta
                
                w0 = (1 - xi) ** 2 + (1 - xi) * xi * (w_p1 + w_p2) + xi * xi * (w_p1 * w_p2)
                w1 = (1 - xi) * xi * (w_m1 + w_m2) + xi * xi * (w_p1 * w_m2 + w_p2 * w_m1)
                w2 = xi * xi * (w_m1 * w_m2)
                
                # Compute log terms for this alpha
                log_terms = []
                m_int = int(m_single)
                tau_exp_single = tau_exp[alpha_idx]
                
                for r in range(0, m_int + 1):
                    if xi == 1:
                        if w0 > 0:
                            log_terms.append(np.log(w0) + r*r*tau_exp_single)
                        if w1 > 0:
                            log_terms.append(np.log(w1) + (r+1)*(r+1)*tau_exp_single)
                        if w2 > 0:
                            log_terms.append(np.log(w2) + (r+2)*(r+2)*tau_exp_single)
                    else:
                        log_term0 = np.log(w0) + np.log(comb(m_int, r, exact=False)) + (m_int-r)*np.log(1-xi) + r*np.log(xi) + r*r*tau_exp_single
                        log_terms.append(log_term0)
                        log_term1 = np.log(w1) + np.log(comb(m_int, r, exact=False)) + (m_int-r)*np.log(1-xi) + r*np.log(xi) + (r+1)*(r+1)*tau_exp_single
                        log_terms.append(log_term1)
                        log_term2 = np.log(w2) + np.log(comb(m_int, r, exact=False)) + (m_int-r)*np.log(1-xi) + r*np.log(xi) + (r+2)*(r+2)*tau_exp_single
                        log_terms.append(log_term2)
                
                # Compute log_base for this alpha
                if len(log_terms) > 0:
                    log_base_single = logsumexp(log_terms)
                    all_log_bases.append(log_base_single)
                else:
                    all_log_bases.append(np.nan)
            
            # Convert to numpy array for vectorized operations
            log_base = np.array(all_log_bases)
            
            # Linear search over steps
            # Privacy loss is cumulative - once min_loss exceeds limit, all subsequent steps will too
            for step in range(1, steps + 1):
                # Compute privacy loss (vectorized over alphas)
                step_loss = step / alpha_m1 * log_base
                total_loss = step_loss + para_loss
                # Use nanmin to ignore nan values (some alphas may overflow)
                min_loss = np.nanmin(total_loss)
                
                # Skip invalid results (nan, -inf means degenerate case where all weights are 0)
                if np.isnan(min_loss) or np.isinf(min_loss) or min_loss > self.privacy_budget_limit:
                    # Loss is cumulative, no need to check further steps
                    break
                
                if step >= max_step:
                    max_step = step
                    min_lam = lam
            # print(f"Lambda: {lam}, step: {step}, min_loss:{min_loss} min_lam: {min_lam}, Max Step: {max_step}")
        print("lambda selected:", min_lam, "with max steps:", max_step)
        return min_lam


        
        # # Search over range of alpha values to find minimum
        # for lam, alpha in paras:  # Start from 2 to avoid division by zero
        #     loss, _, _, xi = T_privacy_loss(alpha, w, t, lam)
        #     if loss < min_eps:
        #         min_eps = loss
        #         min_alpha = alpha
        #         min_lam = xi
        # print(sample_length, min_alpha, "lambda", min_lam)
        # return min_lam


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
        w = math.ceil((self.w + L - 1) / L)
        sample_length = math.floor(self.data_size / L)
        zeta = np.floor(sample_length / math.floor(self.args.batch_size / self.args.lam))     
        xi = self.args.batch_size / math.floor(self.args.batch_size / self.args.lam)
        # print("parameters", xi, zeta)

        def T_privacy_loss(alpha, steps):
            """Compute T-privacy loss function for RDP accounting"""
            sigma2 = self.dp_sigma * self.dp_sigma
            c = (alpha - 1) * alpha / (2 * sigma2)
            a_value = math.ceil(w/2)/zeta
            b_value = math.ceil((w-zeta)/2)/zeta
            if zeta > w:
                _valid = 0
                rho = None
            else:
                _valid = 1
                rho = a_value - b_value - 1/2
            b0 = self.rho_stable_b0_from_c(rho, np.array([c]), valid=_valid)
            if not np.isfinite(b0):
                return float('inf'), float('inf'), float('inf')
            e_ = (2 * math.ceil(b0 * zeta) -1)
            m = max(0, math.floor((w - e_) / zeta))
            # if w < zeta or (zeta <= w <= 2 * zeta and w - zeta < e_):
            #     m = 0
            # elif zeta <= w <= 2 * zeta and w - zeta >= e_:
            #     m = 1
            # else:
            #     m = np.floor((w - e_) / zeta)

            # if np.isnan(m) or m < 0:
            #     return float('inf'), float('inf'), float('inf')

            # 计算权重（与原始相同）
            w_p1 = (zeta - np.floor((w - m * zeta) / 2)) / zeta
            w_p2 = (zeta - np.ceil((w - m * zeta) / 2)) / zeta
            w_m1 = np.floor((w - m * zeta) / 2) / zeta
            w_m2 = np.ceil((w - m * zeta) / 2) / zeta
            
            w0 = (1 - xi) ** 2 + (1 - xi) * xi * (w_p1 + w_p2) + xi * xi * (w_p1 * w_p2)
            w1 = (1 - xi) * xi * (w_m1 + w_m2) + xi * xi * (w_p1 * w_m2 + w_p2 * w_m1)
            w2 = xi * xi * (w_m1 * w_m2)
            
            log_terms = []
            m_int = int(m)
            for r in range(0, m_int + 1):
                if xi == 1:
                    if w0 > 0:
                        log_terms.append(np.log(w0) + r*r*c)
                    if w1 > 0:
                        log_terms.append(np.log(w1) + (r+1)*(r+1)*c)
                    if w2 > 0:
                        log_terms.append(np.log(w2) + (r+2)*(r+2)*c)
                else:
                    log_term0 = np.log(w0) + np.log(comb(m_int, r, exact=False)) + (m_int-r)*np.log(1-xi) + r*np.log(xi) + (alpha-1)*alpha*r*r/2/sigma2
                    log_term1 = np.log(w1) + np.log(comb(m_int, r, exact=False)) + (m_int-r)*np.log(1-xi) + r*np.log(xi) + (alpha-1)*alpha*(r+1)*(r+1)/2/sigma2
                    log_term2 = np.log(w2) + np.log(comb(m_int, r, exact=False)) + (m_int-r)*np.log(1-xi) + r*np.log(xi) + (alpha-1)*alpha*(r+2)*(r+2)/2/sigma2
                    log_terms.append(log_term0)
                    log_terms.append(log_term1)
                    log_terms.append(log_term2)
            
            # 使用logsumexp安全地计算log(base)
            log_base = logsumexp(log_terms)
            # print(log_base)

            # # Compute log_exp terms for all alphas (vectorized)
            # log_exp1 = (alpha - 1) * alpha * (m * m) / sigma2 / 2
            # log_exp2 = (alpha - 1) * alpha * ((m + 1) * (m + 1)) / sigma2 / 2
            # log_exp3 = (alpha - 1) * alpha * ((m + 2) * (m + 2)) / sigma2 / 2
            
            # # Weight calculations (vectorized)
            # w_p1 = (zeta - np.floor((w - m * zeta) / 2)) / zeta
            # w_p2 = (zeta - np.ceil((w - m * zeta) / 2)) / zeta
            # w_m1 = np.ceil((w - m * zeta) / 2) / zeta
            # w_m2 = np.floor((w - m * zeta) / 2) / zeta
            
            # w0 = (1 - xi) ** 2 + (1 - xi) * xi * (w_p1 + w_p2) + xi * xi * (w_p1 * w_p2)
            # w1 = (1 - xi) * xi * (w_m1 + w_m2) + xi * xi * (w_p1 * w_m2 + w_p2 * w_m1)
            # w2 = xi * xi * (w_m1 * w_m2)
            
            # # Use logsumexp for numerical stability
            # log_w0 = np.where(w0 > 0, np.log(w0), -np.inf)
            # log_w1 = np.where(w1 > 0, np.log(w1), -np.inf)
            # log_w2 = np.where(w2 > 0, np.log(w2), -np.inf)
            
            # log_terms = np.stack([log_w0 + log_exp1, log_w1 + log_exp2, log_w2 + log_exp3], axis=0)
            # log_base = logsumexp(log_terms, axis=0)
            # print("log_base", log_base)
            
            # 计算损失
            step_loss = steps / (alpha - 1) * log_base  # 注意：log_base已经是log值
            
            # 参数损失部分
            para_loss = (np.log(1/self.dp_delta) + (alpha - 1) * np.log(1 - 1/alpha) - np.log(alpha)) / (alpha - 1)
            
            # print("logsumexp step loss", alpha, step_loss, "para_loss", para_loss)
            return step_loss+para_loss, step_loss, para_loss
        
        # Find optimal alpha that minimizes privacy loss
        min_alpha = 2  # alpha must be > 1
        min_eps = sys.maxsize
        min_step_loss = 0 
        min_para_loss = 0
        
        # Search over range of alpha values to find minimum
        for alpha_value in range(2, 64):  # Start from 2 to avoid division by zero
            loss, step_loss, para_loss = T_privacy_loss(alpha_value, steps)
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
        # Build a temporary training dataset to get data_size and w before selecting lambda
        from data_provider.data_loader import Dataset_Custom
        timeenc = 0 if self.args.embed != 'timeF' else 1
        temp_dataset = Dataset_Custom(
            root_path=self.args.root_path,
            data_path=self.args.data_path,
            flag='train',
            size=[self.args.seq_len, self.args.label_len, self.args.pred_len],
            features=self.args.features,
            target=self.args.target,
            timeenc=timeenc,
            freq=self.args.freq,
        )
        self.data_size = len(temp_dataset)
        self.w = int(self.data_size * self.args.w)
        # self.w = int(self.args.w)

        # Now select the window sampling rate using accurate data_size and w
        lam = self.select_space_distance()
        # Propagate selected lambda to args for downstream data loader/sampler usage
        setattr(self.args, 'lam', float(lam))
        print(f"Selected lambda (sampling rate): {self.args.lam}")
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
                eps = self._update_privacy_accountant(self.total_steps)
                # Privacy budget check is done inside _update_privacy_accountant
                
                # Show batch progress every 10 batches
                if (i + 1) % 5 == 0:
                    print(f"  Batch [{i+1}/{len(train_loader)}] processed")
                    print("Current privacy budget: ", eps, "Total steps:", self.total_steps)
            
            # Log epoch statistics（样本加权平均）
            avg_epoch_loss = (epoch_loss_sum / epoch_num) if epoch_num > 0 else 0
            print(f"Epoch [{epoch+1}/{self.args.train_epochs}] completed. Average loss: {avg_epoch_loss:.4f}")
            
            # Clear GPU memory before validation/test to prevent OOM
            if self.args.use_gpu:
                torch.cuda.empty_cache()
            
            # Validation and early stopping using EarlyStopping class
            val_loss = self.vali(vali_loader, criterion)
            
            # # Test evaluation using the SAME criterion as validation
            # test_loss = self.test(setting=None, criterion=criterion)
            # print(f"Epoch [{epoch+1}/{self.args.train_epochs}] - Validation Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f}")
            
            early_stopping(val_loss, self.model, path)
            
            # Removed best_meta.json writes; checkpoint.pth reflects the best so far
            
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