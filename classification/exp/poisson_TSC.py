"""
Poisson Sampling TSC - Time Series Classification with Poisson Sampling DP-SGD
Removed stratification logic, only Poisson sampling remains.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import math
import sys
import logging
from scipy.special import comb, logsumexp
from models.DLinear import Model as DLinearClassifier
from models.DLinear_Stable import Model as DLinearStableClassifier
from models.DLinear_Stable import Model_GAP as DLinearGAPClassifier
from models.PatchTST import Model as PatchTSTClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from data_provider.data_factory_stratified import data_provider

logger = logging.getLogger(__name__)


class Exp_Poisson_TSC:
    """
    Time Series Classification with Poisson Sampling DP-SGD.
    Uses Poisson sampling for privacy-preserving training.
    """
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.optimizer = optim.SGD(self.model.parameters(), lr=args.learning_rate)
        
        self.lr_step_size = getattr(args, 'lr_step_size', 10)
        self.lr_gamma = getattr(args, 'lr_gamma', 0.9)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.lr_step_size,
            gamma=self.lr_gamma
        )
        
        # Label mode
        self.label_mode = getattr(args, 'label_mode', 'sequence')
        
        # Differential Privacy parameters
        self.dp_sigma = getattr(args, 'dp_sigma', 1.0)
        self.dp_delta = getattr(args, 'dp_delta', 1e-5)
        self.clipping_norm = getattr(args, 'clipping_norm', 1.0)
        self.privacy_budget_limit = getattr(args, 'privacy_budget_limit', 10.0)
        self.total_steps = 0
        self.should_stop_training = False
        if not hasattr(self.args, 'lam'):
            setattr(self.args, 'lam', 1.0)
        
        # Data parameters
        self.data_size = 0
        self.w = 0
        self.sampling_rate = 0
        
        # Lambda parameter for spaced sampling (to be selected automatically)
        self.lam = 1.0

    def _acquire_device(self):
        if self.args.use_gpu:
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model_type = getattr(self.args, 'model', 'Rocket')
        if model_type == 'DLinear':
            model = DLinearClassifier(self.args)
        elif model_type == 'DLinear_Stable':
            model = DLinearStableClassifier(self.args)
        elif model_type == 'DLinear_GAP':
            model = DLinearGAPClassifier(self.args)
        elif model_type == 'PatchTST':
            model = PatchTSTClassifier(self.args)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        return model

    def _get_data(self, flag):
        if flag == 'train':
            _, data_loader, data_size = data_provider(self.args, flag)
            self.data_size = data_size
            w_ratio = getattr(self.args, 'w', 0.01)
            self.w = int(data_size * w_ratio)
            print(f"DP Params: Data Size={data_size}, w={self.w}")
        else:
            _, data_loader = data_provider(self.args, flag)
        return data_loader

    def _select_optimizer(self):
        return optim.SGD(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.CrossEntropyLoss(reduction='none')

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
        epoch = 100
        L = self.args.seq_len
        stride = getattr(self.args, 'sampling_stride', None) or L
        w = math.ceil((self.w + L - 1) / stride)
        sample_length = math.floor((self.data_size - L) / stride) + 1
        t = int(math.floor(sample_length / max(1, self.args.batch_size)))
        steps = epoch * t
        
        sigma2 = self.dp_sigma * self.dp_sigma
        # Pre-compute alpha-dependent constants (vectorized over alpha)
        # Alpha ranges from 2 to 64 (inclusive), alpha_m1 = alpha - 1 for RDP formula
        alphas = np.arange(2, 65)  # 2 to 64 inclusive
        alpha_m1 = alphas - 1  # (alpha - 1) term in RDP accounting
        # tau values for all alphas (some may overflow to inf, resulting in nan for b0)
        # This is OK because we use nanmin later to filter out invalid alphas
        tau_exp = alpha_m1 * alphas / (2 * sigma2)

        # When tau values overflow, A/B/C become nan, and so does b0
        # This is intentional - these alpha values will be filtered by nanmin
        # b0 = self._stable_b0_from_c(tau_exp, upper=0.5)
        
        # para_loss for all alphas (constant term independent of step)
        para_loss = (np.log(1/self.dp_delta) + alpha_m1 * np.log(1 - 1/alphas) - np.log(alphas)) / alpha_m1

        lam_min = 1.0 / t  # lambda minimum: ensures batch_per_lam <= data_size
        lams = np.arange(lam_min, 1, 0.01)
        max_step = 0
        min_lam = lam_min

        for lam in lams:
            batch_per_lam = max(1, math.floor(self.args.batch_size / lam))
            if batch_per_lam > sample_length:
                continue
            zeta = int(max(np.floor(sample_length / batch_per_lam), 1))
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
                            log_term0 = np.log(w0) + r*r*tau_exp_single
                            log_terms.append(log_term0)
                        if w1 > 0:
                            log_term1 = np.log(w1) + (r+1)*(r+1)*tau_exp_single
                            log_terms.append(log_term1)
                        if w2 > 0:
                            log_term2 = np.log(w2) + (r+2)*(r+2)*tau_exp_single
                            log_terms.append(log_term2)
                    else:
                        log_binom = np.log(comb(m_single, r, exact=False)) + (m_single-r)*np.log(1-xi) + r*np.log(xi)
                        if w0 > 0:
                            log_terms.append(np.log(w0) + log_binom + r*r*tau_exp_single)
                        if w1 > 0:
                            log_terms.append(np.log(w1) + log_binom + (r+1)*(r+1)*tau_exp_single)
                        if w2 > 0:
                            log_terms.append(np.log(w2) + log_binom + (r+2)*(r+2)*tau_exp_single)
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

    def _update_privacy_accountant(self, steps):
        """
        Compute privacy budget ε using optimized RDP accounting with T-privacy loss
        This provides tighter privacy bounds compared to simple approximation
        """
        L = self.args.seq_len
        stride = getattr(self.args, 'sampling_stride', None) or L
        w = math.ceil((self.w + L - 1) / stride)
        sample_length = math.floor((self.data_size - L) / stride) + 1
        zeta = np.floor(sample_length / math.floor(self.args.batch_size / self.args.lam))     
        xi = self.args.batch_size / math.floor(self.args.batch_size / self.args.lam)

        def T_privacy_loss(alpha, steps):
            """Compute T-privacy loss function for RDP accounting with numerical stability"""
            sigma2 = self.dp_sigma * self.dp_sigma
            c = (alpha - 1) * alpha / (2 * sigma2)
            # When tau values overflow, return inf to skip this alpha
            a_value = math.ceil(w/2)/zeta
            b_value = math.ceil((w-zeta)/2)/zeta
            if zeta > w:
                _valid = 0
                rho = None
            else:
                _valid = 1
                rho = a_value - b_value - 1/2
            b0 = self.rho_stable_b0_from_c(rho, np.array([c]), valid=_valid)
            
            e_ = (2 * math.ceil(b0 * zeta) -1)
            m = max(np.floor((w - e_) / zeta), 0)

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
                        log_terms.append(np.log(w0) + r*r*(alpha-1)*alpha/2/sigma2)
                    if w1 > 0:
                        log_terms.append(np.log(w1) + (r+1)*(r+1)*(alpha-1)*alpha/2/sigma2)
                    if w2 > 0:
                        log_terms.append(np.log(w2) + (r+2)*(r+2)*(alpha-1)*alpha/2/sigma2)
                else:
                    log_binom = np.log(comb(m, r, exact=False)) + (m-r)*np.log(1-xi) + r*np.log(xi)
                    if w0 > 0:
                        log_terms.append(np.log(w0) + log_binom + (alpha-1)*alpha*r*r/2/sigma2)
                    if w1 > 0:
                        log_terms.append(np.log(w1) + log_binom + (alpha-1)*alpha*(r+1)*(r+1)/2/sigma2)
                    if w2 > 0:
                        log_terms.append(np.log(w2) + log_binom + (alpha-1)*alpha*(r+2)*(r+2)/2/sigma2)
            
            valid_terms = [t for t in log_terms if not (np.isnan(t) or (np.isinf(t) and t < 0))]
            if not valid_terms:
                return float('inf'), float('inf'), float('inf')
            
            log_base = logsumexp(valid_terms)
            
            if np.isnan(log_base) or np.isinf(log_base) or log_base < 0:
                return float('inf'), float('inf'), float('inf')

            step_loss = steps / (alpha - 1) * log_base
            para_loss = (np.log(1/self.dp_delta) + (alpha - 1) * np.log(1 - 1/alpha) - np.log(alpha)) / (alpha - 1)
            total_loss = step_loss + para_loss
            return total_loss, step_loss, para_loss
        
        # Find optimal alpha that minimizes privacy loss
        min_alpha = 2  # alpha must be > 1
        min_eps = sys.maxsize
        min_step_loss = 0 
        min_para_loss = 0
        
        # Search over range of alpha values to find minimum
        for alpha_value in range(2, 65):  # Start from 2 to avoid division by zero
            loss, step_loss, para_loss = T_privacy_loss(alpha_value, steps)
            if loss < min_eps:
                min_eps = loss
                min_alpha = alpha_value
                min_step_loss = step_loss
                min_para_loss = para_loss
        # print("Optimal alpha:", min_alpha, "Min ε:", min_eps, "Step loss:", min_step_loss, "Para loss:", min_para_loss)
        # Convert RDP to (ε, δ)-DP using optimal conversion
        
        # print("privacy loss", min_step_loss, min_para_loss)
        eps = min_eps
        # Check if privacy budget limit is exceeded
        if eps >= self.privacy_budget_limit:
            print(f"Privacy budget limit reached! Current ε={eps:.4f}, limit={self.privacy_budget_limit}")
            self.should_stop_training = True
        return eps

    def train(self, setting):
        # First call to get data_size for lambda selection
        train_loader_temp = self._get_data(flag='train')
        vali_loader = self._get_data(flag='val')
        test_loader = self._get_data(flag='test')
        
        # Automatically select optimal lambda using select_space_distance
        self.lam = self.select_space_distance()
        self.args.lam = self.lam
        print(f"Using lambda={self.lam} for spaced sampling")
        
        # Re-create train_loader with the optimized lambda
        train_loader = self._get_data(flag='train')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        print(f"Starting Poisson Sampling DP-SGD Training. Sigma={self.dp_sigma}, Clipping={self.clipping_norm}, Lambda={self.lam}")
        
        # Best model tracking
        best_vali_acc = 0.0
        best_vali_f1 = 0.0
        best_model_path = os.path.join(path, 'checkpoint.pth')

        for epoch in range(self.args.train_epochs):
            if self.should_stop_training:
                print("Stopping training early due to privacy budget.")
                break
                
            self.model.train()
            train_loss = []
            
            for i, (batch_x, batch_y) in enumerate(train_loader):
                if self.should_stop_training:
                    break
                
                next_eps = self._update_privacy_accountant(self.total_steps + 1)
                if next_eps >= self.privacy_budget_limit:
                    print(f"Privacy budget will be exceeded after next step! Predicted ε={next_eps:.4f}, limit={self.privacy_budget_limit}")
                    self.should_stop_training = True
                    vali_loss, vali_acc, vali_f1 = self.valid(vali_loader)
                    if vali_f1 > best_vali_f1:
                        best_vali_f1 = vali_f1
                        torch.save(self.model.state_dict(), best_model_path)
                    break
                    
                batch_x = batch_x.float().to(self.device)
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(0)
                
                # Handle batch_y based on label_mode
                batch_y = batch_y.long().to(self.device)
                if self.label_mode == 'point':
                    # Point mode: batch_y should be [B, seq_len]
                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(0)
                else:
                    # Sequence mode: batch_y should be [B]
                    if batch_y.dim() == 0:
                        batch_y = batch_y.unsqueeze(0)
                    elif batch_y.dim() > 1:
                        batch_y = batch_y.squeeze()
                        if batch_y.dim() == 0:
                            batch_y = batch_y.unsqueeze(0)
        
                self.optimizer.zero_grad()
                
                trainable_params = [p for p in self.model.parameters() if p.requires_grad]
                clipped_grads = {id(p): torch.zeros_like(p) for p in trainable_params}
                
                batch_grad_norms = []
                batch_size = batch_x.size(0)

                for idx in range(batch_size):
                    sample_x = batch_x[idx:idx+1]
                    sample_target = batch_y[idx:idx+1] if self.label_mode == 'point' else batch_y[idx:idx+1]
                    
                    output = self.model(sample_x)
                    
                    if self.label_mode == 'point':
                        # Point mode: output [1, seq_len, num_classes], target [1, seq_len]
                        # Reshape for loss: [seq_len, num_classes], [seq_len]
                        output_flat = output.view(-1, output.size(-1))
                        target_flat = sample_target.view(-1)
                        loss = self.criterion(output_flat, target_flat).mean()
                    else:
                        # Sequence mode
                        loss = self.criterion(output, sample_target).mean()
                    
                    loss.backward()
                    
                    total_norm = 0
                    for p in trainable_params:
                        if p.grad is not None:
                            total_norm += p.grad.data.norm(2).item() ** 2
                    total_norm = total_norm ** 0.5
                    batch_grad_norms.append(total_norm)
                    
                    clip_coef = min(1, self.clipping_norm / (total_norm + 1e-6))
                    
                    for p in trainable_params:
                        if p.grad is not None:
                            clipped_grads[id(p)] += p.grad.data * clip_coef
                            p.grad.detach_()
                            p.grad.zero_()
                
                for p in trainable_params:
                    noise = torch.normal(0, self.dp_sigma * self.clipping_norm * 2, size=clipped_grads[id(p)].shape, device=self.device)
                    p.grad = (clipped_grads[id(p)] + noise) / self.args.batch_size
                
                self.optimizer.step()
                self.scheduler.step()  
                self.total_steps += 1
                
                with torch.no_grad():
                    outputs = self.model(batch_x)
                    if self.label_mode == 'point':
                        outputs_flat = outputs.view(-1, outputs.size(-1))
                        batch_y_flat = batch_y.view(-1)
                        loss_val = self.criterion(outputs_flat, batch_y_flat).mean().item()
                    else:
                        loss_val = self.criterion(outputs, batch_y).mean().item()
                
                train_loss.append(loss_val)
                
                eps = self._update_privacy_accountant(self.total_steps)
                
                # Perform validation every 5 steps
                if self.total_steps % 5 == 0:
                    vali_loss, vali_acc, vali_f1 = self.valid(vali_loader)
                    # Save best model based on validation F1 score
                    if vali_f1 > best_vali_f1:
                        best_vali_f1 = vali_f1
                        torch.save(self.model.state_dict(), best_model_path)
                    print(f"Step: {self.total_steps} | Train Loss: {loss_val:.7f} | Vali Acc: {vali_acc:.7f} | Vali F1: {vali_f1:.7f} | ε: {eps:.4f}")

            train_loss = np.average(train_loss)
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} | LR: {current_lr:.6f}")
            
            vali_loss, vali_acc, vali_f1 = self.valid(vali_loader)
            test_loss, test_acc, test_f1 = self.valid(test_loader)
            print(f"Epoch: {epoch + 1} End | Vali Acc: {vali_acc:.7f} Vali F1: {vali_f1:.7f} | Test Acc: {test_acc:.7f} Test F1: {test_f1:.7f}")

        # Load best model before returning
        if os.path.exists(best_model_path):
            print(f"Loading best model with validation F1: {best_vali_f1:.4f}")
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        
        return self.model

    def valid(self, vali_loader):
        self.model.eval()
        total_loss = []
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.long().to(self.device)

                # Model now outputs [batch_size, num_classes] or [batch_size, seq_len, num_classes]
                outputs = self.model(batch_x)
                
                if self.label_mode == 'point':
                    # Point-wise: outputs [B, seq_len, num_classes], batch_y [B, seq_len]
                    # Handle dimension issues
                    if outputs.dim() == 2:
                        outputs = outputs.unsqueeze(0)
                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(0)
                    
                    # Reshape for loss calculation: [B*seq_len, num_classes], [B*seq_len]
                    B, L, C = outputs.shape
                    outputs_flat = outputs.view(B * L, C)
                    batch_y_flat = batch_y.view(B * L)
                    
                    loss = self.criterion(outputs_flat, batch_y_flat).mean()
                    pred = outputs.argmax(dim=-1).detach().cpu().numpy().flatten()
                    true = batch_y.detach().cpu().numpy().flatten()
                else:
                    # Sequence-level: outputs [B, num_classes], batch_y [B]
                    # Handle single sample case
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(0)
                    if batch_y.dim() == 0:
                        batch_y = batch_y.unsqueeze(0)
                    batch_y = batch_y.squeeze()
                    if batch_y.dim() == 0:
                        batch_y = batch_y.unsqueeze(0)
                    
                    loss = self.criterion(outputs, batch_y).mean()
                    pred = outputs.argmax(dim=1).detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        accuracy = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='macro')
        
        self.model.train()
        return total_loss, accuracy, f1

    def test(self, setting, test=0):
        test_loader = self._get_data(flag='test')
        
        self.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                if batch_x.dim() == 2:
                    batch_x = batch_x.unsqueeze(0)
                batch_y = batch_y.long().to(self.device)

                # Model now outputs [batch_size, num_classes] or [batch_size, seq_len, num_classes]
                outputs = self.model(batch_x)
                
                if self.label_mode == 'point':
                    # Point-wise: outputs [B, seq_len, num_classes], batch_y [B, seq_len]
                    if outputs.dim() == 2:
                        outputs = outputs.unsqueeze(0)
                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(0)
                    
                    pred = outputs.argmax(dim=-1).detach().cpu().numpy().flatten()
                    true = batch_y.detach().cpu().numpy().flatten()
                else:
                    # Sequence-level
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(0)
                    if batch_y.dim() == 0:
                        batch_y = batch_y.unsqueeze(0)
                    batch_y = batch_y.squeeze()
                    if batch_y.dim() == 0:
                        batch_y = batch_y.unsqueeze(0)
                    
                    pred = outputs.argmax(dim=1).detach().cpu().numpy()
                    true = batch_y.detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)

        preds = np.concatenate(preds)
        trues = np.concatenate(trues)
        
        accuracy = accuracy_score(trues, preds)
        f1 = f1_score(trues, preds, average='macro', zero_division=0)
        
        print('Test Accuracy: {:.6f}, Test F1: {:.6f}'.format(accuracy, f1))
        # print("\nClassification Report:")
        # print(classification_report(trues, preds, zero_division=0))
        # print("\nConfusion Matrix:")
        # print(confusion_matrix(trues, preds))
        
        return accuracy, f1
