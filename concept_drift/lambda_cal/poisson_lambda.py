from dataclasses import dataclass
import math
from numbers import Integral

import numpy as np
from scipy.special import comb, logsumexp


@dataclass(frozen=True)
class LambdaConfig:

    time_series_length: int

    segment_length: int
    stride_length: int | None = None
    batch_size: int = 128
    dp_sigma: float = 1.0
    dp_delta: float = 1e-5
    epsilon: float = 10.0
    w: float = 0.05

    private_length: int | None = None
    search_epochs: int = 100
    lambda_step: float = 0.01
    alpha_min: int = 2
    alpha_max: int = 64
    min_zeta: int = 7

    def __post_init__(self):
        if self.stride_length is None:
            object.__setattr__(self, "stride_length", self.segment_length)
        for name in ("time_series_length", "segment_length", "stride_length",
                     "batch_size", "search_epochs", "alpha_min", "alpha_max", "min_zeta"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.time_series_length < self.segment_length:
            raise ValueError("time_series_length must be >= segment_length")
        n = (self.time_series_length - self.segment_length) // self.stride_length + 1
        if n < self.batch_size:
            raise ValueError(f"candidate_windows={n} < batch_size={self.batch_size}; original search would divide by zero")
        for name in ("dp_sigma", "epsilon", "lambda_step"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not 0 < self.dp_delta < 1:
            raise ValueError("dp_delta must be between 0 and 1")
        if not 0 <= self.w <= 1:
            raise ValueError("w must be between 0 and 1")
        if self.private_length is not None:
            if (isinstance(self.private_length, bool)
                    or not isinstance(self.private_length, Integral)
                    or not 0 <= self.private_length <= self.time_series_length):
                raise ValueError("private_length must be an integer between 0 and time_series_length")
        if self.alpha_min < 2 or self.alpha_max < self.alpha_min:
            raise ValueError("require 2 <= alpha_min <= alpha_max")


@dataclass(frozen=True)
class LambdaResult:
    lam: float
    max_steps: int
    candidate_windows: int
    affected_windows: int
    base_spacing: int
    step_limit: int
    candidates_total: int
    candidates_evaluated: int
    feasible: bool


def rho_stable_b0_from_c(rho, c: np.ndarray, valid=1) -> np.ndarray:

    if valid == 0:
        return np.zeros_like(c, dtype=np.float64)+0.5
    c = np.asarray(c, dtype=np.float64)

    e5 = np.exp(-5.0 * c)
    e8 = np.exp(-8.0 * c)
    e9 = np.exp(-9.0 * c)


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


def select_lambda(config: LambdaConfig) -> LambdaResult:

    epoch = config.search_epochs
    sample_num = (config.time_series_length - config.segment_length) // config.stride_length + 1
    private_length = config.private_length if config.private_length is not None else int(config.time_series_length * config.w)
    w = math.ceil((private_length + config.segment_length - 1) / config.stride_length)
    t = int(math.floor(sample_num / max(1, config.batch_size)))
    sample_length = sample_num
    steps = epoch * t

    sigma2 = config.dp_sigma * config.dp_sigma


    alphas = np.arange(config.alpha_min, config.alpha_max + 1)
    alpha_m1 = alphas - 1


    tau_exp = alpha_m1 * alphas / (2 * sigma2)


    para_loss = (np.log(1/config.dp_delta) + alpha_m1 * np.log(1 - 1/alphas) - np.log(alphas)) / alpha_m1

    lams = np.arange(1/t, 1, config.lambda_step)
    max_step = 0
    min_lam = 1.0/t

    candidates_evaluated = 0
    for lam in lams:
        batch_per_lam = max(1, math.floor(config.batch_size / lam))
        if batch_per_lam > sample_length:
            continue
        zeta = max(np.floor(sample_length / batch_per_lam), 1)
        if zeta < config.min_zeta:
            continue
        candidates_evaluated += 1
        xi = config.batch_size / batch_per_lam


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
            b0 = rho_stable_b0_from_c(rho, tau_exp, valid=_valid)
            if not np.isfinite(b0[alpha_idx]):
                all_log_bases.append(np.nan)
                continue
            e_single = 2 * math.ceil(b0[alpha_idx] * zeta) -1
            m_single = max(np.floor((w - e_single) / zeta), 0)


            if np.isnan(m_single) or m_single < 0:
                all_log_bases.append(np.nan)
                continue


            w_p1 = (zeta - np.floor((w - m_single * zeta) / 2)) / zeta
            w_p2 = (zeta - np.ceil((w - m_single * zeta) / 2)) / zeta
            w_m1 = np.floor((w - m_single * zeta) / 2) / zeta
            w_m2 = np.ceil((w - m_single * zeta) / 2) / zeta

            w0 = (1 - xi) ** 2 + (1 - xi) * xi * (w_p1 + w_p2) + xi * xi * (w_p1 * w_p2)
            w1 = (1 - xi) * xi * (w_m1 + w_m2) + xi * xi * (w_p1 * w_m2 + w_p2 * w_m1)
            w2 = xi * xi * (w_m1 * w_m2)


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


            if len(log_terms) > 0:
                log_base_single = logsumexp(log_terms)
                all_log_bases.append(log_base_single)
            else:
                all_log_bases.append(np.nan)


        log_base = np.array(all_log_bases)


        for step in range(1, steps + 1):

            step_loss = step / alpha_m1 * log_base
            total_loss = step_loss + para_loss

            min_loss = np.nanmin(total_loss)


            if np.isnan(min_loss) or np.isinf(min_loss) or min_loss > config.epsilon:

                break

            if step >= max_step:
                max_step = step
                min_lam = lam
    return LambdaResult(
        lam=float(min_lam), max_steps=max_step,
        candidate_windows=sample_num, affected_windows=w,
        base_spacing=t, step_limit=steps,
        candidates_total=len(lams), candidates_evaluated=candidates_evaluated,
        feasible=max_step > 0,
    )
