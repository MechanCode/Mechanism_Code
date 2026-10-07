import math

import numpy as np
from scipy.special import comb, logsumexp


class Exp_Poisson_TSC:
    @staticmethod
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

    def select_space_distance(self):
        epoch = 100
        L = self.args.seq_len
        stride = getattr(self.args, 'sampling_stride', None) or L
        w = math.ceil((self.w + L - 1) / stride)
        sample_length = math.floor((self.data_size - L) / stride) + 1
        t = int(math.floor(sample_length / max(1, self.args.batch_size)))
        steps = epoch * t

        sigma2 = self.dp_sigma * self.dp_sigma


        alphas = np.arange(2, 65)
        alpha_m1 = alphas - 1


        tau_exp = alpha_m1 * alphas / (2 * sigma2)


        para_loss = (np.log(1/self.dp_delta) + alpha_m1 * np.log(1 - 1/alphas) - np.log(alphas)) / alpha_m1

        lam_min = 1.0 / t
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


                if np.isnan(min_loss) or np.isinf(min_loss) or min_loss > self.privacy_budget_limit:

                    break

                if step >= max_step:
                    max_step = step
                    min_lam = lam

        print("lambda selected:", min_lam, "with max steps:", max_step)
        return min_lam
