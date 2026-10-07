from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp
from scipy.stats import binom

from .sampling import lambda_grid, pss_partition


@dataclass(frozen=True)
class Geometry:
    length: int
    segment_length: int
    stride: int
    event_length: int
    candidates: int
    affected: int

    @classmethod
    def from_training(cls, training, config):
        if "sequence_boundaries" not in training:
            return cls.from_config(training["rows"], config)
        from .data import validate_boundaries
        boundaries = training["sequence_boundaries"]
        validate_boundaries(boundaries, training["rows"])
        lengths = [stop-start for _, start, stop in boundaries]
        if min(lengths) < config.segment_length:
            raise ValueError("Every training sequence needs a complete segment")
        counts = [(n-config.segment_length)//config.train_stride+1 for n in lengths]
        candidates = sum(counts)
        if candidates < config.batch_size:
            raise ValueError("At least K training segments are required")


        w = max(1, math.ceil(config.event_fraction * training["rows"]))
        affected = min(max(counts), (w+config.segment_length-2)//config.train_stride+1)
        return cls(training["rows"], config.segment_length, config.train_stride, w, candidates, affected)

    @classmethod
    def from_config(cls, length, config):
        if length < config.segment_length:
            raise ValueError("Training sequence has no complete segment")
        candidates = (length - config.segment_length) // config.train_stride + 1
        if candidates < config.batch_size:
            raise ValueError("At least K training segments are required")
        w = max(1, math.ceil(config.event_fraction * length))
        affected = min(candidates, (w + config.segment_length - 2) // config.train_stride + 1)
        return cls(length, config.segment_length, config.train_stride, w, candidates, affected)


def bernoulli_log_pmf(probabilities):

    result = np.array([0.0])
    for p in probabilities:
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Invalid Bernoulli probability")
        if p == 0:
            continue
        if p == 1:
            result = np.r_[-np.inf, result]
        else:
            result = np.logaddexp(np.r_[result + math.log1p(-p), -np.inf],
                                 np.r_[-np.inf, result + math.log(p)])
    return result


def gaussian_moment_rdp(log_pmf, orders, noise_multiplier):

    a = np.asarray(orders, dtype=np.float64)
    j = np.arange(len(log_pmf), dtype=np.float64)
    exponent = 2.0 * a[:, None] * (a[:, None] - 1) * j[None, :] ** 2 / noise_multiplier ** 2
    result = logsumexp(np.asarray(log_pmf)[None, :] + exponent, axis=1) / (a - 1)
    if not np.isfinite(result).all():
        raise ArithmeticError("Non-finite RDP bound")
    return np.maximum(result, 0.0)


def affected_patterns(sizes, affected):

    total = sum(sizes)
    if not sizes or min(sizes) < 1 or not 1 <= affected <= total:
        raise ValueError("Invalid partition or affected block length")
    bounds = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    starts = np.arange(total - affected + 1)
    lefts = np.searchsorted(bounds, starts, side="right") - 1
    rights = np.searchsorted(bounds, starts + affected - 1, side="right") - 1
    patterns = set()
    for start, left, right in zip(starts, lefts, rights):
        entries = [(int(min(bounds[i + 1], start + affected) - max(bounds[i], start)),
                    sizes[i]) for i in range(left, right + 1)]

        patterns.add(tuple(sorted(entries)))
    return sorted(patterns)


def pss_rdp(partition, affected, orders, noise_multiplier, patterns=None):
    if patterns is None:
        patterns = affected_patterns(partition.sizes, affected)
    bound = np.zeros(len(orders))
    for entries in patterns:
        p = [partition.activation * count / size for count, size in entries]
        bound = np.maximum(bound, gaussian_moment_rdp(bernoulli_log_pmf(p), orders, noise_multiplier))
    return bound


class RDPAccountant:
    def __init__(self, orders, per_step, delta):
        self.orders = np.asarray(orders, dtype=np.float64)
        self.per_step = np.asarray(per_step, dtype=np.float64)
        if (self.orders.shape != self.per_step.shape or (self.orders <= 1).any()
                or not np.isfinite(self.per_step).all() or (self.per_step <= 0).any()):
            raise ValueError("Require finite positive per-step RDP values")
        self.conversion = (math.log(1 / delta) + (self.orders - 1) * np.log1p(-1 / self.orders)
                           - np.log(self.orders)) / (self.orders - 1)

    def epsilon(self, steps):
        if steps < 0:
            raise ValueError("steps cannot be negative")
        if steps == 0:
            return 0.0
        return float(max(0.0, np.min(steps * self.per_step + self.conversion)))

    def maximum_steps(self, target, *, maximum_steps):
        upper = max(0.0, float(np.max((target - self.conversion) / self.per_step)))
        steps = min(maximum_steps, math.floor(upper))

        while steps and self.epsilon(steps) > target:
            steps -= 1
        while steps < maximum_steps and self.epsilon(steps + 1) <= target:
            steps += 1
        return steps, self.epsilon(steps + 1) > target


def budget_entry(accountant, target, config):
    steps, exhausted = accountant.maximum_steps(target, maximum_steps=config.max_steps)
    current, following = accountant.epsilon(steps), accountant.epsilon(steps + 1)
    if not math.isfinite(current) or current > target or (exhausted and following <= target):
        raise ArithmeticError("Accountant failed the current/next step budget check")
    return {"target_epsilon": target, "steps": steps, "epsilon": current,
            "next_epsilon": following, "budget_exhausted": bool(exhausted)}


def structured_parameters(geometry, config):

    from .structured_privacy.mixtures import WRMixtureParameters
    indices = np.arange(config.batch_size + 1, dtype=np.float64)
    hit_probability = geometry.affected / geometry.candidates
    log_weights = binom.logpmf(indices, config.batch_size, hit_probability)
    return WRMixtureParameters(
        sequence_length=geometry.length, window_length=geometry.segment_length,
        sample_stride=geometry.stride, event_length=geometry.event_length,
        batch_size=config.batch_size, noise_multiplier=config.noise_multiplier,
        num_candidate_windows=geometry.candidates, max_affected_windows=geometry.affected,
        hit_probability=hit_probability, weights=np.exp(log_weights-logsumexp(log_weights)),
        means_p=-2.0*indices, means_q=2.0*indices)


def build_plan(config, train_identity, methods, progress=print):
    from dataclasses import asdict
    geometry = Geometry.from_training(train_identity, config)
    plan = {"protocol": config.to_dict(), "training": train_identity,
            "geometry": asdict(geometry), "methods": {},
            "noise_std_sum": config.noise_multiplier * config.clipping_norm,
            "noise_std_after_division": config.noise_multiplier * config.clipping_norm / config.batch_size,
            "privacy_scope": "training CSV sequence only; per training run",
            "preprocessing": "fixed NaN-to-zero; float32; no scaling; no temporal imputation"}
    if "sequence_boundaries" in train_identity:
        plan["privacy_scope"] = ("One contiguous event within one machine; event length = ceil(event_fraction * "
                                 "total pooled training rows); per run; conservative compact-index block bound")
    if "poisson" in methods:
        progress("Accounting Poisson sampling")
        q = config.batch_size / geometry.candidates
        logs = binom.logpmf(np.arange(geometry.affected + 1), geometry.affected, q)
        rdp = gaussian_moment_rdp(logs, config.orders, config.noise_multiplier)
        acc = RDPAccountant(config.orders, rdp, config.delta)
        plan["methods"]["poisson"] = {
            "accounting": "manuscript conditional Gaussian RDP bound",
            "sampling_rate": q, "expected_batch_size": config.batch_size,
            "orders": list(config.orders), "rdp_per_step": rdp.tolist(),
            "budgets": [budget_entry(acc, e, config) for e in config.epsilons]}
    if "pss" in methods:
        progress("Accounting PSS: all affected-block positions, without b0 shortcut")
        pattern_cache, candidates = {}, []
        grid = lambda_grid(geometry.candidates, config.batch_size, config.lambda_grid_step)
        for index, lam in enumerate(grid):
            try:
                part = pss_partition(geometry.candidates, config.batch_size, lam, config.pss_mode)
            except ValueError:
                continue
            if part.sizes not in pattern_cache:
                pattern_cache[part.sizes] = affected_patterns(part.sizes, geometry.affected)
            patterns = pattern_cache[part.sizes]
            rdp = pss_rdp(part, geometry.affected, config.orders, config.noise_multiplier, patterns)
            acc = RDPAccountant(config.orders, rdp, config.delta)
            candidates.append({"lambda": lam, "window_count": len(part.sizes),
                               "min_window_size": min(part.sizes), "max_window_size": max(part.sizes),
                               "activation_probability": part.activation,
                               "expected_batch_size": part.expected_batch_size,
                               "patterns_checked": len(patterns), "rdp_per_step": rdp.tolist(),
                               "budgets": [budget_entry(acc, e, config) for e in config.epsilons]})
            if (index + 1) % 10 == 0:
                progress(f"PSS lambda candidates: {index + 1}/{len(grid)}")
        if not candidates:
            raise ValueError("No feasible PSS partition")
        selected = []
        for i, target in enumerate(config.epsilons):

            best = max(candidates, key=lambda row: (row["budgets"][i]["steps"], row["lambda"]))
            selected.append({**{k: v for k, v in best.items() if k != "budgets"}, **best["budgets"][i]})
        plan["methods"]["pss"] = {"accounting": "maximum conditional Gaussian RDP moment over all block locations",
                                    "sampling_mode": config.pss_mode, "orders": list(config.orders),
                                    "lambda_candidates": candidates, "budgets": selected}
    if "structured" in methods:
        from .structured_privacy.accountant import PLDAccountant
        progress("Accounting Structured-WR: Theorem E.1 PLD, actual noise std/C="
                 + str(config.noise_multiplier))
        params = structured_parameters(geometry, config)
        acc = PLDAccountant(params, config.delta, privacy_loss_interval=config.pld_interval,
                            tail_mass=config.tail_mass)
        budgets = []
        for e in config.epsilons:
            progress(f"Structured budget epsilon={e:g}")
            budgets.append(budget_entry(acc, e, config))
        plan["methods"]["structured"] = {
            "accounting": params.bound_type, "composition": "PLD connect-the-dots + FFT",
            "expected_batch_size": config.batch_size, "hit_probability": params.hit_probability,
            "noise_multiplier_std_over_C": config.noise_multiplier, "pld_interval": config.pld_interval,
            "tail_mass": config.tail_mass, "dp_accounting_version": acc.dp_accounting_version,
            "budgets": budgets}
    for name, method in plan["methods"].items():
        for entry in method["budgets"]:
            if not entry["budget_exhausted"] and not config.allow_step_cap:
                raise ValueError(f"{name}, epsilon={entry['target_epsilon']}: max_steps reached before privacy "
                                 "boundary. Increase max_steps or explicitly set allow_step_cap=true.")
    return plan
