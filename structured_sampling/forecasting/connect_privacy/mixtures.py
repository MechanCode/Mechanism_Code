from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Iterable, Sequence, Union

import numpy as np
import scipy
from scipy import stats
from scipy.special import logsumexp

try:
    from dp_accounting.pld import common
    from dp_accounting.pld.privacy_loss_mechanism import (
        AdditiveNoisePrivacyLoss,
        AdjacencyType,
        ConnectDotsBounds,
        TailPrivacyLossDistribution,
    )
except ImportError as exc:
    raise ImportError(
        "Connect-the-dots accounting requires dp-accounting==0.4.4. "
        "Install it with: pip install dp-accounting==0.4.4"
    ) from exc


@dataclass(frozen=True)
class WRMixtureParameters:
    sequence_length: int
    context_len: int
    forecast_len: int
    sample_stride: int
    event_length: int
    batch_size: int
    noise_multiplier: float
    num_candidate_windows: int
    max_affected_windows: int
    hit_probability: float
    weights: np.ndarray

    @property
    def bound_type(self) -> str:
        return "sound Theorem E.1 WR upper bound; potentially loose for K>1"


def build_wr_mixture_parameters(
    sequence_length: int,
    context_len: int,
    forecast_len: int,
    event_length: int,
    batch_size: int,
    noise_multiplier: float,
    sample_stride: int = 1,
) -> WRMixtureParameters:

    if context_len < 1 or forecast_len < 1 or batch_size < 2:
        raise ValueError("context_len/forecast_len must be positive; this launcher uses K > 1")
    if sequence_length < forecast_len:
        raise ValueError("sequence_length must be at least forecast_len")
    if not 1 <= event_length <= sequence_length:
        raise ValueError("event_length must be in [1, sequence_length]")
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if sample_stride < 1:
        raise ValueError("sample_stride must be positive")

    candidates = (sequence_length - forecast_len) // sample_stride + 1
    affected_span = context_len + forecast_len + event_length - 1
    affected = min(candidates, math.ceil(affected_span / sample_stride))
    hit_probability = affected / candidates

    indices = np.arange(batch_size + 1, dtype=np.float64)
    if hit_probability == 1.0:
        weights = np.zeros(batch_size + 1, dtype=np.float64)
        weights[-1] = 1.0
    else:
        log_coefficients = np.array(
            [
                math.lgamma(batch_size + 1)
                - math.lgamma(int(index) + 1)
                - math.lgamma(batch_size - int(index) + 1)
                for index in indices
            ],
            dtype=np.float64,
        )
        log_weights = (
            log_coefficients
            + indices * math.log(hit_probability)
            + (batch_size - indices) * math.log1p(-hit_probability)
        )
        weights = np.exp(log_weights - logsumexp(log_weights))

    return WRMixtureParameters(
        sequence_length=sequence_length,
        context_len=context_len,
        forecast_len=forecast_len,
        sample_stride=sample_stride,
        event_length=event_length,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        num_candidate_windows=candidates,
        max_affected_windows=affected,
        hit_probability=hit_probability,
        weights=weights,
    )


class DoubleMixtureGaussianPrivacyLoss(AdditiveNoisePrivacyLoss):


    def __init__(
      self,
      standard_deviation: float,
      sensitivities_upper: Sequence[float],
      sensitivities_lower: Sequence[float],
      sampling_probs_upper: Sequence[float],
      sampling_probs_lower: Sequence[float],
      pessimistic_estimate: bool = True,
      log_mass_truncation_bound: float = -50,
    ) -> None:

        if standard_deviation <= 0:
            raise ValueError(
                'Standard deviation is not a positive real number: '
                f'{standard_deviation}'
            )

        if log_mass_truncation_bound > 0:
            raise ValueError(
                'Log mass truncation bound is not a non-positive real '
                f'number: {log_mass_truncation_bound}'
            )

        if ((len(sampling_probs_upper) != len(sensitivities_upper))
           or (len(sampling_probs_lower) != len(sensitivities_lower))):

            raise ValueError(
                'sensitivities and sampling_probs must have the same length'
            )

        non_zero_indices_upper = np.asarray(sampling_probs_upper) != 0.0
        sensitivities_upper = np.asarray(sensitivities_upper)[non_zero_indices_upper]
        sampling_probs_upper = np.asarray(sampling_probs_upper)[non_zero_indices_upper]
        non_zero_indices_lower = np.asarray(sampling_probs_lower) != 0.0
        sensitivities_lower = np.asarray(sensitivities_lower)[non_zero_indices_lower]
        sampling_probs_lower = np.asarray(sampling_probs_lower)[non_zero_indices_lower]

        if np.any(sensitivities_upper < 0) or np.any(sensitivities_lower < 0):
            raise ValueError(
                'Sensitivities contain a negative number.'
            )

        if (sensitivities_upper.max() == 0.0) and (sensitivities_lower.max() == 0.0):
            raise ValueError('Must have at least one positive sensitivity.')

        if not (math.isclose(sum(sampling_probs_upper), 1)
                and
                math.isclose(sum(sampling_probs_lower), 1)):
            raise ValueError(
                'Probabilities do not add up to 1'
            )

        if (np.any((sampling_probs_upper <= 0) | (sampling_probs_upper > 1))
           or np.any((sampling_probs_lower <= 0) | (sampling_probs_lower > 1))):

            raise ValueError(
                'Sampling probabilities are in (0,1]'
            )

        self.discrete_noise = False

        self.sampling_probs_upper = sampling_probs_upper
        self.sensitivities_upper = sensitivities_upper
        self.sampling_probs_lower = sampling_probs_lower
        self.sensitivities_lower = sensitivities_lower
        self._standard_deviation = standard_deviation
        self._variance = standard_deviation**2
        self._pessimistic_estimate = pessimistic_estimate
        self._log_mass_truncation_bound = log_mass_truncation_bound


        self._log_sampling_probs_upper = np.log(self.sampling_probs_upper)
        self._pos_sampling_probs_upper = self.sampling_probs_upper[self.sensitivities_upper > 0.0]
        self._sampling_prob_upper = np.clip(self._pos_sampling_probs_upper.sum(), 0, 1)
        self._max_sens_upper = self.sensitivities_upper[self.sampling_probs_upper > 0].max()

        self._log_sampling_probs_lower = np.log(self.sampling_probs_lower)
        self._pos_sampling_probs_lower = self.sampling_probs_lower[self.sensitivities_lower > 0.0]
        self._sampling_prob_lower = np.clip(self._pos_sampling_probs_lower.sum(), 0, 1)
        self._max_sens_lower = self.sensitivities_lower[self.sampling_probs_lower > 0].max()

        self._gaussian_random_variable = stats.norm(scale=standard_deviation)

    def mu_upper_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:

        points_per_sens = np.add.outer(np.atleast_1d(x), self.sensitivities_upper)
        output = (self.noise_cdf(points_per_sens) * self.sampling_probs_upper).sum(axis=1)

        if isinstance(x, numbers.Number):
            return output[0]
        else:
            return output

    def mu_lower_log_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:

        points_per_sens = np.add.outer(np.atleast_1d(x), -self.sensitivities_lower)
        logcdf_per_sens = self.noise_log_cdf(points_per_sens)

        output = scipy.special.logsumexp(
            logcdf_per_sens, axis=1, b=self.sampling_probs_lower
        )
        if isinstance(x, numbers.Number):
            return output[0]
        else:
            return output

    def get_delta_for_epsilon(
      self, epsilon: Union[float, Sequence[float]]
    ) -> Union[float, list[float]]:

        epsilons = np.atleast_1d(epsilon)
        if not np.all(epsilons[1:] >= epsilons[:-1]):
            raise ValueError(f'Epsilon values must be non-decreasing: {epsilons}')

        deltas = np.zeros_like(epsilons, dtype=float)


        if (self._sampling_prob_upper == 0.0) and (self._sampling_prob_lower != 1.0):
            inverse_indices = epsilons < -np.log1p(-self._sampling_prob_lower)


        elif (self._sampling_prob_lower == 0.0) and (self._sampling_prob_upper != 1.0):
            inverse_indices = epsilons > np.log1p(-self._sampling_prob_upper)
            other_indices = np.logical_not(inverse_indices)
            deltas[other_indices] = -np.expm1(epsilons[other_indices])

        else:
            inverse_indices = np.full_like(epsilons, True, dtype=bool)

        x_cutoffs = self.inverse_privacy_losses(epsilons[inverse_indices])

        deltas[inverse_indices] = self.mu_upper_cdf(x_cutoffs) - np.exp(
            epsilons[inverse_indices] + self.mu_lower_log_cdf(x_cutoffs)
        )


        deltas = np.clip(deltas, 0, 1)
        if isinstance(epsilon, numbers.Number):
            return float(deltas)
        else:


            for i in reversed(range(deltas.shape[0] - 1)):
                deltas[i] = max(deltas[i], deltas[i + 1])
        return deltas

    def privacy_loss_tail(
        self, precision: float = 1e-4
    ) -> TailPrivacyLossDistribution:

        tail_mass = 0.5 * np.exp(self._log_mass_truncation_bound)
        z_value = self._gaussian_random_variable.ppf(tail_mass)
        upper_x_truncation = -z_value

        if self._sampling_prob_upper == 0.0:
            lower_x_truncation = z_value
        else:
            lower_x_truncation = common.inverse_monotone_function(
                self.mu_upper_cdf,
                tail_mass,
                common.BinarySearchParameters(
                    z_value - self._max_sens_upper,
                    z_value,
                    tolerance=precision
                ),
                increasing=True,
            )
        if self._pessimistic_estimate:
            tail_probability_mass_function = {
                math.inf: self.mu_upper_cdf(lower_x_truncation),
                self.privacy_loss(upper_x_truncation): 1 - self.mu_upper_cdf(
                    upper_x_truncation
                ),
            }
        else:
            tail_probability_mass_function = {
                self.privacy_loss(lower_x_truncation): self.mu_upper_cdf(
                    lower_x_truncation
                ),
            }

        return TailPrivacyLossDistribution(
            lower_x_truncation, upper_x_truncation, tail_probability_mass_function
        )

    def connect_dots_bounds(self) -> ConnectDotsBounds:

        tail_pld = self.privacy_loss_tail()

        return ConnectDotsBounds(
            epsilon_upper=self.privacy_loss(tail_pld.lower_x_truncation),
            epsilon_lower=self.privacy_loss(tail_pld.upper_x_truncation),
        )

    def privacy_loss(self, x: float) -> float:


        p_upper = logsumexp(stats.norm.logpdf(x, loc=-1 * self.sensitivities_upper,
                                              scale=self._standard_deviation),
                            b=self.sampling_probs_upper)

        p_lower = logsumexp(stats.norm.logpdf(x, loc=self.sensitivities_lower,
                                              scale=self._standard_deviation),
                            b=self.sampling_probs_lower)

        return p_upper - p_lower

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError(
            'DoubleMixtureGaussianPrivacyLoss uses multiple sensitivities, so '
            'privacy loss without subsampling is ill-defined. Use '
            'privacy_loss_for_single_gaussian instead.'
        )

    def inverse_privacy_loss_without_subsampling(
      self, privacy_loss: float
    ) -> float:
        raise NotImplementedError(
            'MixtureGaussianPrivacyLoss uses multiple sensitivities, so '
            'inverse_privacy_loss_without_subsampling is ill-defined. Use '
            'inverse_privacy_loss_for_single_gaussian instead.'
        )

    def inverse_privacy_loss(
        self, privacy_loss: float, precision: float = 1e-6
    ) -> float:

        return float(
            self.inverse_privacy_losses(np.atleast_1d(privacy_loss), precision)[0]
        )

    def inverse_privacy_losses(
        self,
        privacy_losses: np.ndarray,
        precision: float = 1e-6,
    ) -> np.ndarray:

        if not (np.diff(privacy_losses) >= 0).all():
            raise ValueError(
                f'Expected non-decreasing privacy_losses, got: {privacy_losses}.'
            )
        if len(privacy_losses) == 0:
            return np.ndarray([])


        min_pl = privacy_losses[0]
        max_pl = privacy_losses[-1]


        if (self._sampling_prob_upper == 0.0) and (self._sampling_prob_lower != 1.0):
            log_1m_prob = (
                math.log1p(-self._sampling_prob_lower)
            )
            if max_pl > -log_1m_prob:
                raise ValueError(
                    f'max of privacy_losses ({max_pl}) is larger than '
                    f'-log(1 - sampling_prob)={-log_1m_prob}.'
                )
            finite_indices = np.logical_not(np.isclose(privacy_losses, -log_1m_prob))
            max_pl = np.max(privacy_losses[finite_indices])


        elif (self._sampling_prob_lower == 0.0) and (self._sampling_prob_upper != 1.0):
            log_1m_prob = (math.log1p(-self._sampling_prob_upper))

            if min_pl <= log_1m_prob:
                raise ValueError(
                    f'min of privacy_losses ({min_pl}) is smaller than '
                    f'log(1 - sampling_prob)={log_1m_prob}'
                )
            finite_indices = np.logical_not(np.isclose(privacy_losses, log_1m_prob))
            min_pl = np.min(privacy_losses[finite_indices])

        else:
            finite_indices = np.full_like(privacy_losses, True, dtype=bool)


        left_bound = -1
        while True:
            loss = self.privacy_loss(left_bound)
            if not (loss < max_pl):
                break
            left_bound *= 2

        right_bound = 1
        while self.privacy_loss(right_bound) > min_pl:
            right_bound *= 2

        bounds = (left_bound, right_bound)


        if (self._sampling_prob_upper == 0.0) and (self._sampling_prob_lower != 1.0):
            output = np.full_like(privacy_losses, -np.inf)

        else:
            output = np.full_like(privacy_losses, np.inf)

        output[finite_indices] = self._inverse_privacy_losses_with_range(
            privacy_losses[finite_indices], bounds, precision
        )

        return output

    def _inverse_privacy_losses_with_range(
      self,
      privacy_losses: np.ndarray,
      bounds: tuple[float, float],
      precision: float = 1e-6,
    ) -> Iterable[float]:

        if len(privacy_losses) == 0:
            return []
        if bounds[1] - bounds[0] <= precision:
            return np.repeat(
                np.floor(bounds[1] / precision) * precision, len(privacy_losses)
            )

        mid = (bounds[0] + bounds[1]) / 2
        pl_split = self.privacy_loss(mid)
        lower_indices = privacy_losses < pl_split
        higher_indices = privacy_losses >= pl_split
        output = np.zeros_like(privacy_losses)
        output[lower_indices] = self._inverse_privacy_losses_with_range(
            privacy_losses[lower_indices], (mid, bounds[1]), precision
        )
        output[higher_indices] = self._inverse_privacy_losses_with_range(
            privacy_losses[higher_indices], (bounds[0], mid), precision
        )
        return output

    def noise_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:

        return self._gaussian_random_variable.cdf(x)

    def noise_log_cdf(
        self, x: Union[float, Iterable[float]]
    ) -> Union[float, np.ndarray]:

        return self._gaussian_random_variable.logcdf(x)

    @classmethod
    def from_privacy_guarantee(
        cls,
        privacy_parameters: common.DifferentialPrivacyParameters,
        sensitivity: float = 1,
        pessimistic_estimate: bool = True,
        sampling_prob: float = 1.0,
        adjacency_type: AdjacencyType = AdjacencyType.REMOVE,
    ) -> 'DoubleMixtureGaussianPrivacyLoss':
        raise NotImplementedError(
            'MixtureGaussianPrivacy loss cannot be uniquely '
            'instantiated from privacy parameters.'
        )

def create_wr_privacy_loss(
    parameters: WRMixtureParameters,
    *,
    tail_mass: float = 1e-15,
) -> AdditiveNoisePrivacyLoss:

    if not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass must be in (0, 1)")

    sensitivities = 2.0 * np.arange(parameters.batch_size + 1, dtype=np.float64)
    sampling_probs = parameters.weights

    return DoubleMixtureGaussianPrivacyLoss(
        standard_deviation=parameters.noise_multiplier,
        sensitivities_upper=sensitivities,
        sensitivities_lower=sensitivities,
        sampling_probs_upper=sampling_probs,
        sampling_probs_lower=sampling_probs,
        pessimistic_estimate=True,
        log_mass_truncation_bound=math.log(tail_mass),
    )
