from __future__ import annotations

from dataclasses import dataclass
import math
import numbers
from typing import Iterable, List, Sequence, Union

import numpy as np
import scipy
from scipy import stats
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import norm

try:
    from dp_accounting.pld import common
    from dp_accounting.pld.privacy_loss_mechanism import (
        AdditiveNoisePrivacyLoss,
        AdjacencyType,
        ConnectDotsBounds,
        GaussianPrivacyLoss,
        MixtureGaussianPrivacyLoss,
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
    window_length: int
    sample_stride: int
    event_length: int
    batch_size: int
    noise_multiplier: float
    num_candidate_windows: int
    max_affected_windows: int
    hit_probability: float
    weights: np.ndarray
    means_p: np.ndarray
    means_q: np.ndarray

    @property
    def bound_type(self) -> str:
        if self.batch_size == 1:
            return "tight Theorem 4.2 WR profile"
        return "sound Theorem E.1 WR upper bound; potentially loose for K>1"


def build_wr_mixture_parameters(
    sequence_length: int,
    window_length: int,
    event_length: int,
    batch_size: int,
    noise_multiplier: float,
    sample_stride: int = 1,
) -> WRMixtureParameters:

    if window_length < 1 or batch_size < 1 or sample_stride < 1:
        raise ValueError("Window length, stride and batch size must be positive")
    if sequence_length < window_length or not 1 <= event_length <= sequence_length:
        raise ValueError("Invalid sequence or event length")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be finite and positive")
    candidates = (sequence_length - window_length) // sample_stride + 1
    affected = min(candidates, (window_length + event_length - 2) // sample_stride + 1)
    hit_probability = affected / candidates
    indices = np.arange(batch_size + 1, dtype=np.float64)
    log_weights = stats.binom.logpmf(indices, batch_size, hit_probability)
    weights = np.exp(log_weights - logsumexp(log_weights))
    return WRMixtureParameters(
        sequence_length=sequence_length,
        window_length=window_length,
        sample_stride=sample_stride,
        event_length=event_length,
        batch_size=batch_size,
        noise_multiplier=noise_multiplier,
        num_candidate_windows=candidates,
        max_affected_windows=affected,
        hit_probability=hit_probability,
        weights=weights,
        means_p=-2.0 * indices,
        means_q=2.0 * indices,
    )


def log_gaussian_density(
    values: np.ndarray | float, mean: float, std: float
) -> np.ndarray:

    values = np.asarray(values, dtype=np.float64)
    return -0.5 * ((values - mean) / std) ** 2 - math.log(std) - 0.5 * math.log(
        2 * math.pi
    )


def log_mixture_density(
    values: np.ndarray | float,
    weights: np.ndarray,
    means: np.ndarray,
    std: float,
) -> np.ndarray:

    values = np.asarray(values, dtype=np.float64)
    positive = weights > 0
    terms = (
        np.log(weights[positive])[:, None]
        + log_gaussian_density(values.ravel()[None, :], means[positive, None], std)
    )
    return logsumexp(terms, axis=0).reshape(values.shape)


def mixture_cdf(
    values: np.ndarray | float,
    weights: np.ndarray,
    means: np.ndarray,
    std: float,
) -> np.ndarray:

    values = np.asarray(values, dtype=np.float64)
    return np.sum(
        weights[:, None]
        * norm.cdf((values.ravel()[None, :] - means[:, None]) / std),
        axis=0,
    ).reshape(values.shape)


def privacy_profile_at_epsilon(
    epsilon: float,
    weights_p: np.ndarray,
    means_p: np.ndarray,
    weights_q: np.ndarray,
    means_q: np.ndarray,
    std: float,
    *,
    tail_std_multiplier: float = 12.0,
) -> float:

    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    radius = max(float(np.max(np.abs(means_p))), float(np.max(np.abs(means_q))))
    radius += tail_std_multiplier * std

    def integrand(value: float) -> float:
        log_p = float(log_mixture_density(value, weights_p, means_p, std))
        log_q = float(log_mixture_density(value, weights_q, means_q, std))
        if log_p <= epsilon + log_q:
            return 0.0
        return math.exp(log_p) * (1.0 - math.exp(epsilon + log_q - log_p))

    integral, error = quad(
        integrand,
        -radius,
        radius,
        epsabs=1e-12,
        epsrel=1e-10,
        limit=300,
    )
    p_tail = float(mixture_cdf(-radius, weights_p, means_p, std))
    p_tail += 1.0 - float(mixture_cdf(radius, weights_p, means_p, std))
    return min(1.0, max(0.0, integral + error + p_tail))


def dominating_pair(
    parameters: WRMixtureParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    if parameters.batch_size == 1:
        r = parameters.hit_probability
        return (
            np.array([1.0 - r, r], dtype=np.float64),
            np.array([0.0, 2.0], dtype=np.float64),
            np.array([1.0], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
        )
    return (
        parameters.weights,
        parameters.means_p,
        parameters.weights,
        parameters.means_q,
    )


class SwitchingPrivacyLoss(AdditiveNoisePrivacyLoss):


    def __init__(self,
                 epsilon_threshold: float,
                 below_threshold_pl: AdditiveNoisePrivacyLoss,
                 above_threshold_pl: AdditiveNoisePrivacyLoss):

        self.epsilon_threshold = epsilon_threshold
        self.below_threshold_pl = below_threshold_pl
        self.above_threshold_pl = above_threshold_pl

        if below_threshold_pl.discrete_noise != above_threshold_pl.discrete_noise:
            raise ValueError('PLs must be both discrete or both continuous.')

        self.discrete_noise = below_threshold_pl.discrete_noise

        if self.discrete_noise:
            raise NotImplementedError('Only continuous PLs supported currently.')

        if not np.isclose(
                below_threshold_pl.get_delta_for_epsilon(epsilon_threshold),
                above_threshold_pl.get_delta_for_epsilon(epsilon_threshold)):
            raise ValueError('Tradeoff functions must intersect at epsilon_threshold.')

    def mu_upper_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def mu_lower_log_cdf(self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def get_delta_for_epsilon(
            self, epsilon: Union[float, List[float]]) -> Union[float, List[float]]:

        is_scalar = isinstance(epsilon, numbers.Number)
        epsilons = np.array([epsilon]) if is_scalar else np.asarray(epsilon)
        deltas = np.zeros_like(epsilons, dtype=float)

        below_threshold_mask = (epsilons < self.epsilon_threshold)
        above_threshold_mask = ~below_threshold_mask

        if below_threshold_mask.sum() > 0:
            deltas[below_threshold_mask] = self.below_threshold_pl.get_delta_for_epsilon(
                epsilons[below_threshold_mask])

        if above_threshold_mask.sum() > 0:
            deltas[above_threshold_mask] = self.above_threshold_pl.get_delta_for_epsilon(
                epsilons[above_threshold_mask])

        return float(deltas[0]) if is_scalar else deltas

    def privacy_loss_tail(self) -> TailPrivacyLossDistribution:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def connect_dots_bounds(self) -> ConnectDotsBounds:
        below_threshold_bounds = self.below_threshold_pl.connect_dots_bounds()
        above_threshold_bounds = self.above_threshold_pl.connect_dots_bounds()

        epsilon_upper = max(below_threshold_bounds.epsilon_upper,
                            above_threshold_bounds.epsilon_upper)

        epsilon_lower = min(below_threshold_bounds.epsilon_lower,
                            above_threshold_bounds.epsilon_lower)

        return ConnectDotsBounds(epsilon_upper=epsilon_upper,
                                 epsilon_lower=epsilon_lower)

    def privacy_loss(self, x: float) -> float:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def privacy_loss_without_subsampling(self, x: float) -> float:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss(self, privacy_loss: float) -> float:
        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def inverse_privacy_loss_without_subsampling(self,
                                                 privacy_loss: float) -> float:

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def noise_cdf(self, x: Union[float,
                  Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    def noise_log_cdf(
            self, x: Union[float, Iterable[float]]) -> Union[float, np.ndarray]:

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')

    @classmethod
    def from_privacy_guarantee(
            cls,
            privacy_parameters: common.DifferentialPrivacyParameters,
            sensitivity: float = 1,
            pessimistic_estimate: bool = True,
            sampling_prob: float = 1.0,
            adjacency_type: AdjacencyType = AdjacencyType.REMOVE) -> 'AdditiveNoisePrivacyLoss':

        raise NotImplementedError(
            'SwitchingPL is currently only meant for use with connect_the_dots.')


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
    ) -> 'MixtureGaussianPrivacyLoss':
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


    if parameters.hit_probability == 1.0:
        separation = 2.0 if parameters.batch_size == 1 else 4.0 * parameters.batch_size
        return GaussianPrivacyLoss(
            standard_deviation=parameters.noise_multiplier,
            sensitivity=separation,
            pessimistic_estimate=True,
            log_mass_truncation_bound=math.log(tail_mass),
        )

    sensitivities = 2.0 * np.arange(parameters.batch_size + 1, dtype=np.float64)
    sampling_probs = parameters.weights

    if parameters.batch_size == 1:


        return SwitchingPrivacyLoss(
            epsilon_threshold=0.0,
            below_threshold_pl=MixtureGaussianPrivacyLoss(
                parameters.noise_multiplier,
                sensitivities,
                sampling_probs,
                pessimistic_estimate=True,
                log_mass_truncation_bound=math.log(tail_mass),
                adjacency_type=AdjacencyType.ADD,
            ),
            above_threshold_pl=MixtureGaussianPrivacyLoss(
                parameters.noise_multiplier,
                sensitivities,
                sampling_probs,
                pessimistic_estimate=True,
                log_mass_truncation_bound=math.log(tail_mass),
                adjacency_type=AdjacencyType.REMOVE,
            ),
        )

    return DoubleMixtureGaussianPrivacyLoss(
        standard_deviation=parameters.noise_multiplier,
        sensitivities_upper=sensitivities,
        sensitivities_lower=sensitivities,
        sampling_probs_upper=sampling_probs,
        sampling_probs_lower=sampling_probs,
        pessimistic_estimate=True,
        log_mass_truncation_bound=math.log(tail_mass),
    )
