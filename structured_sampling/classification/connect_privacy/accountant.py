from __future__ import annotations

from importlib import metadata
import warnings

try:
    from dp_accounting.pld.privacy_loss_distribution import (
        PrivacyLossDistribution,
        _create_pld_pmf_from_additive_noise,
    )
except ImportError as exc:
    raise ImportError(
        "This connect-the-dots implementation requires dp-accounting==0.4.4. "
        "Install it with: pip install dp-accounting==0.4.4"
    ) from exc

from .mixtures import WRMixtureParameters, create_wr_privacy_loss


def _dp_accounting_version() -> str:
    try:
        return metadata.version("dp-accounting")
    except metadata.PackageNotFoundError:
        return "unknown"


def build_connect_dots_pld(
    parameters: WRMixtureParameters,
    *,
    privacy_loss_interval: float = 1e-3,
    tail_mass: float = 1e-15,
) -> PrivacyLossDistribution:

    if privacy_loss_interval <= 0:
        raise ValueError("privacy_loss_interval must be positive")
    if not 0.0 < tail_mass < 1.0:
        raise ValueError("tail_mass must be in (0, 1)")

    privacy_loss = create_wr_privacy_loss(parameters, tail_mass=tail_mass)
    pld_pmf = _create_pld_pmf_from_additive_noise(
        privacy_loss,
        value_discretization_interval=privacy_loss_interval,
        use_connect_dots=True,
    )
    return PrivacyLossDistribution(pld_pmf)


def _self_compose(
    pld: PrivacyLossDistribution,
    steps: int,
    *,
    tail_mass: float,
) -> PrivacyLossDistribution:

    if steps < 1:
        raise ValueError("steps must be positive")
    if steps == 1:
        return pld

    return pld.self_compose(steps, tail_mass_truncation=tail_mass)


def _single_step_infinity_mass(pld: PrivacyLossDistribution) -> float | None:

    pmf = getattr(pld, "_pmf_remove", None)
    value = getattr(pmf, "_infinity_mass", None)
    return None if value is None else float(value)


class PLDAccountant:


    def __init__(
        self,
        parameters: WRMixtureParameters,
        target_delta: float,
        *,
        privacy_loss_interval: float = 1e-3,
        tail_mass: float = 1e-15,
    ) -> None:
        if not 0.0 < target_delta < 1.0:
            raise ValueError("target_delta must be in (0, 1)")
        if privacy_loss_interval <= 0:
            raise ValueError("privacy_loss_interval must be positive")
        if not 0.0 < tail_mass < 1.0:
            raise ValueError("tail_mass must be in (0, 1)")
        self.parameters = parameters
        self.target_delta = target_delta
        self.privacy_loss_interval = privacy_loss_interval
        self.tail_mass = tail_mass
        self.dp_accounting_version = _dp_accounting_version()

        if self.dp_accounting_version not in {"0.4.4", "unknown"}:
            warnings.warn(
                "This implementation was written against dp-accounting==0.4.4; "
                f"detected {self.dp_accounting_version}. The private helper API "
                "may differ across versions.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.single_step_pld = build_connect_dots_pld(
            parameters,
            privacy_loss_interval=privacy_loss_interval,
            tail_mass=tail_mass,
        )
        self._composed_pld_cache: dict[int, PrivacyLossDistribution] = {
            1: self.single_step_pld
        }
        self._epsilon_cache: dict[int, float] = {0: 0.0}

    def composed_pld(self, optimizer_steps: int) -> PrivacyLossDistribution:

        if optimizer_steps < 1:
            raise ValueError("optimizer_steps must be positive")
        if optimizer_steps not in self._composed_pld_cache:
            self._composed_pld_cache[optimizer_steps] = _self_compose(
                self.single_step_pld,
                optimizer_steps,
                tail_mass=self.tail_mass,
            )
        return self._composed_pld_cache[optimizer_steps]

    def epsilon(self, optimizer_steps: int) -> float:

        if optimizer_steps < 0:
            raise ValueError("optimizer_steps cannot be negative")
        if optimizer_steps == 0:
            return 0.0
        if optimizer_steps not in self._epsilon_cache:
            self._epsilon_cache[optimizer_steps] = float(
                self.composed_pld(optimizer_steps).get_epsilon_for_delta(
                    self.target_delta
                )
            )
        return self._epsilon_cache[optimizer_steps]


    def maximum_steps(
        self, target_epsilon: float, *, maximum_steps: int = 100_000
    ) -> tuple[int, bool]:

        if target_epsilon <= 0 or maximum_steps < 1:
            raise ValueError("target_epsilon and maximum_steps must be positive")
        if self.epsilon(1) > target_epsilon:
            return 0, True

        lower, upper = 1, 2
        while upper <= maximum_steps and self.epsilon(upper) <= target_epsilon:
            lower, upper = upper, upper * 2

        if upper > maximum_steps:
            if self.epsilon(maximum_steps) <= target_epsilon:
                return maximum_steps, False
            upper = maximum_steps

        while lower + 1 < upper:
            midpoint = (lower + upper) // 2
            if self.epsilon(midpoint) <= target_epsilon:
                lower = midpoint
            else:
                upper = midpoint
        return lower, True

    def report(self, optimizer_steps: int, target_epsilon: float) -> dict[str, object]:

        params = self.parameters
        report = {
            "method": "N=1 Structured-WR",
            "task": "classification",
            "sequence_length": params.sequence_length,
            "sample_stride": params.sample_stride,
            "window_length": params.window_length,
            "event_length": params.event_length,
            "candidate_windows": params.num_candidate_windows,
            "maximum_affected_windows": params.max_affected_windows,
            "single_draw_hit_probability": params.hit_probability,
            "batch_size": params.batch_size,
            "sampling_scheme": "uniform_with_replacement",
            "clipping_unit": "complete_classification_window",
            "noise_multiplier": params.noise_multiplier,
            "optimizer_steps": optimizer_steps,
            "target_epsilon": target_epsilon,
            "target_delta": self.target_delta,
            "accounted_epsilon": self.epsilon(optimizer_steps),
            "accounting_bound_type": params.bound_type,
            "accounting_method": "PLD connect-the-dots + FFT composition",
            "dp_accounting_version": self.dp_accounting_version,
            "use_connect_dots": True,
            "pld_grid_interval": self.privacy_loss_interval,
            "tail_mass_truncation": self.tail_mass,
            "single_step_infinity_mass": _single_step_infinity_mass(
                self.single_step_pld
            ),
        }
        return report
