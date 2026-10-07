from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

METHODS = ("poisson", "structured", "pss")


@dataclass(frozen=True)
class Config:
    segment_length: int = 100
    train_stride: int = 10
    batch_size: int = 50
    clipping_norm: float = 1.0

    noise_multiplier: float = 10.0
    event_fraction: float = 0.005
    delta: float = 1e-5
    orders: tuple[int, ...] = tuple(range(2, 65))
    epsilons: tuple[float, ...] = (1., 2., 3., 4., 5., 6.)
    seed_start: int = 42
    num_seeds: int = 30
    learning_rate: float = 1e-3
    optimizer: str = "sgd"
    momentum: float = 0.0
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    loss_scale: float = 1.0
    model_kind: str = "conv"
    model_width: int = 64
    model_rank: int = 4
    evaluation_split: str = "full"
    development_fraction: float = 0.5


    evaluation_gap: int | None = None


    pss_mode: str = "forecasting"
    lambda_grid_step: float = 0.01
    pld_interval: float = 0.001
    tail_mass: float = 1e-15
    max_steps: int = 100_000
    allow_step_cap: bool = False
    microbatch_size: int = 10
    eval_batch_size: int = 128
    checkpoint_every: int = 100
    deterministic: bool = True
    f1_threshold: float | None = None

    def validate(self):
        for key in ("allow_step_cap", "deterministic"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be a JSON boolean")
        for key in ("segment_length", "train_stride", "batch_size", "num_seeds",
                    "max_steps", "microbatch_size", "eval_batch_size", "checkpoint_every",
                    "model_width", "model_rank"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.model_kind == "conv" and self.segment_length % 4:
            raise ValueError("segment_length must be divisible by 4 for this autoencoder")
        if self.model_kind not in ("conv", "feature_ae", "temporal_ae", "feature_mean", "shared_ar",
                                   "feature_linear", "hybrid_ae"):
            raise ValueError("Unknown model_kind")
        if self.optimizer not in ("sgd", "adam"):
            raise ValueError("Unknown optimizer")
        for key in ("momentum", "adam_beta1", "adam_beta2"):
            if not math.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) < 1:
                raise ValueError(f"{key} must be finite in [0,1)")
        if self.optimizer == "adam" and self.momentum != 0:
            raise ValueError("SGD momentum must be zero for Adam")
        if self.model_width % 4:
            raise ValueError("model_width must be divisible by four")
        if self.evaluation_split not in ("full", "development", "heldout"):
            raise ValueError("evaluation_split must be full, development, or heldout")
        if not 0 < self.development_fraction < 1:
            raise ValueError("development_fraction must be in (0,1)")
        if self.evaluation_gap is not None and (type(self.evaluation_gap) is not int
                                               or self.evaluation_gap < self.segment_length):
            raise ValueError("evaluation_gap must be an integer at least segment_length")
        if type(self.seed_start) is not int or self.seed_start < 0:
            raise ValueError("seed_start must be a nonnegative integer")
        for key in ("clipping_norm", "noise_multiplier", "learning_rate", "loss_scale", "pld_interval", "adam_epsilon"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("delta", "tail_mass"):
            if not 0 < getattr(self, key) < 1:
                raise ValueError(f"{key} must be in (0,1)")
        if not 0 < self.event_fraction <= 1 or not 0 < self.lambda_grid_step <= 1:
            raise ValueError("event_fraction and lambda_grid_step must be in (0,1]")
        if not self.orders or any(type(a) is not int or a < 2 for a in self.orders):
            raise ValueError("orders must contain integers >=2")
        if len(set(self.orders)) != len(self.orders):
            raise ValueError("orders must be unique")
        if (not self.epsilons or any(not math.isfinite(e) or e <= 0 for e in self.epsilons)
                or len(set(self.epsilons)) != len(self.epsilons)):
            raise ValueError("epsilons must be distinct positive finite values")
        if self.pss_mode not in ("forecasting", "paper"):
            raise ValueError("pss_mode must be forecasting or paper")
        if self.f1_threshold is not None and (
                not math.isfinite(self.f1_threshold) or self.f1_threshold < 0):
            raise ValueError("f1_threshold must be finite and nonnegative")
        return self

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def preset(cls, split="confirmation"):
        if split not in ("confirmation", "development"):
            raise ValueError("split must be confirmation or development")
        return cls(
            segment_length=250,
            train_stride=130,
            clipping_norm=0.05,
            learning_rate=0.3,
            momentum=0.9,
            model_kind="feature_ae",
            evaluation_split="heldout" if split == "confirmation" else "development",
            evaluation_gap=400,
        ).validate()

    @classmethod
    def load(cls, path=None, split=None):
        if path is None:
            return cls.preset(split or "confirmation")
        values = json.loads(Path(path).read_text())
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if split is not None:
            values = {**cls.preset(split).to_dict(), **values}
        for key in ("orders", "epsilons"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values).validate()
