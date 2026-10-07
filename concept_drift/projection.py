import hashlib
from collections.abc import Iterable

import torch


class CountSketchProjector:


    def __init__(self, dimension: int = 128, seed: int = 42) -> None:
        if dimension < 1:
            raise ValueError("dimension must be positive")
        self.dimension = int(dimension)
        self.seed = int(seed)
        self._cache: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = {}

    def _parameter_seed(self, name: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{name}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False) % (2**63 - 1)

    def _mapping(
        self,
        name: str,
        size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (name, size, str(device))
        if key not in self._cache:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._parameter_seed(name))
            buckets = torch.randint(
                self.dimension, (size,), generator=generator, device=device, dtype=torch.int64
            )
            signs = torch.randint(0, 2, (size,), generator=generator, device=device, dtype=torch.int8)
            signs = signs.mul(2).sub(1)
            self._cache[key] = (buckets, signs)
        buckets, signs = self._cache[key]
        return buckets, signs.to(dtype=dtype)

    def project(
        self,
        named_gradients: Iterable[tuple[str, torch.Tensor | None]],
    ) -> torch.Tensor:
        output: torch.Tensor | None = None
        for name, gradient in named_gradients:
            if gradient is None:
                continue
            flat = gradient.detach().reshape(-1)
            if output is None:
                output = torch.zeros(self.dimension, device=flat.device, dtype=flat.dtype)
            buckets, signs = self._mapping(name, flat.numel(), flat.device, flat.dtype)
            output.scatter_add_(0, buckets, flat * signs)
        if output is None:
            raise ValueError("No gradients were supplied to the projector")
        return output
