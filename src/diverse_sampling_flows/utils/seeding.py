"""Independent, resumable random-number namespaces."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import torch


def _namespace_seed(base_seed: int, namespace: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{namespace}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


class SeedStreams:
    """Own persistent generators for disjoint experiment seed namespaces."""

    def __init__(self, base_seed: int) -> None:
        self.base_seed = int(base_seed)
        self.counters: dict[str, int] = {}
        self._generators: dict[str, torch.Generator] = {}

    @staticmethod
    def _key(namespace: str, device: torch.device | str) -> str:
        return f"{namespace}|{torch.device(device)}"

    def generator(
        self, namespace: str, device: torch.device | str = "cpu"
    ) -> torch.Generator:
        key = self._key(namespace, device)
        if key not in self._generators:
            generator = torch.Generator(device=torch.device(device))
            generator.manual_seed(_namespace_seed(self.base_seed, namespace))
            self._generators[key] = generator
        return self._generators[key]

    def seed_for(self, namespace: str) -> int:
        """Return and record a deterministic namespace seed without drawing rows."""
        self.counters.setdefault(namespace, 0)
        return _namespace_seed(self.base_seed, namespace)

    def standard_normal(
        self,
        namespace: str,
        shape: Sequence[int],
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        generator = self.generator(namespace, device)
        result = torch.randn(
            tuple(shape), generator=generator, device=device, dtype=dtype
        )
        self.counters[namespace] = self.counters.get(namespace, 0) + 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "counters": dict(self.counters),
            "namespace_seeds": {
                namespace: _namespace_seed(self.base_seed, namespace)
                for namespace in sorted(self.counters)
            },
            "generator_states": {
                key: generator.get_state().cpu()
                for key, generator in self._generators.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.base_seed = int(state["base_seed"])
        self.counters = {str(k): int(v) for k, v in state["counters"].items()}
        self._generators.clear()
        for key, generator_state in state.get("generator_states", {}).items():
            namespace, device_string = key.rsplit("|", 1)
            generator = self.generator(namespace, torch.device(device_string))
            generator.set_state(generator_state.cpu())

    def manifest(self) -> dict[str, Any]:
        return {
            "base_seed": self.base_seed,
            "counters": dict(sorted(self.counters.items())),
            "namespace_seeds": {
                namespace: _namespace_seed(self.base_seed, namespace)
                for namespace in sorted(self.counters)
            },
        }

