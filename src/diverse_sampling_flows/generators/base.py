"""Generator adapter contract."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn


def configure_deterministic_backend() -> dict[str, Any]:
    """Lock the CUDA math policy used by all primary generator comparisons.

    cuBLAS reads its workspace policy during CUDA initialization, so launchers must
    export the variable before Python starts.  Checking it here prevents a run from
    silently claiming determinism after CUDA has already initialized.
    """
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if workspace not in {":4096:8", ":16:8"}:
        raise RuntimeError(
            "CUBLAS_WORKSPACE_CONFIG=:4096:8 must be exported before Python starts"
        )
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return {
        "cublas_workspace_config": workspace,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
    }


class FrozenGeneratorAdapter(nn.Module, ABC):
    """Map standardized Gaussian tensors to float RGB tensors."""

    latent_shape: tuple[int, ...]

    def freeze(self) -> None:
        self.eval()
        self.requires_grad_(False)

    def validate_noise(self, standardized_noise: torch.Tensor) -> None:
        if standardized_noise.ndim != len(self.latent_shape) + 1 or tuple(
            standardized_noise.shape[1:]
        ) != self.latent_shape:
            raise ValueError(
                "standardized noise must have shape "
                f"[N,{','.join(map(str, self.latent_shape))}], got "
                f"{tuple(standardized_noise.shape)}"
            )

    @abstractmethod
    def generate(
        self,
        standardized_noise: torch.Tensor,
        *,
        prompt: str,
        differentiable: bool = False,
    ) -> torch.Tensor:
        pass

