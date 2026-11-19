"""Frozen CLIP, DINOv2, and SigLIP tensor encoders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

from diverse_sampling_flows.objectives.clip_logdet import normalize_features
from diverse_sampling_flows.objectives.preprocessing import preprocess_images


class FrozenImageEncoder(nn.Module, ABC):
    def freeze(self) -> None:
        self.eval()
        self.requires_grad_(False)

    @abstractmethod
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode float RGB images while retaining image-input gradients."""


class CLIPImageEncoder(FrozenImageEncoder):
    def __init__(
        self, model: nn.Module, image_size: int, mean: list[float], std: list[float]
    ):
        super().__init__()
        self.model = model
        self.image_size, self.mean, self.std = image_size, mean, std
        self.freeze()

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str, **kwargs: Any):
        from transformers import AutoImageProcessor, CLIPModel

        model = CLIPModel.from_pretrained(model_id, revision=revision, **kwargs)
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        size = int(processor.crop_size["height"])
        return cls(model, size, list(processor.image_mean), list(processor.image_std))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = preprocess_images(
            images, size=self.image_size, mean=self.mean, std=self.std
        )
        return normalize_features(self.model.get_image_features(pixel_values=pixels))


class DINOImageEncoder(FrozenImageEncoder):
    def __init__(
        self, model: nn.Module, image_size: int, mean: list[float], std: list[float]
    ):
        super().__init__()
        self.model = model
        self.image_size, self.mean, self.std = image_size, mean, std
        self.freeze()

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str, **kwargs: Any):
        from transformers import AutoImageProcessor, AutoModel

        model = AutoModel.from_pretrained(model_id, revision=revision, **kwargs)
        processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
        size_value = processor.size
        size = int(size_value.get("shortest_edge", size_value.get("height")))
        return cls(model, size, list(processor.image_mean), list(processor.image_std))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = preprocess_images(
            images, size=self.image_size, mean=self.mean, std=self.std
        )
        output = self.model(pixel_values=pixels)
        return normalize_features(output.last_hidden_state[:, 0])


class SigLIPPromptEncoder(FrozenImageEncoder):
    def __init__(
        self,
        model: nn.Module,
        text_inputs: dict[str, torch.Tensor],
        image_size: int,
        mean,
        std,
    ):
        super().__init__()
        self.model, self.text_inputs = model, text_inputs
        self.image_size, self.mean, self.std = image_size, list(mean), list(std)
        self.freeze()

    @classmethod
    def from_pretrained(
        cls, model_id: str, revision: str, *, prompt: str, **kwargs: Any
    ):
        from transformers import AutoProcessor, SiglipModel

        model = SiglipModel.from_pretrained(model_id, revision=revision, **kwargs)
        processor = AutoProcessor.from_pretrained(model_id, revision=revision, **kwargs)
        text_inputs = processor(
            text=[prompt], padding="max_length", return_tensors="pt"
        )
        size_value = processor.image_processor.size
        size = int(size_value.get("height", size_value.get("shortest_edge")))
        return cls(
            model,
            dict(text_inputs),
            size,
            processor.image_processor.image_mean,
            processor.image_processor.image_std,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        pixels = preprocess_images(
            images, size=self.image_size, mean=self.mean, std=self.std
        )
        inputs = {
            key: value.to(pixels.device) for key, value in self.text_inputs.items()
        }
        return self.model(pixel_values=pixels, **inputs).logits_per_image[:, 0]

