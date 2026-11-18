"""Shared shape conventions.

Set tensors use ``[batch, slot, channel, height, width]``. Fields operate on
independent rows shaped ``[batch, channel, height, width]`` and never couple
the batch axis.
"""

from typing import TypeAlias

import torch

SetTensor: TypeAlias = torch.Tensor
FieldTensor: TypeAlias = torch.Tensor
FeatureTensor: TypeAlias = torch.Tensor

