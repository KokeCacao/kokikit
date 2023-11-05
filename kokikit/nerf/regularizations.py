import torch
from torch import Tensor

from ..utils.utils import safe_magnitude


class Regularization():

    def __init__(self, weight: float):
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class EntropyRegularization(Regularization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        x = x.clamp(1e-5, 1 - 1e-5)
        loss = -torch.lerp(torch.log2(1 - x), torch.log2(x), x)
        # loss will produce nan when x is close to 0 or 1, which is fine
        # but remember make nan to be 0 before doing mean()
        loss = loss.nan_to_num().mean()
        return loss * self.weight


class ContrastRegularization(Regularization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[-1] == 3
        x = x.sum(dim=-1) / 3
        mean = x.mean()
        return ((x - mean).pow(2).mean()) * self.weight


class MeanRegularization(Regularization):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        return x.mean() * self.weight


class L2DerivativeRegularization(Regularization):

    def __init__(self, threshold: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.threshold = threshold

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, H, W, 3]
        down = x[:, 1:, :, :]
        up = x[:, :-1, :, :]
        right = x[:, :, 1:, :]
        left = x[:, :, :-1, :]

        # don't calculate portion for normal magnitude < threhshold (assuming we have black background)
        with torch.no_grad():
            ud_invalid = (safe_magnitude(down, dim=-1) < self.threshold) | (safe_magnitude(up, dim=-1) < self.threshold)
            lr_invalid = (safe_magnitude(right, dim=-1) < self.threshold) | (safe_magnitude(left, dim=-1) < self.threshold)
        dh = torch.where(ud_invalid, torch.zeros_like(down), down - up)
        dw = torch.where(lr_invalid, torch.zeros_like(right), right - left)

        return (dh.pow(2).mean() + dw.pow(2).mean()) * self.weight
