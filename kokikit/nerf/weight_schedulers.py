import torch
from torch import Tensor


class WeightScheduler():

    def __init__(self) -> None:
        pass

    def get_weights(self, t: Tensor) -> Tensor:
        # w: [w(1), w(2), ..., w(T)]
        raise NotImplementedError


class ConstantWeightScheduler(WeightScheduler):

    def __init__(self) -> None:
        super().__init__()

    def get_weights(self, t: Tensor) -> Tensor:
        return torch.ones_like(t, dtype=torch.float)


class SDSWeightScheduler(WeightScheduler):

    def __init__(self, alphas_cumprod: Tensor) -> None:
        super().__init__()
        self.alphas_cumprod = alphas_cumprod

    def get_weights(self, t: Tensor) -> Tensor:
        return 1 - self.alphas_cumprod[t - 1]


class FantasiaWeightScheduler(WeightScheduler):

    def __init__(self, alphas_cumprod: Tensor) -> None:
        super().__init__()
        self.alphas_cumprod = alphas_cumprod

    def get_weights(self, t: Tensor) -> Tensor:
        return self.alphas_cumprod[t - 1]**0.5 * (1 - self.alphas_cumprod[t - 1])
