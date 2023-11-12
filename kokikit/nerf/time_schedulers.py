import math
import torch
import warnings
from typing import Generator, Tuple
from torch import Tensor

try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn(f"Cannot import matplotlib.pyplot, skip plotting.", ImportWarning)

from .weight_schedulers import WeightScheduler


class TimeScheduler():

    def __init__(self, scheduler_scale: int) -> None:
        self.scheduler_scale = scheduler_scale

    def get_times(self, i: Tensor) -> Tensor:
        # i: [1, 2, ..., T]
        raise NotImplementedError


class RandomTimeScheduler(TimeScheduler):
    # good at averaging out noise, construct good stape

    def __init__(self, scheduler_scale: int, t_min: float, t_max: float) -> None:
        super().__init__(scheduler_scale=scheduler_scale)
        self.t_min: int = math.ceil(t_min * scheduler_scale) + 1
        self.t_max: int = math.floor(t_max * scheduler_scale) + 1
        assert 1 <= self.t_min < self.t_max <= scheduler_scale

    def get_times(self, i: Tensor) -> Tensor:
        return torch.randint(low=self.t_min, high=self.t_max + 1, size=(i.shape[0],), dtype=torch.long)


class RandomDecayTimeScheduler(TimeScheduler):

    def __init__(self, scheduler_scale: int, t_min_start: float, t_max_start: float, t_min_end: float, t_max_end: float, t_start_at: float, t_end_at: float) -> None:
        super().__init__(scheduler_scale)
        self.t_min_start: int = math.ceil(t_min_start * scheduler_scale) + 1
        self.t_max_start: int = math.floor(t_max_start * scheduler_scale) + 1
        self.t_min_end: int = math.ceil(t_min_end * scheduler_scale) + 1
        self.t_max_end: int = math.floor(t_max_end * scheduler_scale) + 1
        assert 1 <= self.t_min_start < self.t_max_start <= scheduler_scale
        assert 1 <= self.t_min_end < self.t_max_end <= scheduler_scale
        
        self.t_start_at: float = t_start_at
        self.t_end_at: float = t_end_at

    def get_times(self, i: Tensor) -> Tensor:
        percentage = (i.to(dtype=torch.float) / i.shape[0] - self.t_start_at) / (self.t_end_at - self.t_start_at)
        percentage = percentage.clamp(0.0, 1.0)
        t_min = self.t_min_start + (self.t_min_end - self.t_min_start) * percentage
        t_max = self.t_max_start + (self.t_max_end - self.t_max_start) * percentage
        # torch.randint(low=t_min, high=t_max + 1, size=(i.shape[0],), dtype=torch.long)

        t_min_np = t_min.to(dtype=torch.long).tolist()
        t_max_np = t_max.to(dtype=torch.long).tolist()

        times = torch.zeros(i.shape[0], dtype=torch.long)
        for j in range(i.shape[0]):
            times[j] = torch.randint(low=t_min_np[j], high=t_max_np[j] + 1, size=(1,), dtype=torch.long)
        return times


class LinearTimeScheduler(TimeScheduler):

    def __init__(self, scheduler_scale: int, t_min: float, t_max: float) -> None:
        super().__init__(scheduler_scale=scheduler_scale)
        self.t_min: int = math.ceil(t_min * scheduler_scale) + 1
        self.t_max: int = math.floor(t_max * scheduler_scale) + 1
        assert 1 <= self.t_min < self.t_max <= scheduler_scale

    def get_times(self, i: Tensor) -> Tensor:
        return torch.linspace(start=self.t_max, end=self.t_min, steps=i.shape[0], dtype=torch.long)


class DreamTimeTimeScheduler(TimeScheduler):
    # good at enforcing correct texture, construct consistency

    def __init__(self, scheduler_scale: int) -> None:
        super().__init__(scheduler_scale=scheduler_scale)
        # weight config and cache
        assert scheduler_scale == 1000
        self.m1 = 800
        self.m2 = 500
        self.s1 = 300
        self.s2 = 100
        self.t = torch.arange(start=1, end=scheduler_scale + 1, step=1, dtype=torch.long) # [1, 2, ..., T]
        self.wt = self.w(self.t) # [w^*(1), w^*(2), ..., w^*(T)] in O(T)
        self.sum_wt = torch.sum(self.wt) # \sum_{t = 1}^T w^*(t) in O(T)

        # time config and cache
        t: Tensor = torch.arange(start=1, end=scheduler_scale + 1, step=1, dtype=torch.long)
        pts: Tensor = self.p(t).flip(dims=(0,))
        self.pt_cumsum = torch.cumsum(pts, dim=0).flip(dims=(0,)) # [\sum_{t=1}^T p(t), \sum_{t=T}^T p(t)] in O(T)

    def ti(self, i: Tensor) -> Tensor:
        t: Tensor = torch.arange(start=1, end=self.scheduler_scale + 1, step=1, dtype=torch.long)
        values = torch.abs(self.pt_cumsum[t - 1].unsqueeze(-1) - (i.unsqueeze(0) / i.shape[0]))
        return torch.argmin(values, dim=0) + 1 # \argmin_{t} | \sum_{t=t}^T p(t) - i/T | in O(T)

    def w(self, t: Tensor) -> Tensor:
        # if t > self.m1: return np.exp(-(t - self.m1)**2 / (2 * self.s1**2))
        # elif t < self.m2: return np.exp(-(t - self.m2)**2 / (2 * self.s2**2))
        # else: # self.m2 <= t and t <= self.m1 return 1.0
        ret = torch.ones_like(t, dtype=torch.float)
        large_t = t > self.m1
        small_t = t < self.m2
        ret[large_t] = torch.exp(-(t[large_t] - self.m1)**2 / (2 * self.s1**2))
        ret[small_t] = torch.exp(-(t[small_t] - self.m2)**2 / (2 * self.s2**2))
        return ret

    def p(self, t: Tensor) -> Tensor:
        # p(t) = w^*(t) / (\sum_{t = 1}^T w^*(t)) in O(1)
        return self.w(t) / self.sum_wt

    def get_times(self, i: Tensor) -> Tensor:
        return self.ti(i)


class DreamTimeMaxTimeScheduler(DreamTimeTimeScheduler):

    def __init__(self, scheduler_scale: int, t_min: float, t_max: float) -> None:
        super().__init__(scheduler_scale=scheduler_scale)
        self.t_min = t_min
        self.t_max = t_max

    def get_times(self, i: Tensor) -> Tensor:
        max = super().get_times(i).to(dtype=torch.float)
        rand = torch.rand(max.shape, dtype=torch.float, device=max.device)
        rand = rand * (self.t_max - self.t_min) + self.t_min
        return (rand * max).to(dtype=torch.long) + 1


class DreamTimeGaussianTimeScheduler(DreamTimeTimeScheduler):

    def __init__(self, scheduler_scale: int, t_min: float, t_max: float, deviation: float) -> None:
        super().__init__(scheduler_scale=scheduler_scale)
        self.t_min = t_min
        self.t_max = t_max
        self.deviation = deviation

    def get_times(self, i: Tensor) -> Tensor:
        t = super().get_times(i).to(dtype=torch.float)
        randn = torch.randn(t.shape, dtype=torch.float, device=t.device) * (self.deviation * self.scheduler_scale)
        return (t + randn).clamp(self.t_min * self.scheduler_scale, self.t_max * self.scheduler_scale).to(dtype=torch.long) + 1


class DiffusionSchedule():

    def __init__(self, time_scheduler: TimeScheduler, weight_scheduler: WeightScheduler, steps: int, start_step: int = 0) -> None:
        self.start_step: int = start_step
        i: Tensor = torch.arange(start=1, end=steps + 1, step=1, dtype=torch.long)
        assert i.dim() == 1 and i.shape[0] == steps
        # assumes i from 1 to steps, not from 0 to steps - 1
        self.t: Tensor = time_scheduler.get_times(i)
        assert torch.all(self.t >= 1)
        assert torch.all(self.t <= time_scheduler.scheduler_scale)
        self.w: Tensor = weight_scheduler.get_weights(self.t)
        assert torch.all(self.w >= 0)
        assert torch.all(self.w <= 1)

        # convert time from [1, T] to [0, T - 1] for diffuser
        self.t -= 1

    def to_device(self, device: torch.device):
        self.t = self.t.to(device)
        self.t.requires_grad_(False)
        self.w = self.w.to(device)
        self.w.requires_grad_(False)
        return self

    def __len__(self) -> int:
        return len(self.t) - self.start_step

    def __iter__(self) -> Generator[Tuple[Tensor, Tensor], None, None]:
        count = self.start_step
        while count < len(self.t):
            timestamp: Tensor = self.t[count]
            loss_weight: Tensor = self.w[count]
            yield timestamp, loss_weight
            count += 1

    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.t, self.w, marker='o')
        plt.xlabel('t')
        plt.ylabel('w')
        plt.title('Plot of w against t')
        plt.grid(True)
        plt.gca().invert_xaxis() # inverting the x-axis since time goes from large to small
        plt.savefig('TimeWeightScheduler_t_w.png')

        i = torch.arange(start=1, end=len(self.t) + 1, step=1, dtype=torch.long)

        plt.figure(figsize=(10, 6))
        plt.plot(i, self.t, marker='o')
        plt.xlabel('i')
        plt.ylabel('t')
        plt.title('Plot of t against i')
        plt.grid(True)
        plt.savefig('TimeWeightScheduler_i_t.png')
