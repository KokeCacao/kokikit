import torch
from torch import Tensor


class DensityInit():

    def __init__(self) -> None:
        pass

    def edit_density(self, densities: Tensor, original_positions: Tensor) -> Tensor:
        raise NotImplementedError


class GaussianBlobDensityInit(DensityInit):

    def __init__(self, blob_sharpness: float, blob_density: float, additive_density: float, norm_order: float) -> None:
        super().__init__()
        self.blob_sharpness = blob_sharpness
        self.blob_density = blob_density
        self.additive_density = additive_density
        self.norm_order = norm_order

    def edit_density(self, densities: Tensor, original_positions: Tensor) -> Tensor:
        # densities: [N, 1]
        # original_positions: [..., 3]
        positions_radius = torch.linalg.norm(original_positions.reshape(-1, 3), ord=self.norm_order, dim=-1) # [N,]
        blob = self.blob_density * torch.exp(-((positions_radius / self.blob_sharpness)**2) / 2)
        densities = densities + blob.unsqueeze(-1) + self.additive_density # [N, 1]
        return densities


class DreamfusionDensityInit(DensityInit):

    def __init__(self, blob_density: float, blob_radius: float, additive_density: float) -> None:
        super().__init__()
        self.blob_density = blob_density
        self.blob_radius = blob_radius
        self.additive_density = additive_density

    def edit_density(self, densities: Tensor, original_positions: Tensor) -> Tensor:
        positions_radius = torch.linalg.norm(original_positions.reshape(-1, 3), ord=2, dim=-1) # [N,]
        blob = self.blob_density * torch.exp(-(positions_radius**2).sum(-1) / (2 * self.blob_radius**2)) + self.additive_density
        return densities + blob


class EmptyDensityInit(DensityInit):

    def __init__(self, additive_density: float) -> None:
        super().__init__()
        self.additive_density = additive_density

    def edit_density(self, densities: Tensor, original_positions: Tensor) -> Tensor:
        # densities: [N, 1]
        # original_positions: [..., 3]
        densities = densities + self.additive_density # [N, 1]
        return densities