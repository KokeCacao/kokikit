# adapted from https://github.com/3DTopia/LGM
import torch
import numpy as np

from torch import Tensor
from typing import Tuple
from plyfile import PlyData, PlyElement

from ..dataset.nerf_dataset import NeRFDataset
from .aunet import UNet
from ..dmtet.gaussian import InverseSigmoid

def save_ply(gaussians, path, compatible=True):
    # gaussians: [B, N, 14]
    # compatible: save pre-activated gaussians as in the original paper

    assert gaussians.shape[0] == 1, 'only support batch size 1'
    
    means3D = gaussians[0, :, 0:3].contiguous().float()
    opacity = gaussians[0, :, 3:4].contiguous().float()
    scales = gaussians[0, :, 4:7].contiguous().float()
    rotations = gaussians[0, :, 7:11].contiguous().float()
    shs = gaussians[0, :, 11:].unsqueeze(1).contiguous().float() # [N, 1, 3]

    # prune by opacity
    mask = opacity.squeeze(-1) >= 0.005
    means3D = means3D[mask]
    opacity = opacity[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    shs = shs[mask]

    # invert activation to make it compatible with the original ply format
    if compatible:
        opacity = InverseSigmoid()(opacity)
        scales = torch.log(scales + 1e-8)
        shs = (shs - 0.5) / 0.28209479177387814

    xyzs = means3D.detach().cpu().numpy()
    f_dc = shs.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    l = ['x', 'y', 'z']
    # All channels except the 3 DC
    for i in range(f_dc.shape[1]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scales.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(rotations.shape[1]):
        l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]

    elements = np.empty(xyzs.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyzs, f_dc, opacities, scales, rotations), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')

    PlyData([el]).write(path)
    

class LGM(torch.nn.Module):
    def __init__(
        self,
        down_channels: Tuple[int, ...],
        down_attention: Tuple[bool, ...],
        mid_attention: bool,
        up_channels: Tuple[int, ...],
        up_attention: Tuple[bool, ...],
        splat_H: int,
        splat_W: int,
    ):
        super().__init__()
        
        self.split_H = splat_H
        self.split_W = splat_W

        # unet
        self.unet = UNet(
            9, 14, 
            down_channels=down_channels,
            down_attention=down_attention,
            mid_attention=mid_attention,
            up_channels=up_channels,
            up_attention=up_attention,
        )

        # last conv
        self.conv = torch.nn.Conv2d(14, 14, kernel_size=1) # NOTE: maybe remove it if train again

        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * torch.nn.functional.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = torch.nn.functional.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

    @staticmethod
    def get_rays(device: torch.device, H: int = 256, W: int = 256, fovy_rad: float = np.deg2rad(49.1)) -> Tensor:
        thetas, phis, rays_o = NeRFDataset._get_round_rays_o(
                batch_size=4,
                radius_float=1.5,
                idx=None,
                device=device,
        ) # [B,], [B,], [B, 3]
        up_vector, right_vector, forward_vector = NeRFDataset._get_lookat(
            batch_size=4,
            rays_o=rays_o,
            targets=torch.zeros_like(rays_o),
            device=device,
        ) # [B, 3], [B, 3], [B, 3]
        c2w = NeRFDataset._get_c2w(
            batch_size=4,
            up_vector=up_vector,
            right_vector=right_vector,
            forward_vector=forward_vector,
            rays_o=rays_o,
            device=device,
        ) # [B, 4, 4], [:, :3, 3] is rays_o
        
        # [array([[1, 0, 0, 0],
        #        [0, 1, 0, 0],
        #        [0, 0, 1, 1.5],
        #        [0, 0, 0, 1]], dtype=float32),
        # array([[ 0, 0, 1, 1.5],
        #        [ 0, 1, 0, 0],
        #        [-1, 0, 0, 0],
        #        [ 0, 0, 0, 1]], dtype=float32),
        # array([[-1, 0, 0, 0],
        #        [ 0, 1, 0, 0],
        #        [0, 0, -1, -1.5],
        #        [ 0, 0, 0, 1]], dtype=float32),
        # array([[0, 0, -1, -1.5],
        #        [ 0, 1, 0, 0],
        #        [ 1, 0, 0, 0],
        #        [ 0, 0, 0, 1]], dtype=float32)]
        
        focal: float = H / (2 * np.tan(fovy_rad / 2))
        rays_d = NeRFDataset._get_rays(W, H, c2w, focal, W / 2, H / 2, device) # [B, H, W, 3]
        rays_o = rays_o.reshape(rays_d.shape[0], 1, 1, rays_d.shape[-1]).expand_as(rays_d) # [B, 3] -> [B, H, W, 3]
        rays_embeddings = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [B, H, W, 6]
        rays_embeddings = rays_embeddings.permute(0, 3, 1, 2).contiguous() # [B, 6, H, W]
        return rays_embeddings
        

    def forward_gaussians(self, images):
        # images: [B, 4, 9, H, W]
        # return: Gaussians: [B, dim_t]

        B, V, C, H, W = images.shape
        images = images.view(B*V, C, H, W)

        x = self.unet(images) # [B*4, 14, h, w]
        x = self.conv(x) # [B*4, 14, h, w]

        x = x.reshape(B, 4, 14, self.split_H, self.split_W)
        
        ## visualize multi-view gaussian features for plotting figure
        # tmp_alpha = self.opacity_act(x[0, :, 3:4])
        # tmp_img_rgb = self.rgb_act(x[0, :, 11:]) * tmp_alpha + (1 - tmp_alpha)
        # tmp_img_pos = self.pos_act(x[0, :, 0:3]) * 0.5 + 0.5
        # kiui.vis.plot_image(tmp_img_rgb, save=True)
        # kiui.vis.plot_image(tmp_img_pos, save=True)

        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., 0:3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = self.rgb_act(x[..., 11:])

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        
        return gaussians
