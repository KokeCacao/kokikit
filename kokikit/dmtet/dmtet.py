import warnings
import torch
import itertools
import os
import numpy as np
import nvdiffrast.torch as dr

from typing import Optional, Any, Union, Literal, Sequence, Dict, List
from torch import Tensor

from ..utils.utils import safe_normalize
from ..nerf.rays import RayBundle
from ..nerf.ray_samplers import RaySamples
from ..nerf.nerf_fields import NeRFField
from ..nerf.field_base import FieldBase


class DMTetStructure():

    def __init__(self, device):
        self.device = device
        self.triangle_table = torch.tensor([
            [-1, -1, -1, -1, -1, -1],
            [+1, +0, +2, -1, -1, -1],
            [+4, +0, +3, -1, -1, -1],
            [+1, +4, +2, +1, +3, +4],
            [+3, +1, +5, -1, -1, -1],
            [+2, +3, +0, +2, +5, +3],
            [+1, +4, +0, +1, +5, +4],
            [+4, +2, +5, -1, -1, -1],
            [+4, +5, +2, -1, -1, -1],
            [+4, +1, +0, +4, +5, +1],
            [+3, +2, +0, +3, +5, +2],
            [+1, +3, +5, -1, -1, -1],
            [+4, +1, +2, +4, +3, +1],
            [+3, +0, +4, -1, -1, -1],
            [+2, +0, +1, -1, -1, -1],
            [-1, -1, -1, -1, -1, -1],
        ], dtype=torch.long, device=device)
        self.num_triangles_table = torch.tensor([0, 1, 1, 2, 1, 2, 2, 1, 1, 2, 2, 1, 2, 1, 1, 0], dtype=torch.long, device=device)
        self.base_tet_edges = torch.tensor([0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3], dtype=torch.long, device=device)

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:, 0] > edges_ex2[:, 1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)
            b = torch.gather(input=edges_ex2, index=1 - order, dim=1)

        return torch.stack([a, b], -1)

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        # pos_nx3: [N, 3]
        # sdf_n:   [N]
        # tet_fx4: [F, 4]

        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1, 4)
            occ_sum = torch.sum(occ_fx4, -1) # [F,]
            valid_tets = (occ_sum > 0) & (occ_sum < 4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:, self.base_tet_edges].reshape(-1, 2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges, dim=0, return_inverse=True)

            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1, 2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device=self.device) * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long, device=self.device)
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]

        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1, 2, 3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1, 2, 1)
        edges_to_interp_sdf[:, -1] *= -1

        denominator = edges_to_interp_sdf.sum(1, keepdim=True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1]) / denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)

        idx_map = idx_map.reshape(-1, 6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device=self.device))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(input=idx_map[num_triangles == 1], dim=1, index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]).reshape(-1, 3),
            torch.gather(input=idx_map[num_triangles == 2], dim=1, index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]).reshape(-1, 3),
        ), dim=0)

        return verts, faces


class DMTet(FieldBase):

    def __init__(
            self,
            latent_dreamfusion: bool,
            degree_latent: int,
            nerf_field: NeRFField,
            nerf_field_state_dict: str,
            tet_grid_size: Literal[32, 64, 128],
            tet_density_threshold: float,
            dtype: torch.dtype,
            device: torch.device,
            tet_npz_path: str,
            raster_method: Literal['cuda', 'opengl'] = 'opengl', # see: https://nvlabs.github.io/nvdiffrast/#rasterizing-with-cuda-vs-opengl-new
    ):
        super().__init__(
            latent_dreamfusion=latent_dreamfusion,
            degree_latent=degree_latent,
            dtype=dtype,
            device=device,
        )

        if raster_method == 'cuda':
            self.glctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = dr.RasterizeCudaContext()
            warnings.warn("CUDA rasterization only supports image size < 2048 and performance for complex mesh might not be as good as OpenGL", RuntimeWarning)
        elif raster_method == 'opengl':
            self.glctx: Union[dr.RasterizeCudaContext, dr.RasterizeGLContext] = dr.RasterizeGLContext()
        else:
            raise ValueError(f"raster_method must be either 'cuda' or 'opengl', got {raster_method}")

        tet: Dict[str, Any] = np.load(tet_npz_path)
        tet_vertices = tet['vertices']
        tet_indices = tet['indices']

        self.tet_grid_size: Literal[32, 64, 128] = tet_grid_size
        self.tet_density_threshold = tet_density_threshold

        # [nerf_field]
        # TODO: load state dict of pretrained nerf
        # state_dict = torch.load(opt.init_with, map_location=device)
        # model.load_state_dict(state_dict['model'], strict=False)
        self.nerf_field = nerf_field
        if nerf_field_state_dict and os.path.exists(nerf_field_state_dict):
            state_dict = torch.load(nerf_field_state_dict)
            if state_dict['nerf_fields'] is not None:
                # TODO: this is assuming we only have one particle
                nerf_fields_state = state_dict['nerf_fields'][0]
                self.nerf_field.load_state_dict(nerf_fields_state)
            else:
                warnings.warn(f"Cannot initialize DMTet from {nerf_field_state_dict}, nerf_fields is None", RuntimeWarning)
        elif nerf_field_state_dict:
            warnings.warn(f"Cannot initialize DMTet from {nerf_field_state_dict}, file does not exist", RuntimeWarning)

        # [verts], [indices], [tet_scale]
        # IMPORTANT: minus sign to flip all axis of vertices so that mesh shape is correct
        self.verts = -torch.tensor(tet_vertices, dtype=dtype, device=device) * 2 # covers [-1, 1], shape = [V, 3]
        self.indices = torch.tensor(tet_indices, dtype=torch.long, device=device) # shape = [1524684, 4], a point will have roughly 6 indices since 360/60 = 6, except for boundary, (roughly on average 5.496139288 indices per vertex)
        self.tet_scale = torch.tensor([1, 1, 1], dtype=dtype, device=device) # shape = [3]

        # [dmtet]
        self.dmtet = DMTetStructure('cuda')

        # [sdf] and [deform]
        self.sdf = torch.nn.Parameter(torch.zeros_like(self.verts[..., 0]), requires_grad=True) # shape = [V]
        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True) # shape = [V, 3]

    def to_device(self, device: torch.device):
        # TODO: actually do a to_device
        # [density]
        density, _ = self.nerf_field.get_density(self.verts.unsqueeze(-2)) # verts [V, 3] -> [V, 1, 3] in [-1, 1]
        density = density.view(-1) # [V]

        # [tet_scale] and [verts]
        self.tet_scale = self.verts[density > self.tet_density_threshold].abs().amax(dim=0) + 1e-1 # [3]
        self.verts = self.verts * self.tet_scale # [V, 3]

        # [sdf]
        self.sdf.data += (density - self.tet_density_threshold).clamp(-1, 1) # [V]

    def parameters(self):
        return itertools.chain(iter([self.sdf, self.deform]), self.nerf_field.parameters())

    # TODO: properly do renderers
    def _forward(self, ray_bundle: RayBundle, renderers: Sequence[Any]):
        # mvp: [B, 4, 4]

        H = ray_bundle.origins.shape[1]
        W = ray_bundle.origins.shape[2]
        # WARNING: projection will work differently than ray tracer, expect differences in render results
        mvp: Optional[Tensor] = ray_bundle.mvp
        assert mvp is not None

        # get mesh
        sdf = self.sdf
        deform = torch.tanh(self.deform) / self.tet_grid_size

        verts, faces = self.dmtet.__call__(self.verts + deform, sdf, self.indices) # vertex position: [V, 3], face indices: [F, 3]

        # get normals
        i0, i1, i2 = faces[:, 0], faces[:, 1], faces[:, 2] # [F,], [F,], [F,]
        v0, v1, v2 = verts[i0, :], verts[i1, :], verts[i2, :] # vertex positions of triangle faces [F, 3], [F, 3], [F, 3]

        faces = faces.int()

        face_normals = torch.cross(v1 - v0, v2 - v0)
        face_normals = safe_normalize(face_normals) # [F, 3]

        vn = torch.zeros_like(verts) # vertex normals [V, 3]
        vn.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
        vn.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

        vn = torch.where(torch.sum(vn * vn, -1, keepdim=True) > 1e-20, vn, torch.tensor([0.0, 0.0, 1.0], dtype=vn.dtype, device=vn.device)) # [V, 3]

        # rasterization
        verts_clip: Tensor = torch.bmm(
            torch.nn.functional.pad(
                verts,
                pad=(0, 1),
                mode='constant',
                value=1.0,
            ).unsqueeze(0).repeat(mvp.shape[0], 1, 1),
            mvp.permute(0, 2, 1),
        ).float() # [B, V, 4]

        rast: Tensor
        xyzs: Tensor
        if verts_clip.any():
            # See: https://nvlabs.github.io/nvdiffrast/
            rast, rast_db = dr.rasterize(glctx=self.glctx, pos=verts_clip, tri=faces, resolution=(H, W), ranges=None, grad_db=True) # [B, H, W, 4], [B, H, W, 4] # type: ignore
            xyzs, _ = dr.interpolate(attr=verts.unsqueeze(0), rast=rast, tri=faces, rast_db=None, diff_attrs=None) # [B, H, W, 3] # type: ignore
        else:
            rast = torch.zeros((mvp.shape[0], H, W, 4), dtype=verts_clip.dtype, device=verts_clip.device) # [B, H, W, 4]
            xyzs = torch.zeros((mvp.shape[0], H, W, 3), dtype=verts_clip.dtype, device=verts_clip.device) # [B, H, W, 3]

        alpha = (rast[..., 3:] > 0).float() # [B, H, W, 1]
        normal, _ = dr.interpolate(attr=vn.unsqueeze(0).contiguous(), rast=rast, tri=faces, rast_db=None, diff_attrs=None) # [B, H, W, 3] # type: ignore
        normal = safe_normalize(normal)

        # do the lighting here since we have normal from mesh now.
        xyzs = xyzs.view(-1, 1, 3)
        mask = (rast[..., 3:] > 0).view(-1).detach() # [B * H * W]
        color: Tensor = torch.zeros_like(xyzs)
        if mask.any():
            # TODO: make sure shape and input, output value range are correct
            masked_center = xyzs[mask] # [?, 1, 3]
            # dirty hack to type-check
            ray_samples = RaySamples(
                center=masked_center, # [?, 1, 3]
                directions=torch.tensor([0.0, 0.0, 1.0], dtype=masked_center.dtype, device=masked_center.device).view(1, 3).expand(masked_center.shape[0], 3), # [?, 3]
                euclidean_starts=None,
                euclidean_ends=None,
                spacing_starts=None,
                spacing_ends=None,
                origins=ray_bundle.origins, # dummy, not used
                deltas=torch.zeros((1,)), # dummy, not used
                spacing_to_euclidean_fn=None,
                collated=True, # dummy, not used
            )
            colors_out, _ = self.nerf_field.forward(ray_samples) # [?, 1, C], [?, 1, 1]
            color[mask] = colors_out # [?, 1, C]
        color = color.view(-1, H, W, 3) # [B, H, W, 3]

        color: Tensor = dr.antialias(color, rast, verts_clip, faces) # type: ignore
        alpha: Tensor = dr.antialias(alpha, rast, verts_clip, faces) # type: ignore
        color = color.clamp(0, 1) # [B, H, W, 3]
        alpha = alpha.clamp(0, 1) # [B, H, W, 1]

        # mix background color
        depth = rast[:, :, :, [2]] # [B, H, W]
        color = color + (1 - alpha) * 1 # [B, H, W, 3]

        # TODO: regularization loss
        # calculate [images]
        images: List[Tensor] = [] # List[B, H, W, ?]
        losses: List[Tensor] = [] # List[?]
        for renderer in renderers:
            images.append(color.clamp(0, 1).flip((1,)))
            losses.append(torch.zeros(1, device='cuda'))

        return images, losses
