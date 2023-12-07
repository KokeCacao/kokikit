import warnings
import subprocess
import os
import io
import base64
import pathlib
import math
import json
import numpy as np
import torch
import PIL.Image
import torchvision

from torch import Tensor
from .dataset import Dataset
from .nerf_dataset import NeRFDataset
from ..nerf.colliders import Collider
from ..nerf.rays import RayBundle
from typing import Dict, Any, Optional, List, Tuple, Callable

PYCOLMAP = False
FFMPEG = False
FFPROBE = False

try:
    import pycolmap
    PYCOLMAP = True
except ImportError:
    warnings.warn("pycolmap is not installed, so you cannot use reconstruction functionalities.")
try:
    subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    FFMPEG = True
except (subprocess.CalledProcessError, FileNotFoundError):
    warnings.warn("ffmpeg is not installed, so you cannot use reconstruction functionalities.")
try:
    subprocess.run(["ffprobe", "-version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    FFPROBE = True
except (subprocess.CalledProcessError, FileNotFoundError):
    warnings.warn("ffprobe is not installed, so you cannot use reconstruction functionalities.")


# Below functions are from NeRFStudio
# https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/utils/colmap_parsing_utils.py
def qvec2rotmat(qvec):
    return np.array([
        [
            1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
            2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
            2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
        ],
        [
            2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
            1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
            2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
        ],
        [
            2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
            2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
            1 - 2 * qvec[1]**2 - 2 * qvec[2]**2,
        ],
    ])


def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = (
        np.array([ # type: ignore
            [Rxx - Ryy - Rzz, 0, 0, 0],
            [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
            [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
            [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
        ]) / 3.0)
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


from enum import Enum


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"


def parse_colmap_camera_params(camera) -> Dict[str, Any]:
    """
    Parses all currently supported COLMAP cameras into the transforms.json metadata

    Args:
        camera: COLMAP camera
    Returns:
        transforms.json metadata containing camera's intrinsics and distortion parameters

    """
    out: Dict[str, Any] = {
        "w": camera.width,
        "h": camera.height,
    }

    # Parameters match https://github.com/colmap/colmap/blob/dev/src/base/camera_models.h
    camera_params = camera.params
    if camera.model_name == "SIMPLE_PINHOLE":
        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model_name == "PINHOLE":
        # f, cx, cy, k

        # du = 0
        # dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = 0.0
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model_name == "SIMPLE_RADIAL":
        # f, cx, cy, k

        # r2 = u**2 + v**2;
        # radial = k * r2
        # du = u * radial
        # dv = u * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model_name == "RADIAL":
        # f, cx, cy, k1, k2

        # r2 = u**2 + v**2;
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial
        # dv = v * radial
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["p1"] = 0.0
        out["p2"] = 0.0
        camera_model = CameraModel.OPENCV
    elif camera.model_name == "OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2

        # uv = u * v;
        # r2 = u**2 + v**2
        # radial = k1 * r2 + k2 * r2 ** 2
        # du = u * radial + 2 * p1 * u*v + p2 * (r2 + 2 * u**2)
        # dv = v * radial + 2 * p2 * u*v + p1 * (r2 + 2 * v**2)
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV
    elif camera.model_name == "OPENCV_FISHEYE":
        # fx, fy, cx, cy, k1, k2, k3, k4

        # r = sqrt(u**2 + v**2)

        # if r > eps:
        #    theta = atan(r)
        #    theta2 = theta ** 2
        #    theta4 = theta2 ** 2
        #    theta6 = theta4 * theta2
        #    theta8 = theta4 ** 2
        #    thetad = theta * (1 + k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8)
        #    du = u * thetad / r - u;
        #    dv = v * thetad / r - v;
        # else:
        #    du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["k3"] = float(camera_params[6])
        out["k4"] = float(camera_params[7])
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model_name == "FULL_OPENCV":
        # fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6

        # u2 = u ** 2
        # uv = u * v
        # v2 = v ** 2
        # r2 = u2 + v2
        # r4 = r2 * r2
        # r6 = r4 * r2
        # radial = (1 + k1 * r2 + k2 * r4 + k3 * r6) /
        #          (1 + k4 * r2 + k5 * r4 + k6 * r6)
        # du = u * radial + 2 * p1 * uv + p2 * (r2 + 2 * u2) - u
        # dv = v * radial + 2 * p2 * uv + p1 * (r2 + 2 * v2) - v
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["k1"] = float(camera_params[4])
        out["k2"] = float(camera_params[5])
        out["p1"] = float(camera_params[6])
        out["p2"] = float(camera_params[7])
        out["k3"] = float(camera_params[8])
        out["k4"] = float(camera_params[9])
        out["k5"] = float(camera_params[10])
        out["k6"] = float(camera_params[11])
        raise NotImplementedError(f"{camera.model_name} camera model is not supported yet!")
    elif camera.model_name == "FOV":
        # fx, fy, cx, cy, omega
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[1])
        out["cx"] = float(camera_params[2])
        out["cy"] = float(camera_params[3])
        out["omega"] = float(camera_params[4])
        raise NotImplementedError(f"{camera.model_name} camera model is not supported yet!")
    elif camera.model_name == "SIMPLE_RADIAL_FISHEYE":
        # f, cx, cy, k

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     thetad = theta * (1 + k * theta2)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = 0.0
        out["k3"] = 0.0
        out["k4"] = 0.0
        camera_model = CameraModel.OPENCV_FISHEYE
    elif camera.model_name == "RADIAL_FISHEYE":
        # f, cx, cy, k1, k2

        # r = sqrt(u ** 2 + v ** 2)
        # if r > eps:
        #     theta = atan(r)
        #     theta2 = theta ** 2
        #     theta4 = theta2 ** 2
        #     thetad = theta * (1 + k * theta2)
        #     thetad = theta * (1 + k1 * theta2 + k2 * theta4)
        #     du = u * thetad / r - u;
        #     dv = v * thetad / r - v;
        # else:
        #     du = dv = 0
        out["fl_x"] = float(camera_params[0])
        out["fl_y"] = float(camera_params[0])
        out["cx"] = float(camera_params[1])
        out["cy"] = float(camera_params[2])
        out["k1"] = float(camera_params[3])
        out["k2"] = float(camera_params[4])
        out["k3"] = 0
        out["k4"] = 0
        camera_model = CameraModel.OPENCV_FISHEYE
    else:
        # THIN_PRISM_FISHEYE not supported!
        raise NotImplementedError(f"{camera.model_name} camera model is not supported yet!")

    out["camera_model"] = camera_model.value
    return out


# Above functions are from NeRFStudio


def get_video_frame_count(video_path: pathlib.Path) -> int:
    try:
        # Construct the ffprobe command to get frames
        ffprobe_cmd = [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=nb_frames',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            str(video_path),
        ]

        # Execute the command and capture the output
        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        return int(result.stdout.strip())
    except ValueError:
        # Handle cases where nb_frames is not available
        # Fall back to calculating frame count based on duration and frame rate
        ffprobe_cmd = [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            'v:0',
            '-show_entries',
            'stream=duration,r_frame_rate',
            '-of',
            'json',
            str(video_path),
        ]

        result = subprocess.run(ffprobe_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        info = json.loads(result.stdout)

        duration = float(info['streams'][0]['duration'])
        frame_rate = eval(info['streams'][0]['r_frame_rate'])

        return int(duration * frame_rate)


class ColmapDataset(Dataset):

    def __init__(
        self,
        image_dir: Optional[pathlib.Path],
        output_dir: pathlib.Path,
        scale_factor: float,
        cleanup: bool,
        collider: Collider,
        device: torch.device,
    ) -> None:
        if not PYCOLMAP:
            raise ImportError("pycolmap is not installed.")
        if not FFMPEG:
            raise ImportError("ffmpeg is not installed.")
        if not FFPROBE:
            raise ImportError("ffprobe is not installed.")

        self.image_dir: Optional[pathlib.Path] = image_dir
        self.output_dir: pathlib.Path = output_dir
        self.json: Optional[Dict[str, Any]] = None

        self.collider: Collider = collider
        self.device: torch.device = device
        self.near_plane: float = collider.near_plane
        self.far_plane: float = collider.far_plane
        self.scale_factor: float = scale_factor

        if self.image_dir is not None:
            self.json = self.preprocess(cleanup=cleanup)
            self.json = self.to_our_convention(self.json)
            self.json = self.to_center_of_mass(self.json)
            self.json = self.to_unit_cube(self.json)

        self.frames = None # [B, 3, H, W]
        # {
        # "w": 1920,
        # "h": 1080,
        # "fl_x": 1669.1726948720423,
        # "fl_y": 1669.1726948720423,
        # "cx": 960.0,
        # "cy": 540.0,
        # "k1": 0.06062808667705901,
        # "k2": 0.0,
        # "p1": 0.0,
        # "p2": 0.0,
        # "camera_model": "OPENCV",
        # "frames": [
        #     {
        #         "file_path": "/data/kokikit/kokikit/dataset/tmp/IMG_938800138.jpg",
        #         "transform_matrix": [
        #             [
        #                 0.023085351749162247,
        #                 -0.898439363406638,
        #                 -0.43849033833836343,
        #                 5.833914614892413
        #             ],
        #             [
        #                 0.9195573118033555,
        #                 0.19118400292988544,
        #                 -0.3433118514305791,
        #                 -2.1048700068818516
        #             ],
        #             [
        #                 0.39227721937885196,
        #                 -0.3952915219242382,
        #                 0.8305800357889753,
        #                 -2.885523137965988
        #             ],
        #             [
        #                 0.0,
        #                 0.0,
        #                 0.0,
        #                 1.0
        #             ]
        #         ],
        #         "colmap_im_id": 143
        #     },
        #     ...
        # ],
        # "applied_transform": [
        #         [
        #             0.0,
        #             1.0,
        #             0.0,
        #             0.0
        #         ],
        #         [
        #             1.0,
        #             0.0,
        #             0.0,
        #             0.0
        #         ],
        #         [
        #             -0.0,
        #             -0.0,
        #             -1.0,
        #             -0.0
        #         ]
        #     ]
        # }

    def get_train_images_and_ray(self, batch_size: int) -> Tuple[Tensor, RayBundle]:
        assert self.json is not None
        total_frames = len(self.json["frames"])
        frame_indices = np.random.randint(0, total_frames, size=batch_size)

        images = self.get_train_images(frame_indices)
        ray_bundle = self.get_train_ray_bundle(frame_indices)
        return images, ray_bundle

    def get_train_images(self, frame_indices: np.ndarray, transform: Callable = torchvision.transforms.ToTensor()) -> Tensor:
        """Get images by frame_indices, automatically cached to self.frames.

        Args:
            frame_indices (np.ndarray[Any, np.dtype[np.int_]]): indices of frames to get.
            transform (Callable, optional): transformation applied to PIL images. Defaults to torchvision.transforms.ToTensor().

        Returns:
            Tensor: images from frame_indices.
        """
        # TODO: probably should use pytorch dataloader?
        assert self.json is not None

        if self.frames is not None:
            return self.frames[frame_indices]

        self.frames = []
        w = int(self.json["w"] * self.scale_factor)
        h = int(self.json["h"] * self.scale_factor)
        for frame in self.json["frames"]:
            try:
                image_path = pathlib.Path(frame["file_path"]).resolve()
                image = PIL.Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                image_path = (self.output_dir / frame["file_path"]).resolve()
                image = PIL.Image.open(image_path).convert('RGB')
            image = image.resize((w, h)) # rescale
            image = transform(image)
            self.frames.append(image)
        self.frames = torch.stack(self.frames, dim=0).to(self.device)

        return self.frames[frame_indices]

    def get_nerf_panel_info(self, scale_factor=0.01) -> List[Dict[str, Any]]:
        assert self.json is not None
        # Used for:
        # self.set_output("render", {
        #     "dataset": [
        #         {
        #             "c2w": nerf_dataset.c2w,
        #             "image": nerf_dataset.image,
        #             "focal": nerf_dataset.focal,
        #         },
        #        ...
        #     ]
        # })
        focal = self.json["fl_x"]
        w = int(self.json["w"] * scale_factor)
        h = int(self.json["h"] * scale_factor)
        focal = focal * scale_factor

        # fov_x = 2 * np.arctan(self.json["w"] * scale_factor / (2 * focal))
        # fov_y = 2 * np.arctan(self.json["h"] * scale_factor / (2 * focal))
        # print(f"FOV remote: {math.degrees(fov_x)}, {math.degrees(fov_y)}")

        dataset: List[Dict[str, Any]] = []
        for frame in self.json["frames"]:
            try:
                image_path = pathlib.Path(frame["file_path"]).resolve()
                image = PIL.Image.open(image_path).convert('RGB')
            except FileNotFoundError:
                image_path = (self.output_dir / frame["file_path"]).resolve()
                image = PIL.Image.open(image_path).convert('RGB')
            image = image.resize((w, h)) # rescale

            # Convert to base64
            byte = io.BytesIO()
            image.save(byte, format="PNG")
            image = "data:image/png;base64," + base64.b64encode(byte.getvalue()).decode("utf-8")

            c2w = frame["transform_matrix"]
            dataset.append({
                "c2w": c2w,
                "image": image,
                "focal": focal,
            })
        return dataset

    def get_train_ray_bundle(
        self,
        frame_indices: np.ndarray,
    ) -> RayBundle:
        assert self.json is not None
        w_latent = self.json["w"]
        h_latent = self.json["h"]
        cx_latent = self.json["cx"]
        cy_latent = self.json["cy"]
        focal = self.json["fl_x"]
        if focal != self.json["fl_y"]:
            warnings.warn("fl_x and fl_y are not equal, using fl_x.")

        near_plane = self.near_plane
        far_plane = self.far_plane
        device = self.device

        # extrinsic
        c2w = [self.json["frames"][frame_indice]["transform_matrix"] for frame_indice in frame_indices] # [B, 4, 4]
        c2w = torch.tensor(c2w, dtype=torch.float32, device=device) # [B, 4, 4], [:, :3, 3] is rays_o
        rays_o = c2w[:, :3, 3] # [B, 3]
        forward_vector = -c2w[:, :3, 2] # [B, 3]

        # intrinsic
        fov_y: float = 2 * np.arctan(h_latent / (2 * focal)) # [1,]
        fov_x: float = 2 * np.arctan(w_latent / (2 * focal)) # [1,]

        # scale image without changing fov
        w_latent = int(w_latent * self.scale_factor)
        h_latent = int(h_latent * self.scale_factor)
        cx_latent = cx_latent * self.scale_factor
        cy_latent = cy_latent * self.scale_factor
        focal = focal * self.scale_factor

        rays_d = NeRFDataset._get_rays(
            batch_size=len(frame_indices),
            w_latent=w_latent,
            h_latent=h_latent,
            c2w=c2w,
            focal=focal,
            cx=cx_latent,
            cy=cy_latent,
            device=device,
        ) # [B, H, W, 3]
        rays_o = rays_o[:, None, None, :].expand(-1, h_latent, w_latent, -1) # [B, H, W, 3]

        projection = NeRFDataset._get_projection(
            focal=focal,
            h=h_latent, # TODO: don't know whether its h_img or h_latent
            w=w_latent,
            near_plane=near_plane,
            far_plane=far_plane,
            device=device,
        )

        mvp = NeRFDataset._get_mvp(
            projection=projection,
            c2w=c2w,
        ) # [B, 4, 4]

        ray_bundle = RayBundle(
            near_plane=near_plane,
            far_plane=far_plane,
            fov_x=fov_x,
            fov_y=fov_y,
            origins=rays_o,
            directions=rays_d,
            forward_vector=forward_vector,
            collider=self.collider,
            nears=None,
            fars=None,
            mvp=mvp,
            c2w=c2w,
        ) # [B, H, W, 3], [B, H, W, 3], [B,]

        return ray_bundle

    def preprocess(self, cleanup: bool) -> Dict[str, Any]:
        """Preprocess a folder of image or video files using COLMAP. Generating transform.json file.

        Args:
            cleanup (bool, optional): Whether to delete intermediate results. Defaults to True.

        Returns:
            Dict[str, Any]: The transform.json
        """
        if self.json is not None:
            return self.json
        if "transforms.json" in os.listdir(self.output_dir):
            with open(self.output_dir / "transforms.json", "r") as f:
                self.json = json.load(f)
            return self.json
        assert self.image_dir is not None
        images: List[pathlib.Path] = []
        if not (self.output_dir / "0").exists():
            images = self.to_images() # Convert videos to images first
            self.output_dir.mkdir(exist_ok=True)
            database_path = self.output_dir / "database.db"

            pycolmap.extract_features(database_path, self.image_dir, camera_mode=pycolmap.CameraMode.SINGLE)
            pycolmap.match_exhaustive(database_path)
            maps = pycolmap.incremental_mapping(database_path, self.image_dir, self.output_dir)
            maps[0].write(self.output_dir)
        self.json = self.colmap_to_json()

        if cleanup:
            os.remove(self.output_dir / "database.db")
            os.remove(self.output_dir / "database.db-shm")
            os.remove(self.output_dir / "database.db-wal")
            for image in images:
                os.remove(image)

        return self.json

    def to_our_convention(self, json: Dict[str, Any]) -> Dict[str, Any]:
        for frame_idx, _ in enumerate(json["frames"]):
            R_x = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ])

            S_y = np.array([
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])
            
            S_z = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ])

            T = np.dot(S_z, np.dot(S_y, R_x))
            json["frames"][frame_idx]["transform_matrix"] = np.dot(T, np.array(json["frames"][frame_idx]["transform_matrix"])).tolist()
        return json

    def to_center_of_mass(self, json: Dict[str, Any]) -> Dict[str, Any]:
        # Translate to center of mass
        c2w_tensors = torch.tensor([frame["transform_matrix"] for frame in json["frames"]], dtype=torch.float32)
        center_of_mass = torch.mean(c2w_tensors[:, :3, 3], dim=0)
        # Translate each camera origin by the negative center of mass
        for frame_idx, _ in enumerate(json["frames"]):
            for i in range(3):
                json["frames"][frame_idx]["transform_matrix"][i][3] -= center_of_mass[i].item()
        return json

    def to_unit_cube(self, json: Dict[str, Any]) -> Dict[str, Any]:
        # distances_from_origin = []
        # for frame_idx, _ in enumerate(json["frames"]):
        #     mat = np.array(json["frames"][frame_idx]["transform_matrix"])
        #     distances_from_origin.append(np.linalg.norm(mat[:3, 3]))
        # max_distance = max(distances_from_origin)
        # if max_distance == 0:
        #     return json
        # scale_factor = 1 / max_distance

        # for frame_idx, _ in enumerate(json["frames"]):
        #     for i in range(3):
        #         json["frames"][frame_idx]["transform_matrix"][i][3] *= scale_factor
        return json

    def to_images(self, n_frames=200) -> List[pathlib.Path]:
        """Search for videos in the image directory and convert them to images.

        Args:
            n_frames (int, optional): A rough estimate how many frames will be
                                      generated from a video, no guarantee that
                                      the exact number of frames will be generated.
                                      Defaults to 200.

        Returns:
            List[pathlib.Path]: Generated image paths.
        """
        assert self.image_dir is not None
        video_extensions = ['.mp4', '.avi', '.mov']
        generated_images = []

        for file in self.image_dir.iterdir():
            if file.suffix.lower() in video_extensions:
                video_frame_count = get_video_frame_count(video_path=file)
                frame_step = math.ceil(video_frame_count / n_frames)

                ffmpeg_command = [
                    'ffmpeg',
                    '-i',
                    str(file),
                    '-vf',
                    f"select='not(mod(n\,{frame_step}))'",
                    '-vsync',
                    'vfr',
                    str(self.image_dir / f"{file.stem}%05d.jpg"),
                ]
                subprocess.run(ffmpeg_command, check=True)

                # Estimate the number of frames to be generated
                estimated_frame_count = min(n_frames, video_frame_count)

                # Generate the list of expected image file names
                for i in range(estimated_frame_count):
                    frame_number = i * frame_step
                    image_name = f"{file.stem}{frame_number:05d}.jpg"
                    generated_images.append(self.image_dir / image_name)

        return generated_images

    def colmap_to_json(self,) -> Dict[str, Any]:
        """Converts COLMAP's cameras.bin and images.bin to a JSON file.
        Edited from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/process_data/colmap_utils.py#L386

        Args:
            recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
            output_dir: Path to the output directory.

        Returns:
            The number of registered images.
        """

        recon = pycolmap.Reconstruction(self.output_dir / "0")
        cam_id_to_camera = recon.cameras
        im_id_to_image = recon.images

        frames = []
        for im_id, im_data in im_id_to_image.items():
            # NB: COLMAP uses Eigen / scalar-first quaternions
            # * https://colmap.github.io/format.html
            # * https://github.com/colmap/colmap/blob/bf3e19140f491c3042bfd85b7192ef7d249808ec/src/base/pose.cc#L75
            # the `rotation_matrix()` handles that format for us.

            rotation = im_data.rotation_matrix()

            translation = im_data.tvec.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)

            rotation_colmap = c2w[:3, :3]
            translation_colmap = c2w[:3, 3]
            transform_matrix = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
            rotation_custom = np.dot(transform_matrix, rotation_colmap)
            translation_custom = np.dot(transform_matrix, translation_colmap)
            # Construct the transformed c2w matrix
            c2w_custom = np.zeros((4, 4))
            c2w_custom[:3, :3] = rotation_custom
            c2w_custom[:3, 3] = translation_custom
            c2w_custom[3, 3] = 1 # Homogeneous coordinate

            frame = {
                "file_path": str((self.image_dir / im_data.name).resolve()),
                "transform_matrix": c2w_custom.tolist(),
                "colmap_im_id": im_id,
            }
            frames.append(frame)

        if set(cam_id_to_camera.keys()) != {1}:
            raise RuntimeError("Only single camera shared for all images is supported.")
        out = parse_colmap_camera_params(cam_id_to_camera[1])
        out["frames"] = frames

        json_path = self.output_dir / "transforms.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=4)
        return out
