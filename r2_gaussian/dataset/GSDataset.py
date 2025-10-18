import glob
import os
import numpy as np
import torch

from r2_gaussian.dataset import NetScene
from r2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams
from random import randint
from r2_gaussian.utils.gaussian_utils import matrix_to_quaternion

# from .shared_dataset import SharedDataset
from torch.utils.data import Dataset
# from monai.data import Dataset
from random import sample
from torchvision import transforms
from monai.transforms import Resize, Orientation
from plyfile import PlyData


class GSDataset(Dataset):
    def __init__(self, imgs_path, pcs_path=None, state='train', num_points=10000):
        super().__init__()
        self.imgs_path = imgs_path
        self.pcs_path = pcs_path
        self.num_points = num_points  # 目标采样点数

        self.state = state
        self.resize_transform = Resize(spatial_size=(128, 128), mode='bilinear')  # Resize to 128x128
        # self.ct_resize_transform = Resize(spatial_size=(128, 128, 128), mode='trilinear')  # Resize to 128x128
        self.orient_transform = Orientation(axcodes='RAS')
        self.base_seed = 42

    def __len__(self):
        return len(self.imgs_path)

    def load_ply(self, path, index):
        # 使用 plyfile 加载 PLY 文件
        plydata = PlyData.read(path)

        # 提取 xyz 坐标
        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )

        # 提取 density
        # if "density" in plydata.elements[0].data.dtype.names:
        #     densities = np.asarray(plydata.elements[0]["density"])[..., np.newaxis]
        # else:
        #     densities = np.ones((xyz.shape[0], 1))  # 默认密度为1
        #
        # # 提取 scale 属性，假设属性名为 scale_1, scale_2, scale_3, ...
        # scale_names = [
        #     p.name
        #     for p in plydata.elements[0].properties
        #     if p.name.startswith("scale_")
        # ]
        # scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        # scales = np.zeros((xyz.shape[0], len(scale_names)))
        # for idx, attr_name in enumerate(scale_names):
        #     scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
        #
        # # 提取 rotation 属性，假设属性名为 rot_1, rot_2, rot_3, rot_4, ...
        # rot_names = [
        #     p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        # ]
        # rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        # rots = np.zeros((xyz.shape[0], len(rot_names)))
        # for idx, attr_name in enumerate(rot_names):
        #     rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将 numpy 数组转换为 PyTorch 张量
        xyz = torch.tensor(xyz, dtype=torch.float)
        # densities = torch.tensor(densities, dtype=torch.float)
        # scales = torch.tensor(scales, dtype=torch.float)
        # rots = torch.tensor(rots, dtype=torch.float)

        # 按最后一个维度拼接所有特征
        # gt = torch.cat([xyz, scales, rots, densities], dim=-1)  # (num_points, 11)
        gt = xyz
        # 随机采样 num_points 个点
        total_points = gt.shape[0]
        if total_points > self.num_points:
            # 创建一个基于样本索引的固定随机生成器
            generator = torch.Generator()
            generator.manual_seed(self.base_seed + index)  # 基于基础种子和样本索引生成固定种子
            indices = torch.randperm(total_points, generator=generator)[:self.num_points]
            gt = gt[indices]
        elif total_points < self.num_points:
            # 如果点数不足，则重复采样以达到 num_points
            repeat_times = self.num_points // total_points
            remainder = self.num_points % total_points
            gt = gt.repeat((repeat_times, 1))
            if remainder > 0:
                # 再次使用固定种子采样剩余的点
                generator = torch.Generator()
                generator.manual_seed(self.base_seed + index + 1)  # 使用不同的种子
                remainder_indices = torch.randperm(total_points, generator=generator)[:remainder]
                gt = torch.cat([gt, gt[remainder_indices]], dim=0)
        return gt  # 返回合并后的张量

    def make_poses_relative_to_first(self, images_and_camera_poses):
        inverse_first_camera = images_and_camera_poses["world_view_transforms"][0].inverse().clone()
        for c in range(images_and_camera_poses["world_view_transforms"].shape[0]):
            images_and_camera_poses["world_view_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["world_view_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["view_to_world_transforms"][c] = torch.bmm(
                                                images_and_camera_poses["view_to_world_transforms"][c].unsqueeze(0),
                                                inverse_first_camera.inverse().unsqueeze(0)).squeeze(0)
            images_and_camera_poses["full_proj_transforms"][c] = torch.bmm(
                                                inverse_first_camera.unsqueeze(0),
                                                images_and_camera_poses["full_proj_transforms"][c].unsqueeze(0)).squeeze(0)
            images_and_camera_poses["camera_centers"][c] = images_and_camera_poses["world_view_transforms"][c].inverse()[3, :3]
        return images_and_camera_poses

    def __getitem__(self, index):
        # 从路径加载场景
        source_path = self.imgs_path[index]
        model_path = os.path.split(self.imgs_path[index])[1].split('.')[0]
        scene = NetScene(model_path, source_path, shuffle=False)

        # 获取视点堆栈（包含相机的列表）
        viewpoint_stack = scene.getTrainCameras().copy()
        offOrigin = scene.scanner_cfg["offOrigin"]
        dVoxel = scene.scanner_cfg["dVoxel"]
        sVoxel = scene.scanner_cfg["sVoxel"]
        nVoxel = scene.scanner_cfg["nVoxel"]
        # print(scene.scene_scale)
        # 随机选择10张图像和对应的相机参数
        # selected_cameras = sample(viewpoint_stack, 50)
        selected_cameras = viewpoint_stack
        # 使用 linspace 生成等间隔的索引
        # indices = np.linspace(0, len(viewpoint_stack) - 1, 20, dtype=int)

        # 使用这些索引从列表中选取元素
        # selected_cameras = [viewpoint_stack[i] for i in indices]
        images_and_camera_poses = {
            "vol_gts": [],
            "input_imgs": [],
            "render_imgs": [],
            "gt_gs": [],
            "offOrigin": offOrigin,
            "dVoxel": dVoxel,
            "sVoxel": sVoxel,
            "nVoxel": nVoxel,
            "world_view_transforms": [],
            # "view_to_world_transforms": [],
            "full_proj_transforms": [],
            "camera_centers": [],
            "FovX": [],
            "FovY": [],
            "sample_name": [],
            "bbx": scene.bbox
        }
        # real xray 49
        # drr 24
        ap_img = viewpoint_stack[24]
        la_img = viewpoint_stack[0]
        # 应用 Resize 变换调整图像大小
        ap_resized = self.resize_transform(ap_img.original_image)
        la_resized = self.resize_transform(la_img.original_image)
        # images_and_camera_poses["view_to_world_transforms"].append(ap_img.view_world_transform)
        # images_and_camera_poses["view_to_world_transforms"].append(la_img.view_world_transform)
        images_and_camera_poses["input_imgs"].append(ap_resized)
        images_and_camera_poses["input_imgs"].append(la_resized)
        images_and_camera_poses["sample_name"].append(scene.model_path)
        images_and_camera_poses["vol_gts"].append(scene.vol_gt)

        # 遍历选择的相机，填充字典
        for camera in selected_cameras:
            # 获取图像
            image = camera.original_image
            # 将图像和相机参数添加到字典中
            images_and_camera_poses["render_imgs"].append(image)
            images_and_camera_poses["world_view_transforms"].append(camera.world_view_transform)

            images_and_camera_poses["full_proj_transforms"].append(camera.full_proj_transform)
            images_and_camera_poses["camera_centers"].append(camera.camera_center)
            images_and_camera_poses["FovX"].append(camera.FoVx)
            images_and_camera_poses["FovY"].append(camera.FoVy)

        # 加载并处理 'gt' 数据
        # if self.pcs_path:
        ply_path = self.pcs_path[index]
        gt = self.load_ply(ply_path, index)  # gt 是 (num_points, 11)
        # 将列表转换为张量（批量图像和相机参数）
        images_and_camera_poses["gt_gs"] = gt

        images_and_camera_poses["vol_gts"] = torch.stack(images_and_camera_poses["vol_gts"], dim=0)
        images_and_camera_poses["input_imgs"] = torch.stack(images_and_camera_poses["input_imgs"], dim=0)
        images_and_camera_poses["render_imgs"] = torch.stack(images_and_camera_poses["render_imgs"], dim=0)
        images_and_camera_poses["offOrigin"] = torch.tensor(images_and_camera_poses["offOrigin"])
        images_and_camera_poses["dVoxel"] = torch.tensor(images_and_camera_poses["dVoxel"])
        images_and_camera_poses["sVoxel"] = torch.tensor(images_and_camera_poses["sVoxel"])
        images_and_camera_poses["nVoxel"] = torch.tensor(images_and_camera_poses["nVoxel"])

        images_and_camera_poses["world_view_transforms"] = torch.stack(images_and_camera_poses["world_view_transforms"],
                                                                       dim=0)
        # images_and_camera_poses["view_to_world_transforms"] = torch.stack(
        #     images_and_camera_poses["view_to_world_transforms"], dim=0)
        images_and_camera_poses["full_proj_transforms"] = torch.stack(images_and_camera_poses["full_proj_transforms"],
                                                                      dim=0)
        images_and_camera_poses["camera_centers"] = torch.stack(images_and_camera_poses["camera_centers"], dim=0)
        images_and_camera_poses["FovX"] = torch.tensor(images_and_camera_poses["FovX"])
        images_and_camera_poses["FovY"] = torch.tensor(images_and_camera_poses["FovY"])

        # images_and_camera_poses = self.make_poses_relative_to_first(images_and_camera_poses)

        # images_and_camera_poses["source_cv2wT_quat"] = self.get_source_cw2wT(
        #     images_and_camera_poses["view_to_world_transforms"])
        return images_and_camera_poses
