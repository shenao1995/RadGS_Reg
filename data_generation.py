import matplotlib.pyplot as plt
import torch
from diffdrr.drr import DRR
from diffdrr.data import read
# from diffdrr.visualization import plot_drr
import numpy as np
from diffdrr.pose import convert
import cv2
import os
import pickle
import SimpleITK as sitk
from PIL import Image
from r2_gaussian.utils.plot_utils import show_two_slice, show_one_volume
import open3d as o3d


def generate_net_data():
    vert_fold = 'F:/BaiduNetdiskDownload/total_preprocess/resampled_vert'
    out_fold = 'F:/BaiduNetdiskDownload/total_preprocess/vertebra_dataset'
    # vert_fold = 'data/vertebra_ct'
    # out_fold = 'F:/BaiduNetdiskDownload/total_vert'
    drr_save_fold = 'data/drr_data'
    for file in os.listdir(vert_fold):
        if '0' not in file:
            print(file)
            # outputPath = 'data/processed/{}.pickle'.format(os.path.split(ct_path)[-1].split('.')[0])
            outputPath = os.path.join(out_fold, file.split('.')[0] + '.pickle')
            ct_path = os.path.join(vert_fold, file)
            ct_img = sitk.ReadImage(ct_path)
            # raw_data = rotate_volume(ct_img)
            # print(raw_data.shape)
            size = ct_img.GetSize()
            # print(size)
            spacing = ct_img.GetSpacing()
            # print(spacing)
            # print(ct_img.GetOrigin())
            # Origin = ct_img.GetOrigin()
            # print(data_float.shape)
            # subject = read(ct_path, bone_attenuation_multiplier=10.5)
            subject = read(ct_path)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            SDD = 1100.0
            Xray_H = 256
            DELX = 1.2
            # print(DELX)
            drr = DRR(
                subject,  # An object storing the CT volume, origin, and voxel spacing
                sdd=SDD,  # Source-to-detector distance (i.e., focal length)
                height=Xray_H,  # Image height (if width is not provided, the generated DRR is square)
                delx=DELX,  # Pixel spacing (in mm)
            ).to(device)
            true_params = {
                "sdr": SDD,
                "alpha": 0.0,
                "beta": 0.0,
                "gamma": 0.0,
                "bx": 0.0,
                "by": 800.0,
                "bz": 0.0,
            }
            data = {'numTrain': 50, 'numVal': 50, 'DSD': SDD, 'DSO': true_params["by"], 'nDetector': [Xray_H, Xray_H],
                    'dDetector': [DELX, DELX], 'nVoxel': [size[0], size[1], size[2]],
                    'dVoxel': [spacing[0], spacing[1], spacing[2]], 'offOrigin': [0.0, 0.0, 0.0],
                    'offDetector': [0, 0], 'accuracy': 0.5, 'mode': 'cone', 'filter': None, 'totalAngle': 180.0,
                    'startAngle': 0.0, 'randomAngle': False, 'convert': True, 'rescale_slope': 1.0, 'rescale_intercept': 0,
                    'normalize': True, 'noise': True, 'possion_noise': 10000, 'gaussian_noise': [0, 10]}

            train_angles = np.linspace(0, data["totalAngle"] / 180 * np.pi, data["numTrain"] + 1)[:-1] + data[
                "startAngle"] / 180 * np.pi
            data["train"] = {"angles": train_angles}
            vol = subject.density.data[0].cpu().numpy().astype(np.float32)
            # vol = (vol - vol.min()) / (vol.max() - vol.min())
            data["image"] = vol
            points = np.argwhere(data["image"] > 0)
            nVoxel = np.array(size)
            nVoxel[2], nVoxel[0] = nVoxel[0], nVoxel[2]
            # print(nVoxel)
            dVoxel = np.array(spacing) / 1000
            # dVoxel[2], dVoxel[0] = dVoxel[0], dVoxel[2]
            offOrigin = np.array((0.0, 0.0, 0.0))
            sVoxel = nVoxel * dVoxel / 1000
            sampled_positions = points * dVoxel - sVoxel / 2 + offOrigin
            # 4. 将坐标转换为Open3D点云格式
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(sampled_positions)
            out_path = os.path.join(out_fold, file.replace('.nii.gz', '.ply'))
            o3d.io.write_point_cloud(out_path, pcd)
            # 5. 可视化点云
            # coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)  # size控制坐标轴的大小
            # o3d.visualization.draw_geometries([pcd, coordinate_frame])
            # plt.imshow(data["image"][:, :, subject.volume.data[0].shape[2] // 2])
            # plt.savefig(os.path.join('ct_orient_show', file.replace('.nii.gz', '.png')))
            # show_one_volume(data["image"])
            val_angles = np.sort(np.random.rand(data["numVal"]) * 180 / 180 * np.pi) + data["startAngle"] / 180 * np.pi
            data["val"] = {"angles": val_angles}
            train_projections = []
            count = 0
            for rotZ in train_angles:
                count += 1
                rotZ = rotZ + np.pi / 2
                rotations, translations, pose = get_initial_parameters(true_params, rotZ)
                # print(pose.matrix)
                img = drr(pose)
                proj_img = img.squeeze(0).cpu().numpy()
                # show_two_slice(proj_img.squeeze(), proj_img.squeeze(), title1='proj', title2='proj')
                proj_img = (proj_img - proj_img.min()) / (proj_img.max() - proj_img.min())
                if count == 1 or count == 25 or count == 50:
                    normalized_img = cv2.normalize(img.squeeze().cpu().numpy(), None, alpha=0, beta=1,
                                                   norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    # # 将归一化后的图像数据类型转换为 uint8
                    normalized_img = (normalized_img * 255).astype(np.uint8)
                    rgb_image = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)
                    cv2.imwrite(os.path.join(drr_save_fold, file.split('.')[0] + '_' + str(count).zfill(6) + '.png'), rgb_image)
                train_projections.append(proj_img)
            train_projections = np.concatenate(train_projections, axis=0)
            # data["train"]["projections"] = train_projections / 1000.0
            data["train"]["projections"] = train_projections
            val_projections = []
            for rotZ in val_angles:
                rotZ = rotZ + np.pi / 2
                rotations, translations, pose = get_initial_parameters(true_params, rotZ)
                # print(pose.matrix)
                img = drr(pose)
                proj_img = img.squeeze(0).cpu().numpy()
                proj_img = (proj_img - proj_img.min()) / (proj_img.max() - proj_img.min())
                # normalized_img = cv2.normalize(img.squeeze().cpu().numpy(), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                #                                dtype=cv2.CV_32F)
                # # 将归一化后的图像数据类型转换为 uint8
                # normalized_img = (normalized_img * 255).astype(np.uint8)
                # rgb_image = cv2.cvtColor(normalized_img, cv2.COLOR_GRAY2RGB)
                # cv2.imwrite(os.path.join(drr_save_fold, str(count).zfill(6) + '.png'), rgb_image)
                val_projections.append(proj_img)
            del drr
            val_projections = np.concatenate(val_projections, axis=0)
            # data["val"]["projections"] = val_projections / 1000.0
            data["val"]["projections"] = val_projections
            # print(train_angles.shape)
            # print(train_projections.shape)
            with open(outputPath, "wb") as handle:
                pickle.dump(data, handle, pickle.HIGHEST_PROTOCOL)


def get_initial_parameters(true_params, deltaZ):
    alpha = true_params["alpha"] + deltaZ
    beta = true_params["beta"]
    gamma = true_params["gamma"]
    bx = true_params["bx"]
    by = true_params["by"]
    bz = true_params["bz"]
    pose = pose_from_carm(by, bx, bz, alpha, beta, gamma).cuda()
    rotations, translations = pose.convert("euler_angles", "ZXY")
    return rotations, translations, pose


def pose_from_carm(sid, tx, ty, alpha, beta, gamma):
    rot = torch.tensor([[alpha, beta, gamma]], dtype=torch.float32)
    xyz = torch.tensor([[tx, sid, ty]], dtype=torch.float32)
    return convert(rot, xyz, parameterization="euler_angles", convention="ZXY")