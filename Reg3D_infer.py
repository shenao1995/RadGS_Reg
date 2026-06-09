import logging
import os
import sys
import tempfile
from glob import glob
import SimpleITK as sitk
import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.metrics import DiceMetric, MeanIoU, HausdorffDistanceMetric
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
# from RegFormer3D import Reg3DFormer
# from Reg3Donly import Reg3DFormer
from ReconFormer import RecFormer
from RecFormer_ResNet import RecResFormer

from r2_gaussian.utils.plot_utils import show_two_slice, show_two_volume, show_three_volume
from monai.metrics import SSIMMetric, PSNRMetric
# import pyvista as pv
from skimage.measure import marching_cubes
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay
from matplotlib import cm
from matplotlib import colors
from monai.transforms import (
    Activationsd,
    AsDiscreted,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    Resized,
    SaveImage,
    CropForegroundd,
    ScaleIntensityd,

)
from collections import Counter


def inference_method(inference_files, log_dir):
    img_size = 64
    keys = ["moving", "fixed"]
    infer_transforms = Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            Resized(keys=keys, spatial_size=(img_size, img_size, img_size), mode='trilinear',
                    align_corners=True),
            ScaleIntensityd(keys=keys),
        ]
    )
    # define dataset and dataloader
    val_ds = Dataset(data=inference_files, transform=infer_transforms)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=2, collate_fn=list_data_collate)
    out_fold = 'recon_out/' + os.path.split(log_dir)[-1]
    if not os.path.exists(out_fold):
        os.mkdir(out_fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RecFormer(
        in_channels=2,
        out_channels=1,
        img_size=img_size,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        proj_type="perceptron",
        norm_name="instance",
        res_block=True,
    ).to(device)
    # model = Reg3DFormer(
    #     in_channels=2,
    #     spatial_dims=3
    # ).to(device)
    # model = RecResFormer(
    #     in_channels=2,
    #     out_channels=1,
    #     spatial_dims=3
    # ).to(device)
    model.load_state_dict(torch.load(log_dir + '/model1.pth'))
    model.eval()
    spacing = torch.tensor([0.5, 0.5, 0.5]).to(device)
    ssim_metric = SSIMMetric(spatial_dims=3)
    psnr_metric = PSNRMetric(max_val=1)
    saver = SaveImage(output_dir=out_fold, output_ext=".nii.gz", output_postfix='',
                      separate_folder=False, writer='NibabelWriter', resample=True, mode='trilinear')
    ssim_values = []
    psnr_values = []
    with torch.no_grad():
        count = 0
        for val_data in val_loader:
            count += 1
            val_movings, val_fixeds = (
                val_data["moving"].to(device),
                val_data["fixed"].to(device),
            )
            sample_name = os.path.split(val_data['moving_meta_dict']['filename_or_obj'][0])[-1]

            # if 'dukemei_L3' in sample_name:
            print(sample_name)
            # val_recon_fixed, val_pose_params, val_transformed_movs = model(val_movings, val_fixeds, spacing)
            val_recon_fixed = model(val_movings, val_fixeds)
            # val_pose_params, val_transformed_movs = model(val_movings, val_fixeds, spacing)
            # pred_rotated = torch.rot90(val_recon_fixed.squeeze(), -1, dims=(0, 2))
            out_vol_pred = sitk.GetImageFromArray(val_recon_fixed.squeeze().cpu().numpy())
            out_vol_pred.SetSpacing([0.5, 0.5, 0.5])
            sitk.WriteImage(out_vol_pred,
                            os.path.join(out_fold, sample_name.replace('gt', 'pred')))
            # show_two_volume(val_movings.squeeze(), val_recon_fixed.squeeze(), axis=0)
            # visualize_3d(val_recon_fixed.squeeze(), val_movings.squeeze(), spacing.cpu().numpy())
            ssim_metric(y_pred=val_movings, y=val_recon_fixed)
            psnr_metric(y_pred=val_movings, y=val_recon_fixed)
            # saver(val_recon_fixed[0])
            std_ssim_3d = ssim_metric.aggregate().item()
            std_psnr_3d = psnr_metric.aggregate().item()
            print(std_ssim_3d)
            ssim_values.append(std_ssim_3d)
            psnr_values.append(std_psnr_3d)
        ssim_3d = ssim_metric.aggregate().item()
        psnr_3d = psnr_metric.aggregate().item()

        print(f"SSIM 3D: {ssim_3d * 100:.2f}±{np.std(ssim_values) * 100:.2f}")
        print(f"PSNR 3D: {psnr_3d:.2f}±{np.std(psnr_values):.2f}")
        ssim_metric.reset()
        psnr_metric.reset()


def visualize_3d(target, registered, spacing, threshold=None):
    plotter = pv.Plotter(line_smoothing=True)
    grid = pv.ImageData()
    grid.dimensions = np.array(registered.shape)  # 维度需要 +1
    grid.origin = (0, 0, 0)  # 设置原点
    grid.spacing = spacing  # 设置间距
    grid.point_data["difference"] = registered.cpu().numpy().flatten(order="F")  # 将差异数据添加到点数据
    masked_grid = grid.threshold([0.15, 1.0], scalars="difference")
    plotter.add_volume(masked_grid, cmap="viridis", clim=[0.15, 1])
    plotter.show()
    # 创建PyVista的spine_bbox和计算其中心
    # pv_spine_bbox, spine_center = create_pv_box_from_open3d_bbox(spine_bbox)
    # 计算图像的中心位置
    # D, H, W = target.shape
    # image_center = np.array([W * spacing[2] / 2, H * spacing[1] / 2, D * spacing[0] / 2])
    # 计算需要的平移向量
    # shift = spine_center - image_center
    # print(f"Shifting images by: {shift}")
    # registered = torch.where(registered < 0.15, np.nan, registered)
    # target = torch.where(target < 0.15, np.nan, target)
    # difference = np.abs(registered - target)
    # # # 将差异数据转换为 PyVista 的 ImageData 对象
    # grid = pv.ImageData()
    # grid.dimensions = np.array(registered.shape)  # 维度需要 +1
    # grid.origin = (0, 0, 0)  # 设置原点
    # grid.spacing = spacing  # 设置间距
    # grid.point_data["difference"] = difference.flatten(order="F")  # 将差异数据添加到点数据
    # # 设置差异阈值
    # masked_grid = grid.threshold([0.05, 1.0], scalars="difference")  # 过滤差异大于阈值的区域
    # # 创建绘图窗口
    # plotter = pv.Plotter()
    # # 添加热图
    # plotter.add_volume(
    #     target.cpu().numpy(),
    #     clim=[0, 1],
    #     cmap="viridis",  # 使用 coolwarm 颜色映射
    #     opacity=1
    # )
    # 设置标题和背景
    # plotter.add_title("3D Heatmap of Prediction vs Ground Truth", font_size=16)
    # plotter.set_background("white")
    # plotter.show()

    # registered = torch.where(registered < 0.15, np.nan, registered)
    # target = torch.where(target < 0.15, np.nan, target)
    # target_mesh = array2mesh(target.cpu().numpy(), spacing)
    # register_mesh = array2mesh(registered.cpu().numpy(), spacing)
    # plotter = pv.Plotter()
    # plotter.add_mesh(target_mesh, color="red")
    # plotter.add_mesh(register_mesh, color="gray")
    # plotter.show_grid()
    # plotter.add_axes()
    # plotter.show()


def array2mesh(array, space):
    grid = pv.ImageData()
    grid.dimensions = np.array(array.shape)  # 维度需要 +1
    grid.origin = (0, 0, 0)  # 设置原点
    grid.spacing = space  # 设置间距
    verbose = False
    grid.point_data["values"] = (
            array.flatten(order="F") > 0
    )
    try:
        mesh = grid.contour_labeled(smoothing=True, progress_bar=verbose)
    except AttributeError as e:
        raise AttributeError(
            f"{e}, ensure you are using pyvista>=0.43 and vtk>=9.3"
        )
    mesh.extract_largest(inplace=True, progress_bar=verbose)
    mesh.point_data.clear()
    mesh.cell_data.clear()
    mesh.smooth_taubin(
        n_iter=100,
        feature_angle=120.0,
        boundary_smoothing=False,
        feature_smoothing=False,
        non_manifold_smoothing=True,
        normalize_coordinates=True,
        inplace=True,
        progress_bar=verbose,
    )
    # mesh.point_data["values"] = (
    #     difference.flatten(order="F") > 0
    # )
    mesh.fill_holes(100, inplace=True, progress_bar=verbose)
    mesh.clean(inplace=True, progress_bar=verbose)
    return mesh
