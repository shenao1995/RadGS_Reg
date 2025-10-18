import os
import glob
import torch
import matplotlib.pyplot as plt
from monai.config import print_config
from monai.data import Dataset, DataLoader, decollate_batch, list_data_collate
# from ReconNet import ResReconNet
from r2_gaussian.utils.plot_utils import show_two_volume, show_three_volume, show_two_slice, projs_to_gif
import numpy as np
from sklearn.model_selection import train_test_split
from monai.transforms import (
    Resize,
    ScaleIntensity
)
from collections import Counter
from monai.metrics import MSEMetric, SSIMMetric, PSNRMetric
from r2_gaussian.dataset.GSDataset import GSDataset

from r2_gaussian.gaussian import net_render, net_query
from DVGSNet import DVGSNet
import cv2
import pyvista as pv
import SimpleITK as sitk
from r2_gaussian.utils.image_utils import metric_vol, metric_proj


def inference_method(log_dir):
    # print(inference_files)
    ct_size = 128
    img_size = 128

    batch_size = 1
    data_fold = 'G:/GS_DATA/real_vert_drr'

    imgs_path = sorted(glob.glob(os.path.join(data_fold, "*.pickle")))
    pcs_path = sorted(glob.glob(os.path.join(data_fold, "*.ply")))
    imgs_train, imgs_test, pcs_train, pcs_test = (
        train_test_split(np.array(imgs_path), np.array(pcs_path), test_size=0.2, random_state=5))
    val_set = GSDataset(imgs_path=imgs_test, pcs_path=pcs_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=list_data_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ch = 2
    num_points = 10000
    pred_save_path = 'gs_model_output/vol_output_real'
    nvs_out_fold = 'gs_model_output/nvs_gif'

    # net = ResReconNet(in_channels=input_ch, out_channels=1, base_channels=32, depth=8, target_size=target_size,
    #                   img_size=img_size).to(device)
    net = DVGSNet(in_channels=input_ch, base_channels=32, num_points=num_points, img_size=img_size).to(device)
    ct_resize_transform = Resize(spatial_size=(128, 128, 128), mode='trilinear')  # Resize to 128x128
    intensity_scalar = ScaleIntensity()
    # mse_metric = MSEMetric()
    ssim_metric = SSIMMetric(spatial_dims=2)
    psnr_metric = PSNRMetric(max_val=1.0)
    vol_ssim_metric = SSIMMetric(spatial_dims=3)
    vol_psnr_metric = PSNRMetric(max_val=1.0)
    tv_vol_size = 128
    net.load_state_dict(torch.load(log_dir, map_location=device))
    net.eval()
    with torch.no_grad():
        count = 0

        for val_data in val_loader:
            count += 1
            val_input_projs = val_data["input_imgs"].to(device)
            val_ct = val_data["vol_gts"].to(device)
            val_gt_render_imgs = val_data["render_imgs"].to(device)
            # val_gaussian_params = net(val_input_projs.squeeze(2), val_dVoxel, val_sVoxel, val_offOrigin)
            val_gaussian_params = net(val_input_projs.squeeze(2))
            val_pred_list = []
            val_gt_list = []
            pred_vol_list = []
            gt_vol_list = []
            for b_idx in range(val_gt_render_imgs.size(0)):
                val_gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in val_gaussian_params.items()}
                single_proj_list = []
                gt_proj_list = []
                print(val_data["sample_name"][b_idx][0])
                sample_name = val_data["sample_name"][b_idx][0]
                for r_idx in range(val_gt_render_imgs.shape[1]):
                    # Render X-ray projection
                    # title_count += 1
                    # fig = plt.figure()
                    # ax = fig.add_subplot(111, projection='3d')
                    # ax.scatter(val_gaussian_splat_batch['xyz'].squeeze().cpu().numpy()[:, 0],
                    #            val_gaussian_splat_batch['xyz'].squeeze().cpu().numpy()[:, 1],
                    #            val_gaussian_splat_batch['xyz'].squeeze().cpu().numpy()[:, 2], c='r', marker='o')
                    # ax.set_xlabel('X')
                    # ax.set_ylabel('Y')
                    # ax.set_zlabel('Z')
                    # plt.show()
                    render_pkg = net_render(val_gaussian_splat_batch,
                                            val_data["world_view_transforms"][b_idx, r_idx],
                                            val_data["full_proj_transforms"][b_idx, r_idx],
                                            val_data["camera_centers"][b_idx, r_idx],
                                            256,
                                            256,
                                            val_data["FovX"][b_idx, r_idx],
                                            val_data["FovY"][b_idx, r_idx])
                    pred_image = render_pkg["render"]
                    # 生成体素网格
                    # print(voxel_grid.shape)
                    single_proj_list.append(pred_image.squeeze().cpu().numpy())
                    gt_image = val_gt_render_imgs[b_idx, r_idx, ...]
                    scaled_pred_proj = intensity_scalar(pred_image)
                    # resized_vol_pred = ct_resize_transform(scaled_pred_ct)
                    scaled_gt_proj = intensity_scalar(gt_image)
                    gt_proj_list.append(gt_image.squeeze().cpu().numpy())
                    # show_two_slice(gt_image.squeeze(), pred_image.squeeze(), title1='gt', title2='pred')
                    val_pred_list.append(scaled_pred_proj)
                    val_gt_list.append(scaled_gt_proj)
                # projs = np.stack(single_proj_list, axis=-1)
                # gt_projs = np.stack(gt_proj_list, axis=-1)
                # projs_to_gif(projs, os.path.join(nvs_out_fold, f"{sample_name}_pred.gif"))
                # projs_to_gif(gt_projs, os.path.join(nvs_out_fold, f"{sample_name}_gt.gif"))
            val_bbox = val_data["bbx"][b_idx]
            val_vol_dVoxel = val_data["dVoxel"][b_idx].squeeze()
            val_vol_nVoxel = val_data["nVoxel"][b_idx].squeeze()
            val_vol_sVoxel = val_vol_dVoxel * val_vol_nVoxel
            # Randomly get the tiny volume center
            val_vol_center = (val_bbox[0] + val_vol_sVoxel / 2) + (
                    val_bbox[1] - val_vol_sVoxel - val_bbox[0]
            ) * torch.rand(3)
            vol_pred = net_query(
                val_gaussian_splat_batch,
                val_vol_center,
                val_vol_nVoxel,
                val_vol_sVoxel,
            )["vol"]
            scaled_pred_ct = intensity_scalar(vol_pred)
            # resized_vol_pred = ct_resize_transform(scaled_pred_ct)
            scaled_gt_ct = intensity_scalar(val_ct.squeeze())
            pred_rotated = torch.rot90(scaled_pred_ct.squeeze(), -1, dims=(0, 2))
            gt_rotated = torch.rot90(scaled_gt_ct.squeeze(), 2, dims=(1, 2))
            # gt_rotated = torch.rot90(scaled_gt_ct.squeeze(), -2, dims=(0, 1))
            gt_rotated = torch.rot90(gt_rotated.squeeze(), 1, dims=(0, 2))

            out_vol_pred = sitk.GetImageFromArray(pred_rotated.cpu().numpy())
            out_vol_pred.SetSpacing([0.5, 0.5, 0.5])
            out_vol_gt = sitk.GetImageFromArray(gt_rotated.squeeze().cpu().numpy())
            out_vol_gt.SetSpacing([0.5, 0.5, 0.5])
            sitk.WriteImage(out_vol_gt, os.path.join(pred_save_path, '{}_gt.nii.gz').format(sample_name))
            sitk.WriteImage(out_vol_pred,
                            os.path.join(pred_save_path, '{}_pred.nii.gz').format(sample_name))
            pred_vol_list.append(scaled_pred_ct.unsqueeze(0))
            gt_vol_list.append(gt_rotated.squeeze().unsqueeze(0))
            # vol_ssim_metric(y_pred=scaled_pred_ct.unsqueeze(0).unsqueeze(0),
            #                 y=gt_rotated.squeeze().unsqueeze(0).unsqueeze(0))
            # vol_psnr_metric(y_pred=scaled_pred_ct.unsqueeze(0).unsqueeze(0),
            #                 y=gt_rotated.squeeze().unsqueeze(0).unsqueeze(0))

            # out_vol_img = show_two_slice(gt_rotated.squeeze().cpu().numpy()[64, :, :],
            #                              pred_rotated.cpu().numpy()[64, :, :], title1='gt', title2='pred',
            #                              save=True)
            # plt.subplot(1, 2, 1)
            # plt.title('GT Volume')
            # plt.imshow(gt_rotated.squeeze().cpu().numpy()[64, :, :])
            # plt.axis('off')
            # plt.savefig(os.path.join(pred_save_path, '{}_gt.png').format(sample_name),
            #             bbox_inches='tight', pad_inches=0)
            # plt.subplot(1, 2, 2)
            # plt.title('Pred Volume')
            # plt.imshow(pred_rotated.cpu().numpy()[64, :, :])
            # plt.axis('off')
            # plt.savefig(os.path.join(pred_save_path, '{}_pred.png').format(sample_name),
            #             bbox_inches='tight', pad_inches=0)
            # cv2.imwrite(os.path.join(pred_save_path, '{}_{}_vol.png').format(count, b_idx), out_vol_img)
            # np.save(os.path.join(pred_save_path, '{}_{}_vol_gt.npy').format(count, b_idx), scaled_gt_ct.squeeze().cpu().numpy())
            # np.save(os.path.join(pred_save_path, '{}_{}_vol_pred.npy').format(count, b_idx), scaled_pred_ct.squeeze().cpu().numpy())
            # show_two_volume(gt_rotated.squeeze(), pred_rotated,
            #                 title1='gt', title2='pred', axis=0)
            # cv2.imwrite(os.path.join(pred_save_path, '{}_{}_img.png').format(count, b_idx), pred_img)
            # plotter = pv.Plotter(window_size=[1000, 1000], line_smoothing=True, off_screen=False)
            # plotter.add_volume(scaled_pred_ct.squeeze().cpu().numpy(), cmap="viridis", opacity="linear")
            # plotter.show()
            # save_volume(pred_rotated,
            #             os.path.join(pred_save_path, '{}_{}_img.png').format(count, b_idx))
            # ssim_metric(y_pred=val_outputs, y=val_ct)
            # psnr_metric(y_pred=val_outputs, y=val_ct)
            val_pred_images = torch.concat(val_pred_list, 0).unsqueeze(1)
            val_gt_images = torch.concat(val_gt_list, 0).unsqueeze(1)
            val_pred_vols = torch.concat(pred_vol_list, 0).unsqueeze(1)
            val_gt_vols = torch.concat(gt_vol_list, 0).unsqueeze(1)
        ssim_metric(y_pred=val_pred_images, y=val_gt_images)
        psnr_metric(y_pred=val_pred_images, y=val_gt_images)
        vol_ssim_metric(y_pred=val_pred_vols, y=val_gt_vols)
        vol_psnr_metric(y_pred=val_pred_vols, y=val_gt_vols)
        print("SSIM metric:%.4f, PSNR:%.4f" % (ssim_metric.aggregate().item(),
                                               psnr_metric.aggregate().item()))
        print("3D SSIM metric:%.4f, 3D PSNR:%.4f" % (vol_ssim_metric.aggregate().item(),
                                                     vol_psnr_metric.aggregate().item()))


def save_volume(volume, save_path):
    cpos = [
        (-458.0015547298666, -207.26124611865254, 324.4699978427509),
        (129.02644270914504, 111.50694084289574, 98.55158287937994),
        (0.0, 0.0, 79.59633400474613),
    ]
    window_size = [800, 1000]
    colormap = "viridis"
    # volume = np.load(volume_path)
    half_size = volume.shape[0] // 3
    # volume[:half_size, :, :] = 0  # Set half to zero to show the inner structure
    clim = [0.0, 1.0]

    plotter = pv.Plotter(window_size=window_size, line_smoothing=True, off_screen=True)
    plotter.add_volume(volume.cpu().numpy(), cmap=colormap, opacity="linear", clim=clim)
    # plotter.add_volume(volume.cpu().numpy(), cmap=colormap, opacity="linear")
    plotter.camera_position = cpos
    plotter.show(screenshot=save_path)
