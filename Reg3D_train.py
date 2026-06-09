import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import csv
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from monai.losses import SSIMLoss
from monai.data import create_test_image_3d, list_data_collate, decollate_batch, pad_list_data_collate
from RegFormer3D import Reg3DFormer
# from Reg3Donly import Reg3DFormer
from ReconFormer import RecFormer
from RecFormer_ResNet import RecResFormer
from Reg3D_infer import inference_method
from monai.transforms import (
    Activations,
    EnsureChannelFirstd,
    SqueezeDimd,
    AsDiscrete,
    Compose,
    LoadImaged,
    Resized,
    ScaleIntensityd,
)
from monai.visualize import plot_2d_or_3d_image
import matplotlib.pyplot as plt
from NCC3d import GNCCLoss3D, NormalizedCrossCorrelation3d
from monai.metrics import SSIMMetric, PSNRMetric
from r2_gaussian.utils.log_utils import net_output_and_logger
from r2_gaussian.utils.plot_utils import show_two_slice, show_two_volume
from sklearn.preprocessing import StandardScaler
import pickle


def model_engine():
    fold_path = 'gs_model_output/vol_output_real'
    pose_path = 'gs_model_output/pose_label.csv'
    pose_data = pd.read_csv(pose_path)
    mov_arr, fixed_arr = get_img_only(fold_path)
    # mov_arr, fixed_arr, pose_arr = get_img_pose(fold_path, pose_data)
    # SS = StandardScaler()
    # pose_arr = SS.fit_transform(pose_arr)

    # mov_train, mov_test, fixed_train, fixed_test, pose_train, pose_test = train_test_split(mov_arr, fixed_arr,
    #                                                                                        pose_arr,
    #                                                                                        test_size=0.2,
    #                                                                                        random_state=4)
    mov_train, mov_test, fixed_train, fixed_test = train_test_split(mov_arr, fixed_arr,
                                                                    test_size=0.2,
                                                                    random_state=15)
    # train_files = [{"moving": Moving, "fixed": Fixed, "pose": Pose}
    #                for Moving, Fixed, Pose in zip(mov_train, fixed_train, pose_train)]
    # val_files = [{"moving": Moving, "fixed": Fixed, "pose": Pose}
    #              for Moving, Fixed, Pose in zip(mov_test, fixed_test, pose_test)]
    train_files = [{"moving": Moving, "fixed": Fixed}
                   for Moving, Fixed in zip(mov_train, fixed_train)]
    val_files = [{"moving": Moving, "fixed": Fixed}
                 for Moving, Fixed in zip(mov_test, fixed_test)]
    model_fold = 'Recon_model8'
    log_dir = 'reg_model/{}'.format(model_fold)
    # tb_writer = net_output_and_logger(log_dir)
    # pickle.dump(SS, open('reg_model/{}/pose_scaler.pkl'.format(model_fold), 'wb'))
    # training(train_files, val_files, log_dir, tb_writer)
    inference_method(val_files, log_dir)


def get_img_only(data_path):
    mov_fold = 'gs_model_output/vol_output_mov'
    moving_list = []
    fixed_list = []
    for fixed in os.listdir(data_path):
        if 'pred' in fixed:
            fixed_list.append(os.path.join(data_path, fixed))
            # moving_list.append(os.path.join(mov_fold, fixed.replace('gt', 'mov')))
            moving_list.append(os.path.join(data_path, fixed.replace('pred', 'gt')))
    print(len(moving_list), len(fixed_list))
    return np.array(moving_list), np.array(fixed_list)


def get_img_pose(data_path, pose_pd):
    mov_fold = 'gs_model_output/vol_output_mov'
    moving_list = []
    fixed_list = []
    pose_list = []
    for index, row in pose_pd.iterrows():
        name = row['image_id']
        fixed_list.append(os.path.join(data_path, name.replace('mov', 'gt')))
        moving_list.append(os.path.join(mov_fold, name))
        pose_list.append([float(row['inv_rotZ_deg']), float(row['inv_rotY_deg']), float(row['inv_rotX_deg']),
                          float(row['inv_transZ_mm'] / 4), float(row['inv_transY_mm'] / 4),
                          float(row['inv_transX_mm'] / 4)])

    # for fixed in os.listdir(data_path):
    #     if 'pred' in fixed and os.path.exists(os.path.join(mov_fold, fixed.replace('gt', 'mov'))):
    #         fixed_list.append(os.path.join(data_path, fixed))
    #         moving_list.append(os.path.join(mov_fold, fixed.replace('gt', 'mov')))
    # print(len(moving_list), len(fixed_list))
    return np.array(moving_list), np.array(fixed_list), np.array(pose_list)


def training(train_files, val_files, model_dir, tensorboard_writer, StdS=None):
    keys = ["moving", "fixed"]
    img_size = 64
    # 训练集预处理
    train_transforms = Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            # CropForegroundd(keys=keys, source_key="img"),
            # 尺寸归一化
            Resized(keys=keys, spatial_size=(img_size, img_size, img_size), mode='trilinear',
                    align_corners=True),
            # 像素归一化
            # ConvertToMultiVertebraClassesd(keys="seg"),
            ScaleIntensityd(keys=keys),
            # CropForegroundd(keys=keys, source_key="img"),
            # RandScaleIntensityd(keys=keys[:-1], factors=0.1, prob=0.5),
            # RandRotated(keys=keys, range_z=np.pi / 6, prob=0.2, keep_size=True),
            # RandRotate90d(keys=keys, prob=0.2, spatial_axes=(0, 1)),
            # RandFlipd(keys=keys, spatial_axis=0, prob=0.2),
            # RandZoomd(keys=keys, min_zoom=0.6, max_zoom=1.2, prob=0.2),
        ]
    )
    # 验证集预处理
    val_transforms = Compose(
        [
            LoadImaged(keys=keys, ensure_channel_first=True, image_only=False),
            Resized(keys=keys, spatial_size=(img_size, img_size, img_size), mode='trilinear',
                    align_corners=True),
            ScaleIntensityd(keys=keys),
            # ScaleIntensityd(keys="dis"),
            # NormalizeIntensityd(keys=keys[:-1]),
            # ConcatItemsd(keys=keys[:-1], name="inputs"),
        ]
    )
    # create a training data loader
    # 加载训练集
    batch_size = 8
    train_ds = monai.data.CacheDataset(data=train_files, transform=train_transforms, num_workers=2)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        collate_fn=list_data_collate
    )
    # create a validation data loader
    # 加载测试集
    val_ds = monai.data.CacheDataset(data=val_files, transform=val_transforms, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=2, collate_fn=list_data_collate)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 单卡训练
    # model = RecFormer(
    #     in_channels=2,
    #     out_channels=1,
    #     img_size=img_size,
    #     feature_size=16,
    #     hidden_size=768,
    #     mlp_dim=3072,
    #     num_heads=12,
    #     proj_type="perceptron",
    #     norm_name="instance",
    #     res_block=True,
    # ).to(device)
    model = RecResFormer(
        in_channels=2,
        out_channels=1,
        spatial_dims=3
    ).to(device)
    # model = Reg3DFormer(
    #     in_channels=2,
    #     spatial_dims=3
    # ).to(device)

    # DCE_loss = DiceCELoss(sigmoid=True)
    MSE_loss = torch.nn.MSELoss()
    NCC_loss = NormalizedCrossCorrelation3d()
    Dice_loss = Reg_DiceLoss()
    SSIM_loss = SSIMLoss(spatial_dims=3)
    ssim_metric = SSIMMetric(spatial_dims=3)
    psnr_metric = PSNRMetric(max_val=1)
    Centroid_loss = CentroidLoss()  # 新增重心损失

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 50, gamma=0.5, last_epoch=-1)
    # start a typical PyTorch training
    epoch_num = 200
    # optimizer = torch.optim.Adam(model.parameters(), 0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)
    # 每隔2次验证一次
    val_interval = 2
    # 保存最佳的AUC
    best_metric = 0
    # 测试集dice最高时的epoch次数
    best_metric_epoch = -1
    # 用于保存训练集的loss
    epoch_loss_values = []
    # 用于保存测试集的loss
    val_loss_list = []
    # 用于保存每次验证时的Dice
    metric_values = []
    spacing = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64).to(device)
    # 训练循环
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        if epoch < epoch_num // 2:
            # 前半部分训练，ssim_loss权重从1到0.5
            weight_ssim = 1 - (epoch / (epoch_num // 2)) * 0.5  # 从1逐渐减小到0.5
            weight_mse = 1 - weight_ssim  # 为MSE损失分配剩余的权重
        else:
            # 后半部分训练，ssim_loss权重从0.5逐渐减小到0
            weight_ssim = 0.5 - ((epoch - epoch_num // 2) / (epoch_num // 2)) * 0.5  # 从0.5逐渐减小到0
            weight_mse = 1 - weight_ssim  # 为MSE损失分配剩余的权重
        for batch_data in train_loader:
            step += 1
            movings, fixeds = (
                batch_data["moving"].to(device),
                batch_data["fixed"].to(device),
            )
            # pose_gt = batch_data["pose"].to(device, dtype=torch.float32)
            # recon_fixed, pose_params, transformed_movs = model(movings, fixeds, spacing, StdS)
            # pose_params, transformed_movs = model(movings, fixeds, spacing, StdS)
            recon_fixed = model(movings, fixeds)
            # gt_transformed_movs = model.apply_rigid_transform(movings, pose_gt, spacing)
            # show_two_volume(gt_transformed_movs.squeeze()[1], fixeds.squeeze()[1], axis=0)
            # center_loss = Centroid_loss(transformed_movs, fixeds)
            # pose_loss = MSE_loss(pose_gt, pose_params)
            mse_loss = MSE_loss(recon_fixed, movings)
            # dice_loss = Dice_loss(transformed_movs, fixeds)  # 计算Dice损失
            ssim_loss = SSIM_loss(recon_fixed, movings)
            # ncc_loss = NCC_loss(transformed_movs, fixeds).mean()
            # total_loss = center_loss * 0.1 + ncc_loss * 0.2 + pose_loss
            total_loss = mse_loss + ssim_loss * 0.2
            # 损失回传
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size + 1}, "
                f"train_loss: {total_loss.item():.4f}, mse_loss: {mse_loss.item():.4f}, "
                f"ssim_loss: {ssim_loss.item():.4f}")
            # print(
            #     f"{step}/{len(train_ds) // train_loader.batch_size + 1}, "
            #     f"train_loss: {total_loss.item():.4f}, pose_loss: {pose_loss.item():.4f}")
            tensorboard_writer.add_scalar("train/train_loss", total_loss.item(), epoch_len * epoch + step)

        epoch_loss /= step
        # scheduler.step()
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # 开始验证测试集
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_step = 0
                for val_data in val_loader:
                    val_step += 1
                    val_movings, val_fixeds = (
                        val_data["moving"].to(device),
                        val_data["fixed"].to(device),
                    )
                    # val_recon_fixed, val_pose_params, val_transformed_movs = model(val_movings, val_fixeds, spacing, StdS)
                    # val_pose_params, val_transformed_movs = model(val_movings, val_fixeds, spacing,
                    #                                               StdS)
                    val_recon_fixed = model(val_movings, val_fixeds)
                    # val_ncc_loss = NCC_loss(val_transformed_movs, val_recon_fixed).mean()
                    # val_mse_loss = MSE_loss(val_transformed_movs, val_recon_fixed)
                    val_ssim_loss = SSIM_loss(val_recon_fixed, val_movings)
                    val_mse_loss = MSE_loss(val_recon_fixed, val_movings)
                    val_loss = val_ssim_loss + val_mse_loss
                    val_epoch_loss += val_loss.item()
                    val_epoch_len = len(val_ds) // val_loader.batch_size
                    tensorboard_writer.add_scalar("train/val_loss", val_loss.item(), val_epoch_len * epoch + val_step)
                    ssim_metric(y_pred=val_recon_fixed, y=val_movings)
                    psnr_metric(y_pred=val_recon_fixed, y=val_movings)
                val_epoch_loss /= val_step
                val_loss_list.append(val_epoch_loss)
                ssim_3d = ssim_metric.aggregate().item()
                psnr_3d = psnr_metric.aggregate().item()
                ssim_metric.reset()
                psnr_metric.reset()
                metric_values.append(ssim_3d)
                tensorboard_writer.add_scalar(
                    "val" + "/psnr_3d", psnr_3d, epoch + 1
                )
                tensorboard_writer.add_scalar(
                    "val" + "/ssim_3d", ssim_3d, epoch + 1
                )
                if len(val_movings.squeeze().shape) != 4:
                    val_transformed_cts_show = val_recon_fixed.squeeze().unsqueeze(0)
                    val_rencon_cts_show = val_movings.squeeze().unsqueeze(0)
                else:
                    val_transformed_cts_show = val_recon_fixed.squeeze()
                    val_rencon_cts_show = val_movings.squeeze()

                val_epoch_loss += val_loss.item()
                image_show_3d = np.concatenate(
                    [
                        show_two_slice(
                            val_transformed_cts_show[0, ..., i],
                            val_rencon_cts_show[0, ..., i],
                            f"slice {i} gt",
                            f"slice {i} pred",
                            vmin=val_transformed_cts_show[0, ..., i].min(),
                            vmax=val_transformed_cts_show[0, ..., i].max(),
                            save=True,
                        )
                        # for i in np.linspace(0, 128, 7).astype(int)[1:-1]
                        for i in np.linspace(0, val_rencon_cts_show[0, :].shape[2], 7).astype(int)[1:-1]
                    ],
                    axis=0,
                )
                image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
                tensorboard_writer.add_images(
                    "reconstruction/slice-gt_pred_diff",
                    image_show_3d,
                    global_step=epoch,
                )
                if ssim_3d >= best_metric:
                    best_metric = ssim_3d
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, 'model1.pth'))
                    print("saved new best metric sed_model")
                print(
                    f"current epoch: {epoch + 1} current mean ssim: {ssim_3d:.4f}"
                    f" current mean psnr: {psnr_3d:.4f}"
                    f"\nbest mean ssim: {best_metric:.4f} "
                    f"at epoch: {best_metric_epoch}"
                )
    print(
        f"train completed, best_metric: {best_metric:.4f} "
        f"at epoch: {best_metric_epoch}")
    # writer.close()
    # 绘制损失值曲线
    plt.title("Epoch Average Loss")
    x = [i + 1 for i in range(len(epoch_loss_values))]
    y = epoch_loss_values
    val_x = [val_interval * (i + 1) for i in range(len(val_loss_list))]
    val_y = val_loss_list
    plt.xlabel('Epoch')
    plt.plot(x, y)
    plt.plot(val_x, val_y)
    plt.legend(['Training Loss', 'Validation Loss'])
    # 保存曲线图像
    plt.savefig(os.path.join(model_dir, 'train_loss.jpg'), dpi=300)
    plt.close()
    # 绘制测试集的Dice训练次数的关系曲线
    plt.title("Val Mean SSIM")
    x = [val_interval * (i + 1) for i in range(len(metric_values))]
    y = metric_values
    plt.xlabel("epoch")
    plt.plot(x, y)
    # plt.legend()
    # 保存曲线图像
    plt.savefig(os.path.join(model_dir, 'train_ssim.jpg'), dpi=300)
    # plt.show()


class Reg_DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, threshold=0.0):
        super(Reg_DiceLoss, self).__init__()
        self.smooth = smooth
        self.threshold = threshold  # 新增：阈值，用于二值化

    def forward(self, pred, target):
        # 对预测结果应用sigmoid函数，将其转化为概率
        # pred = torch.sigmoid(pred)

        # 二值化：根据阈值0.5将预测值转为0或1
        pred = (pred > self.threshold).float()
        target = (target > self.threshold).float()  # 同样二值化目标值
        # show_two_volume(pred.squeeze()[0], target.squeeze()[0], axis=0)

        intersection = torch.sum(pred * target)
        dice_score = (2. * intersection + self.smooth) / (torch.sum(pred) + torch.sum(target) + self.smooth)
        return 1 - dice_score  # 返回Dice损失


def compute_centroid(img):
    """ 计算 3D 图像的重心，梯度可传播 """
    B, C, D, H, W = img.shape

    z = torch.arange(D, device=img.device).view(1, 1, D, 1, 1).expand(B, C, D, H, W)
    y = torch.arange(H, device=img.device).view(1, 1, 1, H, 1).expand(B, C, D, H, W)
    x = torch.arange(W, device=img.device).view(1, 1, 1, 1, W).expand(B, C, D, H, W)

    sum_intensity = torch.sum(img, dim=(2, 3, 4), keepdim=True) + 1e-5
    centroid_x = torch.sum(x * img, dim=(2, 3, 4), keepdim=True) / sum_intensity
    centroid_y = torch.sum(y * img, dim=(2, 3, 4), keepdim=True) / sum_intensity
    centroid_z = torch.sum(z * img, dim=(2, 3, 4), keepdim=True) / sum_intensity

    return torch.cat([centroid_x, centroid_y, centroid_z], dim=1).squeeze(-1).squeeze(-1).squeeze(-1)


class CentroidLoss(nn.Module):
    def __init__(self, img_size=(64, 64, 64)):
        super(CentroidLoss, self).__init__()
        self.max_dist = torch.sqrt(torch.tensor(img_size[0]**2 + img_size[1]**2 + img_size[2]**2))  # 计算最大可能的欧氏距离

    def forward(self, pred, target):
        centroid_pred = compute_centroid(pred)  # [B, 3]
        centroid_target = compute_centroid(target)  # [B, 3]

        euclidean_distance = torch.norm(centroid_pred - centroid_target, dim=1)  # [B]
        # centroid_loss = euclidean_distance / self.max_dist.to(pred.device)  # 归一化到 0-1

        return euclidean_distance.mean()  # 取 batch 平均


if __name__ == '__main__':
    model_engine()
