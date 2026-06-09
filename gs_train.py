import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import csv
import torch
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
from monai.metrics import SSIMMetric, PSNRMetric
from monai.data import list_data_collate

from r2_gaussian.utils.plot_utils import show_two_slice
from r2_gaussian.gaussian import net_render
from r2_gaussian.utils.loss_utils import l1_loss, ssim
from r2_gaussian.utils.log_utils import net_output_and_logger
from r2_gaussian.dataset.GSDataset import GSDataset

import matplotlib.pyplot as plt
# from DVGSNet import DVGSNet
from DVGSNet_CAL import DVGSNet
from chamfer_loss import ChamferLoss
from gs_infer import inference_method


def training(model_dir, tensorboard_writer):
    ct_size = 128
    img_size = 128

    data_fold = 'G:/GS_DATA/vertebra_dataset'
    # data_fold = 'F:/BaiduNetdiskDownload/total_preprocess/real_vert'
    imgs_path = sorted(glob.glob(os.path.join(data_fold, "*.pickle")))
    pcs_path = sorted(glob.glob(os.path.join(data_fold, "*.ply")))
    imgs_train, imgs_test, pcs_train, pcs_test = (
        train_test_split(np.array(imgs_path), np.array(pcs_path), test_size=0.2, random_state=4))
    batch_size = 2
    train_ds = GSDataset(imgs_path=imgs_train, pcs_path=pcs_train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=list_data_collate)
    val_set = GSDataset(imgs_path=imgs_test, pcs_path=pcs_test)
    val_loader = DataLoader(val_set, batch_size=batch_size, collate_fn=list_data_collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 单卡训练
    target_size = (ct_size, ct_size, ct_size)  # 可以修改为任意2的幂次尺寸
    input_ch = 2
    num_points = 10000
    # model = DVGSNet(in_channels=input_ch, base_channels=32, num_points=num_points, img_size=img_size).to(device)
    model = DVGSNet(
        in_channels=input_ch,
        base_channels=32,
        num_points=num_points,
        img_size=img_size,
        use_cal=False,
        view_channels=1,
        context_dim=512,
        query_dim=32,
        head_hidden_dim=256,
        max_delta=0.1,
    ).to(device)
    # MSE_loss = torch.nn.MSELoss()
    CD_loss = ChamferLoss(chunk_size=1024, max_target_points=4096, squared=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.5, last_epoch=-1)
    # epoch_num = 150
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=1,
    #     gamma=np.power(1e-3, 1 / epoch_num)
    # )
    ssim_metric = SSIMMetric(spatial_dims=2)
    psnr_metric = PSNRMetric(max_val=25.0)
    # start a typical PyTorch training
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_num)

    lambda_dssim = 0.2
    loss_dl1 = 1
    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = []
    val_loss_list = []
    metric_values = []

    if os.path.exists(model_dir + '/model1.pth'):
        model.load_state_dict(torch.load(model_dir + '/model1.pth', map_location=device))
        epoch_num = 150
    else:
        epoch_num = 150
    for epoch in range(epoch_num):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{epoch_num}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            input_projs = batch_data["input_imgs"].to(device)
            # show_two_slice(input_projs[0, 0, :].squeeze(), input_projs[0, 1, :].squeeze(), title1='ap', title2='la')
            # gt_vols = batch_data["vol_gts"].to(device)
            gt_gs = batch_data["gt_gs"].to(device)
            gt_render_imgs = batch_data["render_imgs"].to(device)
            # offOrigin = batch_data["offOrigin"].to(device)
            # dVoxel = batch_data["dVoxel"].to(device)
            # sVoxel = batch_data["sVoxel"].to(device)
            # print(dVoxel.shape)
            # print(input_projs.shape)
            # input_projs = torch.cat([aps, las], dim=1)
            # outputs = model(input_projs.squeeze(2))
            # gaussian_params, vol_outputs = model(input_projs.squeeze(2), dVoxel, sVoxel, offOrigin)
            # gaussian_params = model(input_projs.squeeze(), dVoxel.squeeze(), sVoxel.squeeze(), offOrigin.squeeze())
            gaussian_params = model(input_projs.squeeze(2))
            out_xyz = gaussian_params['xyz']
            # out_density = gaussian_params['density']
            # out_scales = gaussian_params["scaling"]
            # out_rotations = gaussian_params["rotation"]
            # out_gs_params = torch.cat([out_xyz, out_density, out_scales, out_rotations], dim=-1)

            rendered_images = []
            gt_images = []

            for b_idx in range(gt_render_imgs.size(0)):
                gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in gaussian_params.items()}
                for r_idx in range(gt_render_imgs.shape[1]):
                    # Render X-ray projection
                    # print(gaussian_splat_batch)
                    render_pkg = net_render(gaussian_splat_batch,
                                            batch_data["world_view_transforms"][b_idx, r_idx],
                                            batch_data["full_proj_transforms"][b_idx, r_idx],
                                            batch_data["camera_centers"][b_idx, r_idx],
                                            256,
                                            256,
                                            batch_data["FovX"][b_idx, r_idx],
                                            batch_data["FovY"][b_idx, r_idx])
                    image, viewspace_point_tensor, visibility_filter, radii = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                    )
                    # Compute loss
                    gt_image = gt_render_imgs[b_idx, r_idx, ...]
                    # if step == 5:
                    # show_two_slice(gt_image.squeeze(), image.squeeze(), title1='gt', title2='pred')
                    rendered_images.append(image)
                    gt_images.append(gt_image)
            # mse_loss = MSE_loss(out_gs_params, gt_gs)
            cd_loss = CD_loss(out_xyz, gt_gs)
            rendered_images = torch.stack(rendered_images, dim=0)
            gt_images = torch.stack(gt_images, dim=0)
            render_loss = l1_loss(rendered_images, gt_images)
            loss_dssim = 1.0 - ssim(rendered_images, gt_images)
            # total_loss = render_loss * loss_dl1 + lambda_dssim * loss_dssim + mse_loss * loss_dmse
            total_loss = render_loss * loss_dl1 + loss_dssim * lambda_dssim + cd_loss * 0.5
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()
            epoch_loss += total_loss.item()
            print(
                f"{step}/{len(train_ds) // train_loader.batch_size + 1}, "
                f"train_loss: {total_loss.item():.4f}")
        epoch_loss /= step
        scheduler.step()
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
        # 开始验证测试集
        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_epoch_loss = 0
                val_step = 0

                image_show_2d = []
                for val_data in val_loader:
                    val_step += 1
                    # val_aps, val_las, val_ct = (
                    #     val_data["ap"].to(device),
                    #     val_data["la"].to(device),
                    #     val_data["ct"].to(device),
                    # )
                    val_input_projs = val_data["input_imgs"].to(device)
                    # val_ct = val_data["vol_gts"].to(device)
                    val_gt_render_imgs = val_data["render_imgs"].to(device)
                    # val_offOrigin = val_data["offOrigin"].to(device)
                    # val_dVoxel = val_data["dVoxel"].to(device)
                    # val_sVoxel = val_data["sVoxel"].to(device)
                    # val_gt_img = torch.concat([val_pc, val_dens], dim=1)
                    # val_projs = torch.cat([val_aps, val_las], dim=1)
                    # val_outputs = model(val_input_projs.squeeze(2))
                    # val_gaussian_params, val_outputs_val = model(val_input_projs.squeeze(2),
                    #                                              val_dVoxel, val_sVoxel, val_offOrigin)
                    val_gaussian_params = model(val_input_projs.squeeze())
                    val_pred_list = []
                    val_gt_list = []
                    for b_idx in range(val_gt_render_imgs.size(0)):
                        val_gaussian_splat_batch = {k: v[b_idx].contiguous() for k, v in val_gaussian_params.items()}
                        for r_idx in range(val_gt_render_imgs.shape[1]):
                            # Render X-ray projection
                            render_pkg = net_render(val_gaussian_splat_batch,
                                                    val_data["world_view_transforms"][b_idx, r_idx],
                                                    val_data["full_proj_transforms"][b_idx, r_idx],
                                                    val_data["camera_centers"][b_idx, r_idx],
                                                    256,
                                                    256,
                                                    val_data["FovX"][b_idx, r_idx],
                                                    val_data["FovY"][b_idx, r_idx])
                            pred_image = render_pkg["render"]
                            # Compute loss
                            gt_image = val_gt_render_imgs[b_idx, r_idx, ...]
                            # if best_metric > 0.85:
                            #     show_two_slice(gt_image.squeeze(), pred_image.squeeze(), title1='gt', title2='pred')
                            val_pred_list.append(pred_image)
                            val_gt_list.append(gt_image)
                            if tensorboard_writer and val_step == 1 and b_idx == 0 and r_idx % 2 == 0:
                                image_show_2d.append(
                                    torch.from_numpy(
                                        show_two_slice(
                                            gt_image[0],
                                            pred_image[0],
                                            f"{r_idx} gt",
                                            f"{r_idx} render",
                                            vmin=None,
                                            vmax=None,
                                            save=True,
                                        )
                                    )
                                )
                    # val_mse_loss = MSE_loss(val_outputs_val, val_ct)
                    val_pred_images_l = torch.stack(val_pred_list, dim=0)
                    val_gt_images_l = torch.stack(val_gt_list, dim=0)
                    val_render_loss = l1_loss(val_pred_images_l, val_gt_images_l)
                    val_loss_dssim = 1.0 - ssim(val_pred_images_l, val_gt_images_l)
                    # val_loss = val_render_loss * loss_dl1 + lambda_dssim * val_loss_dssim + val_mse_loss * loss_dmse
                    val_loss = val_render_loss * loss_dl1 + lambda_dssim * val_loss_dssim
                    # val_loss = val_mse_loss
                    val_epoch_loss += val_loss.item()
                    # mse_metric(val_outputs.squeeze(), val_ct.squeeze())
                    # ssim_metric(y_pred=val_outputs, y=val_ct)
                    # psnr_metric(y_pred=val_outputs, y=val_ct)
                    # dice_metric_batch(y_pred=val_outputs, y=val_labels)
                val_pred_images = torch.concat(val_pred_list, 0).unsqueeze(1)
                val_gt_images = torch.concat(val_gt_list, 0).unsqueeze(1)
                # psnr_2d, psnr_2d_projs = metric_proj(val_gt_images, val_pred_images, "psnr")
                # ssim_2d, ssim_2d_projs = metric_proj(val_gt_images, val_pred_images, "ssim")
                ssim_metric(y_pred=val_pred_images, y=val_gt_images)
                psnr_metric(y_pred=val_pred_images, y=val_gt_images)
                val_epoch_loss /= val_step
                val_loss_list.append(val_epoch_loss)
                # aggregate the final mean dice result
                # 计算Dice
                # mse_result = mse_metric.aggregate().item()
                psnr_2d = psnr_metric.aggregate().item()
                ssim_2d = ssim_metric.aggregate().item()
                if tensorboard_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tensorboard_writer.add_images(
                        "val_images",
                        image_show_2d,
                        global_step=epoch + 1,
                    )
                    tensorboard_writer.add_scalar(
                        "val" + "/psnr_2d", psnr_2d, epoch + 1
                    )
                    tensorboard_writer.add_scalar(
                        "val" + "/ssim_2d", ssim_2d, epoch + 1
                    )
                    # print(
                    #     f"[EPOCH {epoch + 1}] Evaluating: psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
                    # )
                metric_values.append(ssim_2d)
                if ssim_2d >= best_metric:
                    best_metric = ssim_2d
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(), os.path.join(model_dir, 'model_real2.pth'))
                    print("saved new best metric sed_model")
                print(
                    # f"current epoch: {epoch + 1} current mean 3D PSNR: {psnr_result:.4f} "
                    # f"current mean 3D SSIM: {ssim_result:.4f} "
                    f"current mean 2D SSIM: {ssim_2d} "
                    f"current mean 2D PSNR: {psnr_2d} "
                    f"\nbest mean SSIM: {best_metric:.4f} "
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


if __name__ == '__main__':
    # model_engine()
    model_fold = 'DVGSNet_input128_wR_cd0.5_modify'
    # model_fold = 'GSNet_input128_wR_model2.pth'
    log_dir = 'gs_model/{}'.format(model_fold)
    tb_writer = net_output_and_logger(log_dir)
    training(log_dir, tb_writer)
    # inference_method(log_dir + '/model1.pth')
