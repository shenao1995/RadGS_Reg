import argparse
import csv
import glob
import json
import os
import os.path as osp
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from monai.data import DataLoader, list_data_collate
from sklearn.model_selection import train_test_split

from networks.DVGSNet import DVGSNet
from r2_gaussian.dataset.GSDataset import GSDataset
from r2_gaussian.gaussian import net_query, net_render
from r2_gaussian.utils.image_utils import metric_proj, metric_vol
from r2_gaussian.utils.plot_utils import projs_to_gif, show_two_slice
from tools.data_contract import align_gt_volume_to_query, read_split_pairs


def normalize_positive_volume(volume, eps=1e-6):
    """Convert HU-like CT data to a positive, per-volume shape target."""
    volume = torch.clamp(volume, min=0.0)
    max_value = volume.amax(dim=(-3, -2, -1), keepdim=True)
    return volume / max_value.clamp_min(eps)


def unwrap_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                checkpoint = value
                break

    if not isinstance(checkpoint, dict):
        raise TypeError("The checkpoint must be a state_dict or contain a state_dict.")

    if all(key.startswith("module.") for key in checkpoint.keys()):
        checkpoint = {
            key.removeprefix("module."): value
            for key, value in checkpoint.items()
        }

    return checkpoint


def infer_model_config(state_dict, fallback_img_size=128):
    num_properties = 11
    first_conv_key = next(
        (
            key
            for key, value in state_dict.items()
            if key.startswith("encoder.encoder.0")
            and key.endswith("weight")
            and getattr(value, "ndim", 0) == 4
        ),
        None,
    )
    input_ch = int(state_dict[first_conv_key].shape[1]) if first_conv_key else 2
    base_channels = int(state_dict[first_conv_key].shape[0]) if first_conv_key else 32

    if "fc_layers.4.weight" in state_dict:
        output_dim = int(state_dict["fc_layers.4.weight"].shape[0])
        fc_rank = int(state_dict["fc_layers.2.weight"].shape[0])
    elif "fc_layers.2.weight" in state_dict:
        output_dim = int(state_dict["fc_layers.2.weight"].shape[0])
        fc_rank = None
    else:
        raise KeyError("Cannot infer num_points from fc_layers weights.")

    if output_dim % num_properties != 0:
        raise ValueError(
            f"FC output dim {output_dim} is not divisible by {num_properties}."
        )
    num_points = output_dim // num_properties
    fc_hidden_dim = int(state_dict["fc_layers.0.weight"].shape[0])

    fc_input_dim = int(state_dict["fc_layers.0.weight"].shape[1])
    encoded_pixels = fc_input_dim / float(base_channels * 16)
    encoded_size = int(round(encoded_pixels ** 0.5))
    img_size = encoded_size * 16 if encoded_size > 0 else fallback_img_size

    return {
        "in_channels": input_ch,
        "base_channels": base_channels,
        "num_points": num_points,
        "img_size": img_size,
        "fc_hidden_dim": fc_hidden_dim,
        "fc_rank": fc_rank,
    }


def build_model_from_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = unwrap_state_dict(checkpoint)
    model_config = infer_model_config(state_dict)
    contract = {}
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("args"), dict):
        contract = checkpoint["args"]
    else:
        config_path = osp.join(osp.dirname(osp.abspath(checkpoint_path)), "config.json")
        if osp.isfile(config_path):
            import json
            with open(config_path) as handle:
                contract = json.load(handle)
    model_config["rotation_mode"] = contract.get("rotation_mode", "unit")
    model = DVGSNet(**model_config).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.training_contract = dict(contract)
    model.input_scale = float(contract.get("input_scale", 1.0))
    model.density_unit = float(contract.get("density_unit", 1.0))
    model.eval()
    return model, model_config


def get_sample_name(sample_names, index, fallback):
    value = sample_names
    # MONAI collates the per-case [name] field into [[name0, name1, ...]].
    while isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    if isinstance(value, (list, tuple)):
        value = value[index] if index < len(value) else fallback
    while isinstance(value, (list, tuple)):
        value = value[0] if value else fallback
    name = Path(str(value) if value is not None else fallback).name
    # A DICOM UID contains dots even after its .pickle suffix has been removed.
    for suffix in (".nii.gz", ".pickle", ".pkl", ".nii", ".npy", ".json"):
        if name.lower().endswith(suffix):
            return name[:-len(suffix)]
    return name


def write_volume(volume, save_path, spacing=None, voxel_to_ras_mm=None):
    array = volume.detach().cpu().numpy().astype(np.float32)
    if voxel_to_ras_mm is not None:
        # NanoDRR/query tensors are XYZ; SimpleITK arrays are ZYX and its world is LPS.
        affine = np.asarray(voxel_to_ras_mm, dtype=np.float64)
        if affine.shape != (4, 4) or not np.isfinite(affine).all():
            raise ValueError("Expected a finite 4x4 voxel-to-RAS-mm affine")
        lps_affine = np.diag([-1., -1., 1., 1.]) @ affine
        voxel_spacing = np.linalg.norm(lps_affine[:3, :3], axis=0)
        if (voxel_spacing <= 0).any():
            raise ValueError("Voxel spacing must be positive")
        direction = lps_affine[:3, :3] / voxel_spacing[None, :]
        if not np.allclose(direction.T @ direction, np.eye(3), atol=1e-5):
            raise ValueError("SimpleITK export requires an orthogonal voxel affine")
        image = sitk.GetImageFromArray(array.transpose(2, 1, 0).copy())
        image.SetSpacing(voxel_spacing.tolist())
        image.SetOrigin(lps_affine[:3, 3].tolist())
        image.SetDirection(direction.reshape(-1).tolist())
    else:
        # Preserve the historical array convention for datasets with no geometry metadata.
        image = sitk.GetImageFromArray(array)
        if spacing is not None:
            spacing = [float(item) for item in spacing.reshape(-1).detach().cpu().tolist()]
            if len(spacing) == 3:
                image.SetSpacing(spacing)
    sitk.WriteImage(image, save_path)


def save_volume_slices(gt_volume, pred_volume, save_path):
    slice_indices = np.linspace(0, gt_volume.shape[-1] - 1, 7)
    slice_indices = np.round(slice_indices).astype(int)[1:-1]
    panel = np.concatenate(
        [
            show_two_slice(
                gt_volume[..., slice_idx],
                pred_volume[..., slice_idx],
                f"slice {slice_idx} gt",
                f"slice {slice_idx} pred",
                vmin=0.0,
                vmax=1.0,
                save=True,
            )
            for slice_idx in slice_indices
        ],
        axis=0,
    )
    cv2.imwrite(save_path, cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))


def maybe_save_projection_gif(
    gaussian_params,
    batch_data,
    batch_index,
    output_dir,
    sample_name,
):
    device = gaussian_params["xyz"].device
    pred_projs = []
    gt_projs = []
    for r_idx in range(batch_data["render_imgs"].shape[1]):
        gt_image = batch_data["render_imgs"][batch_index, r_idx]
        render_pkg = net_render(
            gaussian_params,
            batch_data["world_view_transforms"][batch_index, r_idx].to(device),
            batch_data["full_proj_transforms"][batch_index, r_idx].to(device),
            batch_data["camera_centers"][batch_index, r_idx].to(device),
            int(gt_image.shape[-2]),
            int(gt_image.shape[-1]),
            float(batch_data["FovX"][batch_index, r_idx]),
            float(batch_data["FovY"][batch_index, r_idx]),
        )
        pred_projs.append(render_pkg["render"].squeeze().detach().cpu().numpy())
        gt_projs.append(gt_image.squeeze().detach().cpu().numpy())

    pred_projs = np.stack(pred_projs, axis=-1)
    gt_projs = np.stack(gt_projs, axis=-1)
    projs_to_gif(pred_projs, osp.join(output_dir, f"{sample_name}_pred.gif"))
    projs_to_gif(gt_projs, osp.join(output_dir, f"{sample_name}_gt.gif"))
    psnr_2d, _ = metric_proj(torch.from_numpy(gt_projs), torch.from_numpy(pred_projs), "psnr")
    ssim_2d, _ = metric_proj(torch.from_numpy(gt_projs), torch.from_numpy(pred_projs), "ssim")
    return psnr_2d, ssim_2d


def inference_method(
    checkpoint_path,
    data_fold="G:/GS_DATA/vertebra_dataset",
    pred_save_path="gs_model_output/vol_output",
    batch_size=1,
    test_size=0.2,
    split_seed=4,
    max_cases=None,
    save_gt=True,
    save_npy=True,
    save_slices=True,
    save_render_gif=False,
    vol_foreground_weight=4.0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = osp.abspath(checkpoint_path)
    pred_save_path = osp.abspath(pred_save_path)
    os.makedirs(pred_save_path, exist_ok=True)

    nii_dir = osp.join(pred_save_path, "nii")
    npy_dir = osp.join(pred_save_path, "npy")
    slice_dir = osp.join(pred_save_path, "slices")
    render_dir = osp.join(pred_save_path, "renders")
    for directory in (nii_dir,):
        os.makedirs(directory, exist_ok=True)
    if save_npy:
        os.makedirs(npy_dir, exist_ok=True)
    if save_slices:
        os.makedirs(slice_dir, exist_ok=True)
    if save_render_gif:
        os.makedirs(render_dir, exist_ok=True)

    imgs_path = sorted(glob.glob(osp.join(data_fold, "*.pickle")))
    pcs_path = sorted(glob.glob(osp.join(data_fold, "*.ply")))
    if not imgs_path:
        raise FileNotFoundError(f"No .pickle files found in {data_fold}")
    if len(imgs_path) != len(pcs_path):
        raise ValueError(
            f"Found {len(imgs_path)} pickle files but {len(pcs_path)} ply files."
        )

    if (Path(data_fold) / 'splits' / 'test.txt').is_file():
        imgs_test, pcs_test = read_split_pairs(data_fold, 'test')
    else:
        _, imgs_test, _, pcs_test = train_test_split(
            np.array(imgs_path), np.array(pcs_path), test_size=test_size, random_state=split_seed,
        )
    if max_cases is not None:
        imgs_test = imgs_test[:max_cases]
        pcs_test = pcs_test[:max_cases]

    val_set = GSDataset(imgs_path=imgs_test, pcs_path=pcs_test)
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        collate_fn=list_data_collate,
    )

    model, model_config = build_model_from_checkpoint(checkpoint_path, device)
    print(f"loaded checkpoint: {checkpoint_path}")
    print(f"model config: {model_config}")
    print(f"inference cases: {len(val_set)}")

    rows = []
    with torch.inference_mode():
        case_index = 0
        for batch_data in val_loader:
            input_projs = batch_data["input_imgs"].to(device)
            gaussian_batch = model(input_projs.squeeze(2) * model.input_scale)
            gaussian_batch["density"] = gaussian_batch["density"] / model.density_unit

            for b_idx in range(input_projs.size(0)):
                sample_name = get_sample_name(
                    batch_data.get("sample_name", []),
                    b_idx,
                    f"case_{case_index:04d}",
                )
                gaussians = {
                    key: value[b_idx].contiguous()
                    for key, value in gaussian_batch.items()
                }

                pred_volume = net_query(
                    gaussians,
                    batch_data["offOrigin"][b_idx].to(device).reshape(-1),
                    batch_data["nVoxel"][b_idx].to(device).reshape(-1),
                    batch_data["sVoxel"][b_idx].to(device).reshape(-1),
                )["vol"]
                gt_volume = align_gt_volume_to_query(
                    batch_data["vol_gts"][b_idx].to(device), batch_data['gt_is_query_aligned'][b_idx],
                ).squeeze()

                if gt_volume.shape != pred_volume.shape:
                    gt_volume = F.interpolate(
                        gt_volume[None, None],
                        size=pred_volume.shape,
                        mode="trilinear",
                        align_corners=False,
                    )[0, 0]

                pred_norm = normalize_positive_volume(pred_volume)
                gt_norm = normalize_positive_volume(gt_volume)
                volume_weights = 1.0 + vol_foreground_weight * gt_norm
                vol_l1 = (
                    volume_weights * torch.abs(pred_norm - gt_norm)
                ).sum().div(volume_weights.sum()).item()
                psnr_3d, _ = metric_vol(gt_norm.detach().cpu(), pred_norm.detach().cpu(), "psnr")
                ssim_3d, ssim_3d_axis = metric_vol(
                    gt_norm.detach().cpu(),
                    pred_norm.detach().cpu(),
                    "ssim",
                )

                spacing = batch_data["dVoxel"][b_idx]
                affine = None
                metadata_path = Path(data_fold) / f"{sample_name}.json"
                if metadata_path.is_file():
                    metadata = json.loads(metadata_path.read_text())
                    if metadata.get("schema") == "radgs-nanodrr-v1":
                        affine = metadata["geometry"]["voxel_to_centered_ras_mm"]
                write_volume(pred_norm, osp.join(nii_dir, f"{sample_name}_pred.nii.gz"), spacing, affine)
                if affine is not None:
                    write_volume(pred_volume, osp.join(nii_dir, f"{sample_name}_pred_mu.nii.gz"), voxel_to_ras_mm=affine)
                if save_gt:
                    write_volume(gt_norm, osp.join(nii_dir, f"{sample_name}_gt.nii.gz"), spacing, affine)
                    if affine is not None:
                        write_volume(gt_volume, osp.join(nii_dir, f"{sample_name}_gt_mu.nii.gz"), voxel_to_ras_mm=affine)
                if save_npy:
                    np.save(
                        osp.join(npy_dir, f"{sample_name}_pred.npy"),
                        pred_norm.detach().cpu().numpy().astype(np.float32),
                    )
                    if save_gt:
                        np.save(
                            osp.join(npy_dir, f"{sample_name}_gt.npy"),
                            gt_norm.detach().cpu().numpy().astype(np.float32),
                        )
                if save_slices:
                    save_volume_slices(
                        gt_norm,
                        pred_norm,
                        osp.join(slice_dir, f"{sample_name}_slices.png"),
                    )

                row = {
                    "sample": sample_name,
                    "vol_l1": vol_l1,
                    "psnr_3d": psnr_3d,
                    "ssim_3d": ssim_3d,
                    "ssim_3d_x": ssim_3d_axis[0],
                    "ssim_3d_y": ssim_3d_axis[1],
                    "ssim_3d_z": ssim_3d_axis[2],
                }
                if save_render_gif:
                    psnr_2d, ssim_2d = maybe_save_projection_gif(
                        gaussians,
                        batch_data,
                        b_idx,
                        render_dir,
                        sample_name,
                    )
                    row.update({"psnr_2d": psnr_2d, "ssim_2d": ssim_2d})

                rows.append(row)
                case_index += 1
                print(
                    f"[{case_index}/{len(val_set)}] {sample_name}: "
                    f"vol_l1={vol_l1:.4f}, psnr3d={psnr_3d:.3f}, ssim3d={ssim_3d:.3f}"
                )

    metrics_path = osp.join(pred_save_path, "metrics.csv")
    if rows:
        with open(metrics_path, "w", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved metrics: {metrics_path}")
        print(
            "mean metrics: "
            f"vol_l1={np.mean([row['vol_l1'] for row in rows]):.4f}, "
            f"psnr3d={np.mean([row['psnr_3d'] for row in rows]):.3f}, "
            f"ssim3d={np.mean([row['ssim_3d'] for row in rows]):.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict RadGS volumes.")
    parser.add_argument(
        "--checkpoint",
        default="gs_model/DVGSNet_vol_input128_wR_cd0.5/model1.pth",
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--data-fold",
        default="G:/GS_DATA/vertebra_dataset",
        help="Folder containing .pickle and .ply files.",
    )
    parser.add_argument(
        "--output",
        default="gs_model_output/vol_output",
        help="Folder for predicted NIfTI, NPY, slice PNGs, and metrics.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=4)
    parser.add_argument("--max-cases", type=int, default=None)
    parser.add_argument("--no-gt", action="store_true", help="Do not save GT volumes.")
    parser.add_argument("--no-npy", action="store_true", help="Do not save NPY arrays.")
    parser.add_argument("--no-slices", action="store_true", help="Do not save slice panels.")
    parser.add_argument("--save-render-gif", action="store_true", help="Also save 2D render GIFs.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    inference_method(
        checkpoint_path=args.checkpoint,
        data_fold=args.data_fold,
        pred_save_path=args.output,
        batch_size=args.batch_size,
        test_size=args.test_size,
        split_seed=args.split_seed,
        max_cases=args.max_cases,
        save_gt=not args.no_gt,
        save_npy=not args.no_npy,
        save_slices=not args.no_slices,
        save_render_gif=args.save_render_gif,
    )
