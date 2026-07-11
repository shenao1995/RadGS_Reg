<div align="center">

<p align="center"> <img src="assets/logo.png" width="250px"> </p>

[![project](https://img.shields.io/badge/project-homepage-green)](https://shenao1995.github.io/RadGS-Reg-Homepage/)
[![paper](https://img.shields.io/badge/paper-miccai-blue)](https://link.springer.com/chapter/10.1007/978-3-032-05114-1_44)

<h2>RadGS-Reg: Registering Spine CT with Biplanar X-rays via Joint 3D Radiative Gaussians Reconstruction and 3D/3D Registration</h2>

Vertebral-level CT/X-ray registration through joint 3D Radiative Gaussians reconstruction and 3D/3D registration.

![cover](assets/pipeline.jpg)

![reconstruction demo](assets/recon_results1.gif)

![registration demo](assets/reg_results.gif)

</div>

## Introduction

This repository contains the code used for RadGS-based vertebral reconstruction from biplanar X-rays/DRRs. The current README documents the reconstruction stage, including dataset generation, RadGS attribute prediction, volume querying, and reconstruction evaluation. The 3D/3D registration stage is still being organized and will be documented later.

The reconstruction network predicts per-Gaussian attributes from two input projections:

- `xyz`
- `scaling`
- `rotation`
- `density`

The predicted RadGS representation can be rendered into X-ray projections with `net_render` and queried into a voxel volume with `net_query`.

## 1. Installation

The CUDA rasterizer/voxelizer and `simple-knn` submodules follow the installation style of [R2-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian). A working CUDA-enabled PyTorch environment is required.

### 1.1 Create environment

```sh
conda create -n r2_gaussian python=3.9 -y
conda activate r2_gaussian

# Choose the PyTorch/CUDA version that matches your machine.
# The following is the upstream R2-Gaussian example for CUDA 11.8.
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
```

Install the Python packages used by this project:

```sh
pip install monai SimpleITK opencv-python scikit-learn plyfile open3d diffdrr tensorboard matplotlib imageio pyvista pyyaml
```

### 1.2 Build R2-Gaussian submodules

Clone/update submodules first:

```sh
git submodule update --init --recursive
```

Then compile the CUDA extensions:

```sh
pip install -e r2_gaussian/submodules/simple-knn
pip install -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

On Windows, install Visual Studio Build Tools and a CUDA toolkit compatible with the installed PyTorch version. If the build toolchain cannot be found, run the command prompt with the Visual Studio environment or set:

```bat
SET DISTUTILS_USE_SDK=1
```

R2-Gaussian also provides a Conda/environment-file based installation and optional TIGRE installation for its original data generation and initialization tools. See the upstream [R2-Gaussian README](https://github.com/Ruyi-Zha/r2_gaussian) for more details.

## 2. Dataset

The reconstruction code uses one `*.pickle` file and one `*.ply` file for each vertebra case.

```text
G:/GS_DATA/vertebra_dataset/
  0001_L1.pickle
  0001_L1.ply
  0002_L2.pickle
  0002_L2.ply
  ...
```

The `*.pickle` file stores the CT volume, scanner geometry, projection angles, and generated DRRs. The current generator writes fields such as:

```text
numTrain, numVal
DSD, DSO
nDetector, dDetector
nVoxel, dVoxel
offOrigin, offDetector
mode, totalAngle, startAngle
image
train.angles, train.projections
val.angles, val.projections
```

The companion `*.ply` file stores sampled 3D points from the foreground CT voxels. `GSDataset.py` reads the PLY coordinates directly as `gt_gs`; no extra coordinate normalization is applied in the dataset or renderer.

During training, `GSDataset.py` uses two projections as network input:

- AP-like view: `viewpoint_stack[24]`
- lateral-like view: `viewpoint_stack[0]`

All generated views are still used as projection supervision through `render_imgs`.

## 3. Generate Reconstruction Data

The data generation entry is `data_generation.py`. It uses DiffDRR to generate DRRs from cropped vertebra CT volumes and writes the NAF-style `*.pickle` files used by the reconstruction pipeline.

Before generation, edit these paths in `generate_net_data()`:

```python
vert_fold = "F:/BaiduNetdiskDownload/total_preprocess/resampled_vert"
out_fold = "F:/BaiduNetdiskDownload/total_preprocess/vertebra_dataset"
drr_save_fold = "data/drr_data"
```

Also check the filename filter:

```python
if "0" not in file:
```

This filter skips any filename containing `0`. Remove or modify it if you want to process every CT file.

The default geometry in `data_generation.py` is:

```text
SDD = 1100.0
DSO = 800.0
detector size = 256 x 256
detector spacing = 1.2 mm
numTrain = 50
numVal = 50
totalAngle = 180 degrees
```

Run:

```sh
python -c "from data_generation import generate_net_data; generate_net_data()"
```

For each input CT, the script saves:

- `<case>.pickle`: volume, scanner parameters, train/val DRRs
- `<case>.ply`: foreground point cloud sampled from CT voxels
- optional DRR preview PNGs in `drr_save_fold`

## 4. Running Reconstruction

### 4.1 Training

Set the dataset folder in `gs_train.py`:

```python
data_fold = "G:/GS_DATA/vertebra_dataset"
```

The current training configuration is:

```text
input views = 2
image size = 128 x 128
batch size = 4
num_points = 10000
base_channels = 32
fc_rank = 1024
optimizer = Adam, lr = 1e-5
scheduler = StepLR(step_size=20, gamma=0.5)
```

The loss combines:

```text
L1 projection loss
DSSIM projection loss
Chamfer loss on xyz
volume L1 loss from net_query
```

Current weights:

```text
lambda_dssim = 0.2
chamfer_weight = 0.5
lambda_vol = 0.05
vol_loss_size = 48
vol_loss_samples = 1
```

At the bottom of `gs_train.py`, enable the training entry:

```python
if __name__ == "__main__":
    model_fold = "DVGSNet_vol_input128_wR_cd0.5"
    log_dir = f"gs_model/{model_fold}"
    tb_writer = net_output_and_logger(log_dir)
    training(log_dir, tb_writer)
```

Then run:

```sh
python gs_train.py
```

Training saves:

```text
gs_model/<model_fold>/model1.pth
gs_model/<model_fold>/train_loss.jpg
gs_model/<model_fold>/train_ssim.jpg
gs_model/<model_fold>/events.out.tfevents.*
```

### 4.2 Reconstruction Inference

Use `gs_infer.py` to predict a volume from a trained model:

```sh
python gs_infer.py \
  --checkpoint gs_model/DVGSNet_vol_input128_wR_cd0.5/model1.pth \
  --data-fold G:/GS_DATA/vertebra_dataset \
  --output gs_model_output/vol_output
```

For a quick check:

```sh
python gs_infer.py \
  --checkpoint gs_model/DVGSNet_vol_input128_wR_cd0.5/model1.pth \
  --data-fold G:/GS_DATA/vertebra_dataset \
  --output gs_model_output/vol_output_check \
  --max-cases 1 \
  --no-npy
```

`gs_infer.py` automatically infers the model configuration from the checkpoint and uses the same volume convention as training:

```python
net_query(gaussians, offOrigin, nVoxel, sVoxel)
```

The GT volume is aligned with:

```python
torch.rot90(gt_volume, -2, dims=(-3, -2))
```

### 4.3 Evaluation

During training, validation reports 2D reconstruction metrics:

- mean 2D SSIM
- mean 2D PSNR

If TensorBoard logging is enabled, it also writes:

```text
val_images
reconstruction/slice-gt_pred_diff
val/vol_l1
val/psnr_2d
val/ssim_2d
```

Run TensorBoard with:

```sh
tensorboard --logdir gs_model
```

For post-training evaluation, `gs_infer.py` saves:

```text
gs_model_output/vol_output/
  metrics.csv
  nii/
    <case>_pred.nii.gz
    <case>_gt.nii.gz
  npy/
    <case>_pred.npy
    <case>_gt.npy
  slices/
    <case>_slices.png
```

`metrics.csv` contains:

```text
sample
vol_l1
psnr_3d
ssim_3d
ssim_3d_x
ssim_3d_y
ssim_3d_z
```

Use `--save-render-gif` if you also want GIFs of predicted and GT projections:

```sh
python gs_infer.py \
  --checkpoint gs_model/DVGSNet_vol_input128_wR_cd0.5/model1.pth \
  --data-fold G:/GS_DATA/vertebra_dataset \
  --output gs_model_output/vol_output \
  --save-render-gif
```

## 5. Current Scope and TODO

This README currently covers the reconstruction stage:

```text
cropped vertebra CT -> DRR/PLY dataset -> RadGS attribute prediction -> volume query/evaluation
```

The 3D/3D registration code and joint reconstruction-registration training protocol will be documented after the implementation is cleaned and stabilized.

## 6. Acknowledgement, License, and Citation

This project builds on ideas and code from:

- [R2-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian)
- Gaussian Splatting
- DiffDRR
- MONAI
- SimpleITK

If you use this project, please cite the RadGS-Reg paper and the upstream projects that this code depends on.
