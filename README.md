<div align="center">

<img src="assets/logo.png" width="250" alt="RadGS-Reg logo">

[![Project](https://img.shields.io/badge/project-homepage-green)](https://shenao1995.github.io/RadGS-Reg-Homepage/)
[![Paper](https://img.shields.io/badge/paper-MICCAI-blue)](https://link.springer.com/chapter/10.1007/978-3-032-05114-1_44)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-RecM-yellow?logo=huggingface)](https://huggingface.co/shenao1995/RadGS-Reg-RecM)

<h2>RadGS-Reg: Registering Spine CT with Biplanar X-Rays via Joint 3D Radiative Gaussians Reconstruction and 3D/3D Registration</h2>

![Pipeline](assets/pipeline.jpg)

</div>

## 📰 News

- **2026.10.22:** We have fixed the RecM code and will upload the model and dataset to Hugging Face.
- **2026.01.30:** Code and data have been released.
- **2025.09.21:** Our work has been accepted to MICCAI 2025 **ORAL**!

## 📖 Introduction

RadGS-Reg is a framework for vertebral-level CT/X-ray registration that combines 3D Radiative Gaussians reconstruction from biplanar X-rays with 3D/3D registration.

If you find this repo useful, please give it a star ⭐ and consider citing our paper.

## 🛠️ Installation

Clone the repository:

```bash
git clone --recursive https://github.com/shenao1995/RadGS_Reg.git
cd RadGS_Reg
```

### 🐍 Reconstruction Environment

Use Linux with an NVIDIA GPU and a CUDA 11.8 toolkit.

```bash
conda create -n radgs python=3.11 pip -y
conda activate radgs

pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
pip install numpy==1.26.4 monai==1.3.2 SimpleITK opencv-python-headless \
  scikit-learn scikit-image plyfile open3d tensorboard tensorboardX \
  matplotlib imageio pyvista pyyaml torchinfo "moviepy<2" pandas tqdm ninja
```

Install the CUDA extensions and their source dependencies:

```bash
git clone https://gitlab.inria.fr/bkerbl/simple-knn.git \
  r2_gaussian/submodules/simple-knn
git clone --branch 0.9.9.8 https://github.com/g-truc/glm.git \
  r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/third_party/glm

pip install --no-build-isolation -e r2_gaussian/submodules/simple-knn
pip install --no-build-isolation -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
```

### 🩻 Data Generation Environment

Create a separate environment for NanoDRR:

```bash
conda create -n nanodrr python=3.11 pip -y
conda activate nanodrr

pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install "numpy==1.26.4" SimpleITK torchio plyfile Pillow \
  "nanodrr @ git+https://github.com/eigenvivek/nanodrr.git@f10354191f1e511cadee5d7cd4fc8ad05f5964c4"
```

## 🚀 Running

### 🧩 Reconstruction

Place the prepared dataset in `data/vertebra_dataset`. Provide `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`, each listing sample filenames relative to that directory. Keep all vertebrae from the same CT in one split.

Activate the reconstruction environment:

```bash
conda activate radgs
```

#### 🏋️ Training

Single GPU:

```bash
python gs_train.py \
  --data-dir data/vertebra_dataset \
  --output-dir outputs/RecM \
  --epochs 150
```

Alternatively, train on eight GPUs:

```bash
torchrun --standalone --nproc_per_node=8 gs_train.py \
  --data-dir data/vertebra_dataset \
  --output-dir outputs/RecM \
  --epochs 150
```

Monitor training with TensorBoard:

```bash
tensorboard --logdir outputs/RecM/tensorboard
```

#### 📊 Evaluation

Evaluate a trained checkpoint on the test split:

```bash
python -m scripts.evaluate_reconstruction \
  --checkpoint outputs/RecM/best_detail.pth \
  --data-dir data/vertebra_dataset \
  --split test \
  --output-dir outputs/RecM_eval
```

#### 🔍 Inference

Reconstruct and export volumes:

```bash
python gs_infer.py \
  --checkpoint outputs/RecM/best_detail.pth \
  --data-fold data/vertebra_dataset \
  --output outputs/RecM_infer
```

### 🧭 Registration

### 🤝 Joint Training

## 🩻 Generate Data

Generate L1–L5 samples from paired CT volumes and segmentation masks using [NanoDRR](https://github.com/eigenvivek/nanodrr). For [CTSpine1K](https://huggingface.co/datasets/alexanderdann/CTSpine1K), labels 20–24 correspond to L1–L5.

```bash
conda activate nanodrr

python data_generation.py \
  --ct-dir data/raw_data/volumes \
  --mask-dir data/raw_data/labels \
  --out-dir data/vertebra_dataset \
  --labels 20 21 22 23 24 \
  --spacing-mm 1.0 \
  --volume-size 128 \
  --max-source-spacing-mm 1.0 \
  --max-output-spacing-mm 1.0 \
  --device cuda \
  --backend triton
```

Each sample contains 50 DRRs over `[0°, 180°)`, with AP and lateral views as inputs. Samples with original or generated voxel spacing greater than 1 mm are excluded. After generation, prepare the CT-level split files described in **Running**.

## 🙏 Acknowledgement

We thank the authors of [R²-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian), [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting), [NanoDRR](https://github.com/eigenvivek/nanodrr), [MONAI](https://github.com/Project-MONAI/MONAI), and [SimpleITK](https://github.com/SimpleITK/SimpleITK) for sharing their work.

## 📄 License

Third-party components retain their respective licenses, including the [Gaussian rasterizer and voxelizer license](r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/LICENSE.md).

## 📚 Citation

If this repo helps you, please consider citing our work:

```bibtex
@inproceedings{shen2026radgsreg,
  title={RadGS-Reg: Registering Spine CT with Biplanar X-Rays via Joint 3D Radiative Gaussians Reconstruction and 3D/3D Registration},
  author={Shen, Ao and Fu, Xueming and Jiang, Junfeng and Zeng, Qiang and Tang, Ye and Chen, Zhengming and Nong, Luming and Wang, Feng and Zhou, Shaohua Kevin},
  booktitle={Medical Image Computing and Computer Assisted Intervention -- MICCAI 2025},
  series={Lecture Notes in Computer Science},
  volume={15968},
  pages={456--466},
  year={2026},
  publisher={Springer Nature Switzerland},
  doi={10.1007/978-3-032-05114-1_44}
}
```
