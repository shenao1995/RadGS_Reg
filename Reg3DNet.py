import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.nets.vit import ViT
from monai.utils import ensure_tuple_rep
from typing import Union, Sequence, Tuple
from torchinfo import summary
import numpy as np
import math
import scipy


def euler_angles_to_rotation_matrix(angles):
    """
    将欧拉角转换为旋转矩阵。

    参数:
        angles (torch.Tensor): 形状为 (B, 3)，每行表示 (alpha, beta, gamma)。

    返回:
        torch.Tensor: 旋转矩阵，形状为 (B, 3, 3)。
    """
    alpha = angles[:, 0]
    beta = angles[:, 1]
    gamma = angles[:, 2]

    # 计算各旋转矩阵的元素
    cos_alpha = torch.cos(alpha)
    sin_alpha = torch.sin(alpha)
    cos_beta = torch.cos(beta)
    sin_beta = torch.sin(beta)
    cos_gamma = torch.cos(gamma)
    sin_gamma = torch.sin(gamma)

    # 旋转矩阵 Rz * Ry * Rx
    Rz = torch.stack([
        cos_gamma, -sin_gamma, torch.zeros_like(gamma),
        sin_gamma, cos_gamma, torch.zeros_like(gamma),
        torch.zeros_like(gamma), torch.zeros_like(gamma), torch.ones_like(gamma)
    ], dim=1).reshape(-1, 3, 3)

    Ry = torch.stack([
        cos_beta, torch.zeros_like(beta), sin_beta,
        torch.zeros_like(beta), torch.ones_like(beta), torch.zeros_like(beta),
        -sin_beta, torch.zeros_like(beta), cos_beta
    ], dim=1).reshape(-1, 3, 3)

    Rx = torch.stack([
        torch.ones_like(alpha), torch.zeros_like(alpha), torch.zeros_like(alpha),
        torch.zeros_like(alpha), cos_alpha, -sin_alpha,
        torch.zeros_like(alpha), sin_alpha, cos_alpha
    ], dim=1).reshape(-1, 3, 3)

    # 总旋转矩阵 R = Rz * Ry * Rx
    R = torch.bmm(Rz, torch.bmm(Ry, Rx))  # [B, 3, 3]

    return R


# 定义六自由度输出头
class PoseHead(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int = 256):
        """
        六自由度输出头。

        参数:
            in_features (int): 输入特征的维度。
            hidden_dim (int): 隐藏层的维度。默认值为256。
        """
        super(PoseHead, self).__init__()
        # self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        # self.fc2 = nn.Linear(hidden_dim, 6, bias=False)  # 输出6个参数 (3旋转 + 3平移)
        self.fc2 = nn.Linear(hidden_dim, 6)  # 输出6个参数 (3旋转 + 3平移)
        # self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.tanh(x)
        return x


# 自定义配准网络
class RecFormer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Union[Sequence[int], int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        proj_type: str = "conv",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = True,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        qkv_bias: bool = False,
        save_attn: bool = False,
    ) -> None:
        """
        自定义配准网络，基于MONAI的UNETR。

        参数:
            in_channels (int): 输入图像的通道数。
            out_channels (int): 输出图像的通道数。
            img_size (Union[Sequence[int], int]): 输入图像的尺寸。
            feature_size (int): 网络特征尺寸。默认为16。
            hidden_size (int): ViT的隐藏层大小。默认为768。
            mlp_dim (int): ViT中MLP层的维度。默认为3072。
            num_heads (int): ViT中的注意力头数。默认为12。
            proj_type (str): Patch embedding层类型。默认为"conv"。
            norm_name (Union[Tuple, str]): 规范化层类型。默认为"instance"。
            conv_block (bool): 是否使用卷积块。默认为True。
            res_block (bool): 是否使用残差块。默认为True。
            dropout_rate (float): Dropout率。默认为0.0。
            spatial_dims (int): 空间维度数。默认为3。
            qkv_bias (bool): 是否在自注意力模块中使用偏置。默认为False。
            save_attn (bool): 是否保存注意力权重。默认为False。
        """
        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise ValueError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size should be divisible by num_heads.")

        self.num_layers = 12
        img_size = ensure_tuple_rep(img_size, spatial_dims)
        self.patch_size = ensure_tuple_rep(16, spatial_dims)
        self.feat_size = tuple(img_d // p_d for img_d, p_d in zip(img_size, self.patch_size))
        self.hidden_size = hidden_size
        self.classification = False

        # ViT编码器
        self.vit = ViT(
            in_channels=in_channels,
            img_size=img_size,
            patch_size=self.patch_size,
            hidden_size=hidden_size,
            mlp_dim=mlp_dim,
            num_layers=self.num_layers,
            num_heads=num_heads,
            proj_type=proj_type,  # 已更改为 "conv"
            classification=self.classification,
            dropout_rate=dropout_rate,
            spatial_dims=spatial_dims,
            qkv_bias=qkv_bias,
            save_attn=save_attn,
        )

        # 解码器部分
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=spatial_dims, in_channels=feature_size, out_channels=out_channels)
        self.proj_axes = (0, spatial_dims + 1) + tuple(d + 1 for d in range(spatial_dims))
        self.proj_view_shape = list(self.feat_size) + [self.hidden_size]
        # 六自由度输出头
        self.pose_head = PoseHead(in_features=hidden_size)

    def proj_feat(self, x):
        new_view = [x.size(0)] + self.proj_view_shape
        x = x.view(new_view)
        x = x.permute(self.proj_axes).contiguous()
        return x

    def forward(self, moving, fixed):
        """
        前向传播。

        参数:
            fixed (torch.Tensor): 固定图像，形状 [B, C, D, H, W]
            moving (torch.Tensor): 移动图像，形状 [B, C, D, H, W]

        返回:
            reconstructed_moving (torch.Tensor): 重建的移动图像，形状 [B, C, D, H, W]
            transform_params (torch.Tensor): 6-DOF 参数，形状 [B, 6]
        """
        # 合并固定图像和移动图像作为双通道输入
        x_in = torch.cat([moving, fixed], dim=1)  # [B, 2C, D, H, W]

        # UNETR编码器部分
        x, hidden_states_out = self.vit(x_in)  # x: [B, N, hidden_size], hidden_states_out: List[Tensor]
        # 解码器部分
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[3]
        enc2 = self.encoder2(self.proj_feat(x2))
        x3 = hidden_states_out[6]
        enc3 = self.encoder3(self.proj_feat(x3))
        x4 = hidden_states_out[9]
        enc4 = self.encoder4(self.proj_feat(x4))
        dec4 = self.proj_feat(x)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        recon_fixed = self.out(out)  # [B, out_channels, D, H, W]

        return recon_fixed


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 1
    in_channels = 2  # 固定图像和移动图像合并为2通道
    out_channels = 1  # 根据需要调整
    img_size = (128, 128, 128)
    spacing = torch.tensor([0.5, 0.5, 0.5]).to(device)

    # 创建随机固定图像和移动图像
    fixed = torch.randn(batch_size, 1, *img_size).to(device)  # [B, C, D, H, W]
    moving = torch.randn(batch_size, 1, *img_size).to(device)  # [B, C, D, H, W]
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
    recon_fixed = model(fixed, moving)
    summary(model, input_data=(fixed, moving))
