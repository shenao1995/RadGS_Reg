import torch
import torch.nn as nn
from monai.networks.blocks import Convolution
import math
# from simple_knn._C import distCUDA2
from torchinfo import summary


# def inverse_softplus(x):
#     return torch.log(torch.exp(x) - 1 + 1e-8)  # 加上 1e-8 防止数值不稳定


def inverse_softplus(x, beta=1):
    return torch.log(torch.exp(beta * x) - 1) / beta


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


# 定义 scaling_inverse_activation 函数
def scaling_inverse_activation(x, scale_min_bound, scale_max_bound):
    # 计算 inverse_softplus
    # inverse_sp = inverse_softplus(x)
    # 归一化到 [0, 1]
    inverse_scaling = (x - scale_min_bound) / (scale_max_bound - scale_min_bound)
    # 使用 sigmoid 激活
    scaling_activation = torch.sigmoid(inverse_scaling) * (scale_max_bound - scale_min_bound) + scale_min_bound
    return scaling_activation


class Encoder2D(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super(Encoder2D, self).__init__()
        self.encoder = nn.Sequential(
            Convolution(2, in_channels, base_channels, strides=1, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels, base_channels, strides=1, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels, base_channels * 2, strides=2, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels * 2, base_channels * 2, strides=1, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels * 2, base_channels * 4, strides=2, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels * 4, base_channels * 4, strides=1, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels * 4, base_channels * 8, strides=2, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels * 8, base_channels * 8, strides=1, kernel_size=3, act='RELU', norm='INSTANCE'),
            Convolution(2, base_channels * 8, base_channels * 16, strides=2, kernel_size=3, act='RELU',
                        norm='INSTANCE'),
            Convolution(2, base_channels * 16, base_channels * 16, strides=1, kernel_size=3, act='RELU',
                        norm='INSTANCE'),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class DVGSNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, num_points=10000, img_size=512):
        super(DVGSNet, self).__init__()
        self.encoder = Encoder2D(in_channels, base_channels)
        self.base_channels = base_channels
        self.num_points = num_points

        # 计算编码器输出特征图的空间尺寸
        H_transform = self.compute_encoder_output_size(img_size)
        W_transform = H_transform  # 假设输入图像是方形的

        # 展平特征图
        self.flatten = nn.Flatten()

        # 定义输出高斯球属性的全连接层
        # 假设我们需要输出位置(xyz)、尺度(scale)、旋转(四元数qx, qy, qz, qw)、密度(density)
        # 每个点的属性数量 = 3 (xyz) + 3 (scale) + 4 (rotation quaternion) + 1 (density) = 11
        self.num_properties = 11  # 根据需要输出的属性调整

        # 全连接层
        fc_input_dim = base_channels * 16 * H_transform * W_transform
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input_dim, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.num_points * self.num_properties)
        )

        # 定义激活函数
        self.density_inverse_activation = inverse_softplus
        self.density_activation = nn.Softplus()
        self.rotation_activation = torch.nn.functional.normalize
        # 定义尺度的最小和最大边界
        self.scale_min_bound = 0.001
        self.scale_max_bound = 1.0
        self.scaling_inverse_activation = lambda x: inverse_sigmoid(
            torch.relu((x - self.scale_min_bound) / self.scale_max_bound) + 1e-8
        )
        self.scaling_activation = (
            lambda x: torch.sigmoid(x) * self.scale_max_bound + self.scale_min_bound
        )

    def compute_encoder_output_size(self, input_size):
        H = input_size
        strides = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1]  # 编码器中每一层的步幅
        for stride in strides:
            H = math.ceil(H / stride)
        return H

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        # print(x.shape)
        # 将输出重塑为 (batch_size, num_points, num_properties)
        x = x.view(x.size(0), self.num_points, self.num_properties)

        # 将输出拆分为不同的属性
        xyz = x[:, :, :3]  # 位置，形状为 (batch_size, num_points, 3)
        scales = x[:, :, 3:6]  # 尺度，形状为 (batch_size, num_points, 3)
        rotations = x[:, :, 6:10]  # 旋转（四元数），形状为 (batch_size, num_points, 4)
        densities = x[:, :, 10:11]  # 密度，形状为 (batch_size, num_points, 1)
        # xyz = xyz * dVoxel - sVoxel / 2 + offOrigin  # 计算采样点的空间位置
        # 对密度应用 Softplus 激活函数
        densities = self.density_activation(densities)
        # dist2 = torch.clamp_min(distCUDA2(xyz.squeeze()), 0.001 ** 2)
        # print(dist2.shape)
        # scales = self.scaling_inverse_activation(torch.sqrt(dist2))[..., None].repeat(
        #     1, 3
        # )
        scales = self.scaling_activation(scales)
        # print(dist2.shape)
        # 计算缩放因子
        # scales = scaling_inverse_activation(torch.sqrt(dist2), scale_min_bound=self.scale_min_bound * torch.max(sVoxel),
        #                                     scale_max_bound=self.scale_max_bound * torch.max(sVoxel))[..., None].repeat(
        #     1, 3
        # )
        # 对尺度应用 scaling_inverse_activation 函数
        # scales = scaling_inverse_activation(
        #     scales,
        #     scale_min_bound=self.scale_min_bound,
        #     scale_max_bound=self.scale_max_bound
        # )

        # 对旋转四元数进行规范化
        rotations = self.rotation_activation(rotations)

        # 对 xyz 进行空间变换
        # 假设 xyz 的范围是 [0, 1]，需要调整到正确的体素索引范围
        # 计算体素网格的大小
        # grid_size = sVoxel / dVoxel  # (batch_size, 3)
        #
        # # 确保 grid_size, dVoxel, sVoxel, offOrigin 的形状为 (batch_size, 1, 3)
        # grid_size = grid_size.view(xyz.size(0), 1, 3)
        # dVoxel = dVoxel.view(xyz.size(0), 1, 3)
        # sVoxel = sVoxel.view(xyz.size(0), 1, 3)
        # offOrigin = offOrigin.view(xyz.size(0), 1, 3)

        # 将 xyz 调整到体素索引范围
        # xyz = xyz * grid_size  # 将 xyz 调整到体素索引范围

        # 按照公式进行空间变换
        # xyz = xyz * dVoxel - sVoxel / 2 + offOrigin  # 计算采样点的物理空间位置
        out_dict = {
            'xyz': xyz,
            'scaling': scales,
            'rotation': rotations,
            'density': densities
        }
        out_dict = make_contiguous(out_dict)
        # 返回属性字典
        return out_dict


def make_contiguous(tensor_dict):
    return {k: v.contiguous() for k, v in tensor_dict.items()}


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 4
    img_size = 128
    input_ch = 2
    num_points = 10000
    model = DVGSNet(in_channels=input_ch, base_channels=32, num_points=num_points, img_size=img_size).to(device)
    input_tensor = torch.randn(batch_size, input_ch, img_size, img_size).to(device)
    print(input_tensor.shape)
    output = model(input_tensor)
    print("Output keys:", output.keys())
    print("xyz shape:", output['xyz'].shape)  # 应为 (1, 10000, 3)
    summary(model, input_data=input_tensor)
