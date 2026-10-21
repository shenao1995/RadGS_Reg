import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution


def inverse_softplus(x, beta=1):
    return torch.log(torch.exp(beta * x) - 1) / beta


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


class Encoder2D(nn.Module):
    def __init__(self, in_channels, base_channels=64):
        super().__init__()
        self.encoder = nn.Sequential(
            Convolution(2, in_channels, base_channels, strides=1, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels, base_channels, strides=1, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels, base_channels * 2, strides=2, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels * 2, base_channels * 2, strides=1, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels * 2, base_channels * 4, strides=2, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels * 4, base_channels * 4, strides=1, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels * 4, base_channels * 8, strides=2, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels * 8, base_channels * 8, strides=1, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(
                2,
                base_channels * 8,
                base_channels * 16,
                strides=2,
                kernel_size=3,
                act="RELU",
                norm="INSTANCE",
            ),
            Convolution(
                2,
                base_channels * 16,
                base_channels * 16,
                strides=1,
                kernel_size=3,
                act="RELU",
                norm="INSTANCE",
            ),
        )

    def forward(self, x):
        return self.encoder(x)


class SharedPointHead(nn.Module):
    """Predict RadGS attributes with weights shared across all point queries."""

    def __init__(
        self,
        num_points,
        context_dim,
        point_feature_dim=128,
        hidden_dim=256,
    ):
        super().__init__()
        self.num_points = num_points
        self.point_queries = nn.Parameter(
            torch.empty(num_points, point_feature_dim)
        )
        nn.init.normal_(self.point_queries, mean=0.0, std=1.0)

        self.context_projection = nn.Linear(context_dim, hidden_dim)
        self.query_projection = nn.Linear(point_feature_dim, hidden_dim)
        self.shared_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # The trunk is shared by every point. Small attribute heads retain
        # specialization for quantities with different physical meanings.
        self.xyz_head = nn.Linear(hidden_dim, 3)
        self.scaling_head = nn.Linear(hidden_dim, 3)
        self.rotation_head = nn.Linear(hidden_dim, 4)
        self.density_head = nn.Linear(hidden_dim, 1)

    def forward(self, context):
        context_features = self.context_projection(context).unsqueeze(1)
        query_features = self.query_projection(self.point_queries).unsqueeze(0)
        point_features = F.relu(context_features + query_features, inplace=False)
        point_features = self.shared_mlp(point_features)

        return torch.cat(
            [
                self.xyz_head(point_features),
                self.scaling_head(point_features),
                self.rotation_head(point_features),
                self.density_head(point_features),
            ],
            dim=-1,
        )


class DVGSNetSharedHead(nn.Module):
    """DVGSNet variant with a point-wise shared RadGS output head.

    The encoder and all RadGS attribute activations/bounds match DVGSNet.py.
    Only the large point-specific final fully connected output is replaced.
    """

    def __init__(
        self,
        in_channels=1,
        base_channels=64,
        num_points=10000,
        img_size=512,
        context_dim=1536,
        point_feature_dim=128,
        head_hidden_dim=256,
    ):
        super().__init__()
        self.encoder = Encoder2D(in_channels, base_channels)
        self.base_channels = base_channels
        self.num_points = num_points
        self.num_properties = 11
        self.context_dim = context_dim
        self.point_feature_dim = point_feature_dim
        self.head_hidden_dim = head_hidden_dim

        feature_size = self.compute_encoder_output_size(img_size)
        encoder_output_dim = (
            base_channels * 16 * feature_size * feature_size
        )
        self.flatten = nn.Flatten()
        self.context_encoder = nn.Sequential(
            nn.Linear(encoder_output_dim, context_dim),
            nn.ReLU(inplace=True),
        )
        self.output_head = SharedPointHead(
            num_points=num_points,
            context_dim=context_dim,
            point_feature_dim=point_feature_dim,
            hidden_dim=head_hidden_dim,
        )

        # Keep these settings identical to the existing DVGSNet.
        self.density_inverse_activation = inverse_softplus
        self.density_activation = nn.Softplus()
        self.rotation_activation = F.normalize
        self.scale_min_bound = 0.001
        self.scale_max_bound = 1.0
        self.scaling_inverse_activation = lambda x: inverse_sigmoid(
            torch.relu(
                (x - self.scale_min_bound) / self.scale_max_bound
            )
            + 1e-8
        )
        self.scaling_activation = (
            lambda x: torch.sigmoid(x) * self.scale_max_bound
            + self.scale_min_bound
        )

    @staticmethod
    def compute_encoder_output_size(input_size):
        size = input_size
        strides = [1, 1, 2, 1, 2, 1, 2, 1, 2, 1]
        for stride in strides:
            size = math.ceil(size / stride)
        return size

    def forward(self, x):
        context = self.context_encoder(self.flatten(self.encoder(x)))
        raw_attributes = self.output_head(context)

        xyz = raw_attributes[:, :, :3]
        scales = raw_attributes[:, :, 3:6]
        rotations = raw_attributes[:, :, 6:10]
        densities = raw_attributes[:, :, 10:11]

        scales = self.scaling_activation(scales)
        rotations = self.rotation_activation(rotations)
        densities = self.density_activation(densities)

        return make_contiguous(
            {
                "xyz": xyz,
                "scaling": scales,
                "rotation": rotations,
                "density": densities,
            }
        )


# This alias lets gs_train.py switch networks by changing only the import.
DVGSNet = DVGSNetSharedHead


def make_contiguous(tensor_dict):
    return {key: value.contiguous() for key, value in tensor_dict.items()}


if __name__ == "__main__":
    model = DVGSNet(
        in_channels=2,
        base_channels=32,
        num_points=10000,
        img_size=128,
    )
    inputs = torch.randn(1, 2, 128, 128)
    outputs = model(inputs)

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"FP32 parameter size: {trainable_params * 4 / 1024 ** 2:.2f} MiB")
    for name, value in outputs.items():
        print(name, tuple(value.shape))
