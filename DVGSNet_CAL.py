import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks import Convolution


def _atanh(x, eps=1e-6):
    x = x.clamp(min=-1 + eps, max=1 - eps)
    return 0.5 * torch.log((1 + x) / (1 - x))


def make_grid_anchors(num_points, bound=1.0):
    side = math.ceil(num_points ** (1.0 / 3.0))
    coords = torch.linspace(-bound, bound, side)
    try:
        zz, yy, xx = torch.meshgrid(coords, coords, coords, indexing="ij")
    except TypeError:
        zz, yy, xx = torch.meshgrid(coords, coords, coords)
    anchors = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
    if anchors.size(0) > num_points:
        idx = torch.linspace(0, anchors.size(0) - 1, num_points).long()
        anchors = anchors[idx]
    return anchors


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
            Convolution(2, base_channels * 8, base_channels * 16, strides=2, kernel_size=3, act="RELU", norm="INSTANCE"),
            Convolution(2, base_channels * 16, base_channels * 16, strides=1, kernel_size=3, act="RELU", norm="INSTANCE"),
        )

    def forward(self, x):
        return self.encoder(x)


class AdaptiveFeatureFusion2D(nn.Module):
    """Adaptive AP/LA fusion used only when CAL/two-view mode is enabled."""

    def __init__(self, channels):
        super().__init__()
        self.input_fc = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1),
            nn.GELU(),
        )
        self.weight_fc = nn.Conv2d(channels, 1, kernel_size=1)
        self.output_fc = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, feat_ap, feat_la):
        stacked_feats = torch.stack([feat_ap, feat_la], dim=0)
        mean = stacked_feats.mean(dim=0, keepdim=True).expand_as(stacked_feats)
        var = stacked_feats.var(dim=0, keepdim=True, unbiased=False).expand_as(stacked_feats)
        concat_feats = torch.cat([stacked_feats, mean, var], dim=2)

        global_ap = self.input_fc(concat_feats[0])
        global_la = self.input_fc(concat_feats[1])

        weight_ap = self.weight_fc(global_ap)
        weight_la = self.weight_fc(global_la)
        weights = F.softmax(torch.stack([weight_ap, weight_la], dim=0), dim=0)

        weighted_feat = global_ap * weights[0] + global_la * weights[1]
        return self.output_fc(weighted_feat)


class BAP(nn.Module):
    """Bilinear attention pooling with counterfactual attention."""

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, features, attentions):
        b, _, h, w = features.size()
        _, _, ah, aw = attentions.size()
        if ah != h or aw != w:
            attentions = F.interpolate(attentions, size=(h, w), mode="bilinear", align_corners=False)

        feature_matrix = torch.einsum("bmhw,bchw->bmc", attentions, features) / float(h * w)
        feature_matrix = self._signed_sqrt_l2(feature_matrix.reshape(b, -1))

        if self.training:
            fake_att = torch.empty_like(attentions).uniform_(0, 2)
        else:
            fake_att = torch.ones_like(attentions)
        counterfactual = torch.einsum("bmhw,bchw->bmc", fake_att, features) / float(h * w)
        counterfactual = self._signed_sqrt_l2(counterfactual.reshape(b, -1))
        return feature_matrix, counterfactual

    def _signed_sqrt_l2(self, x):
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + self.eps)
        return F.normalize(x, dim=-1)


class SharedGaussianHead(nn.Module):
    """Predicts Gaussian residual parameters with one shared MLP for all anchors."""

    def __init__(
        self,
        context_dim,
        query_dim=32,
        hidden_dim=256,
        num_properties=11,
        use_point_embeddings=True,
    ):
        super().__init__()
        self.use_point_embeddings = use_point_embeddings
        point_dim = query_dim if use_point_embeddings else 0
        self.anchor_encoder = nn.Sequential(
            nn.Linear(3, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
            nn.GELU(),
        )
        self.mlp = nn.Sequential(
            nn.Linear(context_dim + query_dim + point_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_properties),
        )

    def forward(self, context, anchors, point_embeddings=None):
        b = context.size(0)
        n = anchors.size(0)
        anchor_feat = self.anchor_encoder(anchors).unsqueeze(0).expand(b, -1, -1)
        context = context.unsqueeze(1).expand(-1, n, -1)
        inputs = [context, anchor_feat]
        if self.use_point_embeddings:
            if point_embeddings is None:
                raise ValueError("point_embeddings is required when use_point_embeddings=True")
            inputs.append(point_embeddings.unsqueeze(0).expand(b, -1, -1))
        return self.mlp(torch.cat(inputs, dim=-1))


class DVGSNet(nn.Module):
    """DVGSNet with Gaussian anchors/query embeddings and a shared MLP head.

    This removes the original huge Linear(4096, num_points * 11) layer. The
    number of parameters is nearly independent of num_points, except optional
    lightweight point embeddings and learnable anchors.
    """

    def __init__(
        self,
        in_channels=2,
        base_channels=32,
        num_points=10000,
        img_size=128,
        use_cal=False,
        view_channels=1,
        cal_attention_maps=32,
        cal_subtract_context=True,
        context_dim=512,
        query_dim=32,
        head_hidden_dim=256,
        use_point_embeddings=True,
        learnable_anchors=True,
        anchor_init="grid",
        anchor_bound=1.0,
        anchor_points=None,
        max_delta=0.1,
        bound_xyz=True,
        scale_min_bound=0.001,
        scale_max_bound=1.0,
    ):
        super().__init__()
        self.num_points = num_points
        self.num_properties = 11
        self.use_cal = use_cal
        self.view_channels = view_channels
        self.cal_subtract_context = cal_subtract_context
        self.learnable_anchors = learnable_anchors
        self.anchor_bound = anchor_bound
        self.max_delta = max_delta
        self.bound_xyz = bound_xyz
        self.scale_min_bound = scale_min_bound
        self.scale_max_bound = scale_max_bound

        feature_channels = base_channels * 16
        if use_cal:
            self.encoder_ap = Encoder2D(view_channels, base_channels)
            self.encoder_la = Encoder2D(view_channels, base_channels)
            self.fusion = AdaptiveFeatureFusion2D(feature_channels)
            self.attentions = Convolution(
                2,
                feature_channels,
                cal_attention_maps,
                strides=1,
                kernel_size=1,
                act="RELU",
                norm="INSTANCE",
            )
            self.bap = BAP()
            context_input_dim = cal_attention_maps * feature_channels
            self.context_head = nn.Sequential(
                nn.LayerNorm(context_input_dim),
                nn.Linear(context_input_dim, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim),
            )
        else:
            self.encoder = Encoder2D(in_channels, base_channels)
            self.context_pool = nn.AdaptiveAvgPool2d(1)
            self.context_head = nn.Sequential(
                nn.LayerNorm(feature_channels),
                nn.Linear(feature_channels, context_dim),
                nn.GELU(),
                nn.Linear(context_dim, context_dim),
            )

        anchors = self._init_anchors(num_points, anchor_init, anchor_bound, anchor_points)
        if learnable_anchors:
            self.anchor_params = nn.Parameter(_atanh(anchors / anchor_bound))
        else:
            self.register_buffer("anchors", anchors)

        if use_point_embeddings:
            self.point_embeddings = nn.Parameter(torch.randn(num_points, query_dim) * 0.02)
        else:
            self.point_embeddings = None

        self.gaussian_head = SharedGaussianHead(
            context_dim=context_dim,
            query_dim=query_dim,
            hidden_dim=head_hidden_dim,
            num_properties=self.num_properties,
            use_point_embeddings=use_point_embeddings,
        )
        self.density_activation = nn.Softplus()

    def forward(self, x, x_la=None):
        context = self._forward_context(x, x_la)
        anchors = self.get_anchors()
        raw = self.gaussian_head(context, anchors, self.point_embeddings)

        delta = torch.tanh(raw[:, :, :3]) * self.max_delta
        xyz = anchors.unsqueeze(0) + delta
        if self.bound_xyz:
            xyz = xyz.clamp(min=-self.anchor_bound, max=self.anchor_bound)

        scales = torch.sigmoid(raw[:, :, 3:6]) * (self.scale_max_bound - self.scale_min_bound)
        scales = scales + self.scale_min_bound
        rotations = F.normalize(raw[:, :, 6:10], p=2, dim=-1)
        densities = self.density_activation(raw[:, :, 10:11])

        return {
            "xyz": xyz.contiguous(),
            "scaling": scales.contiguous(),
            "rotation": rotations.contiguous(),
            "density": densities.contiguous(),
        }

    def get_anchors(self):
        if self.learnable_anchors:
            return torch.tanh(self.anchor_params) * self.anchor_bound
        return self.anchors

    def _forward_context(self, x, x_la=None):
        if self.use_cal:
            return self._forward_cal_context(x, x_la)
        feat = self.encoder(x)
        pooled = self.context_pool(feat).flatten(1)
        return self.context_head(pooled)

    def _forward_cal_context(self, x_ap, x_la=None):
        if x_la is None:
            if x_ap.size(1) < self.view_channels * 2:
                raise ValueError(
                    "CAL mode expects either two input tensors, or one tensor "
                    "whose channel dimension contains AP and LA views."
                )
            x_la = x_ap[:, self.view_channels:self.view_channels * 2]
            x_ap = x_ap[:, :self.view_channels]

        feat_ap = self.encoder_ap(x_ap)
        feat_la = self.encoder_la(x_la)
        fused_feat = self.fusion(feat_ap, feat_la)
        attention_maps = self.attentions(fused_feat)
        feature_matrix, counterfactual = self.bap(fused_feat, attention_maps)

        context = self.context_head(feature_matrix)
        if self.cal_subtract_context:
            context = context - self.context_head(counterfactual)
        return context

    @staticmethod
    def _init_anchors(num_points, anchor_init, anchor_bound, anchor_points):
        if anchor_points is not None:
            anchors = torch.as_tensor(anchor_points, dtype=torch.float32)
            if anchors.shape != (num_points, 3):
                raise ValueError(f"anchor_points must have shape ({num_points}, 3)")
            return anchors.clamp(-anchor_bound, anchor_bound)

        if anchor_init == "grid":
            return make_grid_anchors(num_points, anchor_bound)
        if anchor_init == "random":
            return torch.empty(num_points, 3).uniform_(-anchor_bound, anchor_bound)
        raise ValueError("anchor_init must be 'grid' or 'random'")


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    img_size = 128
    input_tensor = torch.randn(batch_size, 2, img_size, img_size).to(device)

    model = DVGSNet(
        in_channels=2,
        base_channels=32,
        num_points=10000,
        img_size=img_size,
        use_cal=True,
        view_channels=1,
        context_dim=512,
        query_dim=32,
        head_hidden_dim=256,
        max_delta=0.1,
    ).to(device)
    output = model(input_tensor)
    print("Output keys:", output.keys())
    print("xyz shape:", output["xyz"].shape)
