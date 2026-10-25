"""Local residual point encoder and a rigid moving-to-fixed pose head.

Custom stage depths [3,4,6,3], inspired by residual point encoders. This is not
the official PointNeXt model and does not load image/voxel ResNet weights.
Coordinates stay in shared RAS millimetres; neither cloud is independently
centred or scaled. Surface extraction is detached; RecM keeps its differentiable
volume/image supervision in the alternating reconstruction substep.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from tools.point_registration import volume_surface_points, farthest_indices, gather
from tools.registration_geometry import centered_affine


class PoseHead(nn.Module):
    def __init__(self, in_features, hidden_dim=256, activation='relu'):
        super().__init__()
        if activation not in ('relu','prelu'):
            raise ValueError('Pose-head activation must be relu or prelu')
        nonlinearity = nn.ReLU(inplace=True) if activation=='relu' else nn.PReLU(hidden_dim,init=.25)
        self.layers = nn.Sequential(nn.Linear(in_features, hidden_dim), nonlinearity,
                                    nn.Linear(hidden_dim, 6))
        # Start at identity; no volume synthesis or deformable displacement field.
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features):
        return self.layers(features)


class LocalResidualBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.edge=nn.Sequential(nn.Linear(width+3,width),nn.LayerNorm(width),nn.GELU())
        self.output=nn.Sequential(nn.Linear(width,width),nn.LayerNorm(width))
        self.residual_scale=nn.Parameter(torch.full((width,),.1))

    def forward(self,features,neighbors,offset):
        local=gather(features,neighbors)-features[:,:,None,:]
        message=self.edge(torch.cat([local,offset.to(features.dtype)],-1)).amax(2)
        return F.gelu(features+self.residual_scale*self.output(message))


class PointResidualEncoder(nn.Module):
    def __init__(self,depths=(3,4,6,3),widths=(32,64,128,256),neighbors=16):
        super().__init__();self.neighbors=neighbors
        self.stem=nn.Sequential(nn.Linear(7,widths[0]),nn.LayerNorm(widths[0]),nn.GELU())
        self.transitions=nn.ModuleList([nn.Identity()]+[nn.Linear(a,b) for a,b in zip(widths[:-1],widths[1:])])
        self.stages=nn.ModuleList([nn.ModuleList([LocalResidualBlock(w) for _ in range(d)]) for d,w in zip(depths,widths)])

    def forward(self,cloud):
        xyz=cloud[...,:3].float()
        raw=torch.cat([xyz/96,cloud[...,3:4].float()/.04,cloud[...,4:].float()],-1)
        features=self.stem(raw)
        for i,(transition,blocks) in enumerate(zip(self.transitions,self.stages)):
            if i:
                index=farthest_indices(xyz,max(16,cloud.shape[1]//(4**i)))
                xyz=gather(xyz,index);features=gather(features,index)
            features=transition(features)
            with torch.no_grad(),torch.autocast(device_type=xyz.device.type,enabled=False):
                distances=torch.cdist(xyz.float(),xyz.float())
                nearest=distances.topk(min(self.neighbors,xyz.shape[1]),largest=False).indices
                offset=(gather(xyz,nearest)-xyz[:,:,None,:])/(4.*2**i)
            for block in blocks:features=block(features,nearest,offset)
        return xyz,features


class PointRegM(nn.Module):
    def __init__(self,input_size=192,num_points=2048,surface_threshold=.005,
                 rotation_limit_deg=10.,translation_limit_mm=20.,depths=(3,4,6,3),neighbors=16,
                 architecture='point_residual34'):
        super().__init__()
        if architecture!='point_residual34':raise ValueError('Unknown point architecture')
        self.config=dict(architecture=architecture,input_size=input_size,num_points=num_points,surface_threshold=surface_threshold,
                         rotation_limit_deg=rotation_limit_deg,translation_limit_mm=translation_limit_mm,depths=list(depths),neighbors=neighbors)
        self.input_size=input_size;self.num_points=num_points;self.surface_threshold=surface_threshold
        self.encoder=PointResidualEncoder(depths=depths,neighbors=neighbors)
        self.cross_attention=nn.MultiheadAttention(256,4,batch_first=True)
        self.fusion=nn.Sequential(nn.Linear(515,256),nn.LayerNorm(256),nn.GELU())
        self.pose_head=PoseHead(512+3+3+3+9+9)
        # Near-identity while allowing encoder gradients on the very first step.
        nn.init.normal_(self.pose_head.layers[-1].weight,std=1e-3)
        self.register_buffer('pose_scale',torch.tensor([math.radians(rotation_limit_deg)]*3+[translation_limit_mm]*3))

    def clouds(self,moving,fixed):
        affine=centered_affine(moving.shape[-3:],192./moving.shape[-1],device=moving.device)
        return (volume_surface_points(moving,affine,self.num_points,self.surface_threshold,self.training),
                volume_surface_points(fixed,affine,self.num_points,self.surface_threshold,self.training))

    def forward(self,moving,fixed):
        if moving.ndim==5:moving,fixed=self.clouds(moving,fixed)
        if moving.ndim!=3 or fixed.ndim!=3 or moving.shape[-1]!=7 or fixed.shape[-1]!=7:
            raise ValueError('PointRegM expects [B,N,7] xyz_mm,density,normal point clouds')
        pm,fm=self.encoder(moving);pf,ff=self.encoder(fixed)
        matched,attention=self.cross_attention(fm,ff,ff,need_weights=True)
        displacement=(attention.float()@pf.float()-pm)/96
        features=self.fusion(torch.cat([fm,matched,displacement.to(fm.dtype)],-1))
        with torch.autocast(device_type=features.device.type,enabled=False):
            cm=moving[...,:3].float().mean(1)/96;cf=fixed[...,:3].float().mean(1)/96
            m=moving[...,:3].float()/96-cm[:,None];f=fixed[...,:3].float()/96-cf[:,None]
            stats=torch.cat([cm,cf,cf-cm,(m.transpose(1,2)@m/m.shape[1]).flatten(1),(f.transpose(1,2)@f/f.shape[1]).flatten(1)],-1)
            pooled=torch.cat([features.float().mean(1),features.float().amax(1),stats],-1)
            return torch.tanh(self.pose_head(pooled))*self.pose_scale






def make_regm(config):
    """Construct the current point RegM from its saved configuration."""
    if config.get("architecture") != "point_residual34":
        raise ValueError("Expected a point_residual34 checkpoint; voxel RegM is not supported")
    return PointRegM(**config)
