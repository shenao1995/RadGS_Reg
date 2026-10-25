"""Physical RAS rigid transforms; tensors are [B,C,X,Y,Z], poses [rx,ry,rz,tx,ty,tz]."""
import math
from contextlib import contextmanager
import torch
import torch.nn.functional as F


@contextmanager
def _precise_geometry():
    """Keep physical-coordinate products out of TF32, then restore the caller.

    TF32 is useful for the encoder, but its rounded coordinate operands move
    sampling points even when all pose/grid tensors have dtype float32.
    Nested geometry calls preserve the outer policy.
    """
    previous = torch.backends.cuda.matmul.allow_tf32
    if previous:
        torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        if previous:
            torch.backends.cuda.matmul.allow_tf32 = True


def euler_xyz_matrix(angles):
    """Active column-vector rotation Rz(rz) @ Ry(ry) @ Rx(rx), angles in radians."""
    x,y,z=angles.unbind(-1);cx,sx=x.cos(),x.sin();cy,sy=y.cos(),y.sin();cz,sz=z.cos(),z.sin()
    return torch.stack((cz*cy,cz*sy*sx-sz*cx,cz*sy*cx+sz*sx,
                        sz*cy,sz*sy*sx+cz*cx,sz*sy*cx-cz*sx,
                        -sy,cy*sx,cy*cx),-1).reshape(*angles.shape[:-1],3,3)


def matrix_euler_xyz(rotation):
    # The experiment capture range is far from the Euler singularities.
    return torch.stack((torch.atan2(rotation[...,2,1],rotation[...,2,2]),
                        torch.atan2(-rotation[...,2,0],torch.linalg.vector_norm(rotation[...,:2,0],dim=-1)),
                        torch.atan2(rotation[...,1,0],rotation[...,0,0])),-1)


@_precise_geometry()
def axis_angle_matrix(vector):
    """Rodrigues map with analytic finite gradients at the identity."""
    x,y,z=vector.unbind(-1);zero=torch.zeros_like(x)
    skew=torch.stack((zero,-z,y,z,zero,-x,-y,x,zero),-1).reshape(*vector.shape[:-1],3,3)
    theta=torch.linalg.vector_norm(vector,dim=-1)
    a=torch.sinc(theta/math.pi)[...,None,None]
    b=(.5*torch.sinc(theta/(2*math.pi)).square())[...,None,None]
    eye=torch.eye(3,device=vector.device,dtype=vector.dtype).expand_as(skew)
    return eye+a*skew+b*(skew@skew)


def rigid_matrix(rotation,translation):
    upper=torch.cat((rotation,translation[...,None]),-1)
    bottom=torch.zeros((*upper.shape[:-2],1,4),device=upper.device,dtype=upper.dtype)
    bottom[...,0,3]=1
    return torch.cat((upper,bottom),-2)


def pose_matrix(pose):
    """Axis-angle radians + RAS translation mm, active moving -> fixed."""
    return rigid_matrix(axis_angle_matrix(pose[...,:3]),pose[...,3:])


@_precise_geometry()
def invert_rigid(matrix):
    rotation=matrix[...,:3,:3].transpose(-1,-2)
    return rigid_matrix(rotation,-(rotation@matrix[...,:3,3,None]).squeeze(-1))


def centered_affine(shape,spacing=1.,device=None,dtype=torch.float32):
    shape=torch.as_tensor(shape,device=device,dtype=dtype)
    spacing=torch.as_tensor(spacing,device=device,dtype=dtype).expand(3)
    affine=torch.eye(4,device=device,dtype=dtype)
    affine[:3,:3]=torch.diag(spacing);affine[:3,3]=-(shape-1)*spacing/2
    return affine


@_precise_geometry()
def grid_to_physical(affine,shape):
    """PyTorch normalized grid order (Z,Y,X) -> physical RAS XYZ; align_corners=False."""
    shape=torch.as_tensor(shape,device=affine.device,dtype=affine.dtype)
    reverse=torch.eye(3,device=affine.device,dtype=affine.dtype).flip(0)
    linear=affine[...,:3,:3]@(torch.diag(shape/2)@reverse)
    offset=(affine[...,:3,:3]@((shape-1)/2)[...,None]).squeeze(-1)+affine[...,:3,3]
    return rigid_matrix(linear,offset)


@_precise_geometry()
def warp_volume(moving,moving_to_fixed,moving_affine=None,fixed_affine=None,output_shape=None,mode='bilinear'):
    """Differentiable inverse-mapped resampling, preserving physical millimetres."""
    if moving.ndim!=5:raise ValueError('Expected moving [B,C,X,Y,Z]')
    shape=tuple(output_shape or moving.shape[-3:]);batch=moving.shape[0]
    transform=moving_to_fixed.to(device=moving.device,dtype=moving.dtype)
    if transform.ndim==2:transform=transform[None]
    if transform.shape[0]==1:transform=transform.expand(batch,-1,-1)
    source=centered_affine(moving.shape[-3:],device=moving.device,dtype=moving.dtype) if moving_affine is None else moving_affine.to(moving)
    target=source if fixed_affine is None else fixed_affine.to(moving)
    hs=grid_to_physical(source,moving.shape[-3:]);ht=grid_to_physical(target,shape)
    # Hs^-1 T^-1 Ht converts fixed output grid locations to moving input locations.
    grid_transform=torch.linalg.solve(hs,invert_rigid(transform)@ht)
    grid=F.affine_grid(grid_transform[:,:3],(batch,moving.shape[1],*shape),align_corners=False)
    return F.grid_sample(moving,grid,mode=mode,padding_mode='zeros',align_corners=False)


@_precise_geometry()
def transform_points(matrix,points):
    return points@matrix[...,:3,:3].transpose(-1,-2)+matrix[...,:3,3].unsqueeze(-2)


def rotation_angle(rotation):
    vee=torch.stack((rotation[...,2,1]-rotation[...,1,2],rotation[...,0,2]-rotation[...,2,0],rotation[...,1,0]-rotation[...,0,1]),-1)/2
    sine=torch.linalg.vector_norm(vee,dim=-1)
    cosine=(rotation.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
    return torch.atan2(sine,cosine)


@_precise_geometry()
def relative_rotation(predicted,truth):
    """R_pred R_truth^T for physical pose errors and geodesic supervision."""
    return predicted[...,:3,:3]@truth[...,:3,:3].transpose(-1,-2)


@_precise_geometry()
def pose_errors(predicted,truth,landmarks_fixed=None):
    relative=relative_rotation(predicted,truth)
    angles=matrix_euler_xyz(relative).abs()*180/math.pi
    translation=(predicted[...,:3,3]-truth[...,:3,3]).abs()
    result={**{f'translation_{a}_mm':translation[...,i] for i,a in enumerate('xyz')},
            **{f'rotation_{a}_deg':angles[...,i] for i,a in enumerate('xyz')},
            'rotation_geodesic_deg':rotation_angle(relative)*180/math.pi}
    if landmarks_fixed is not None:
        moving_points=transform_points(invert_rigid(truth),landmarks_fixed)
        tre=torch.linalg.vector_norm(transform_points(predicted,moving_points)-landmarks_fixed,dim=-1).mean(-1)
        result.update(mTRE_mm=tre,success_2mm=(tre<2).float())
    return result
