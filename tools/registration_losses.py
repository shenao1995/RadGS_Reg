"""Volumetric NCC/SSIM and physically normalized SE(3) supervision."""
import math
import torch
import torch.nn.functional as F
from tools.registration_geometry import rotation_angle, relative_rotation


def ssim3d_map(prediction, target, data_range=.04, window=11, sigma=1.5):
    """True separable 3D Gaussian SSIM; valid windows, no per-prediction scaling."""
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError('SSIM inputs must have equal [B,C,X,Y,Z] shapes')
    coord = torch.arange(window, device=prediction.device, dtype=prediction.dtype) - window//2
    kernel = torch.exp(-coord.square()/(2*sigma**2)); kernel = kernel/kernel.sum()
    def smooth(value):
        for axis in range(3):
            shape = [1, 1, 1, 1, 1]; shape[axis+2] = window
            weight = kernel.reshape(shape).expand(value.shape[1], 1, *shape[2:]).contiguous()
            value = F.conv3d(value, weight, groups=value.shape[1])
        return value
    # Concatenating the moments avoids five separate convolution launches per axis.
    channels = prediction.shape[1]
    moments = smooth(torch.cat((prediction, target, prediction.square(), target.square(), prediction*target), 1))
    mp, mt, pp, tt, pt = moments.split(channels, 1)
    vp = (pp-mp.square()).clamp_min(0); vt = (tt-mt.square()).clamp_min(0)
    cov = pt-mp*mt
    c1, c2 = (.01*data_range)**2, (.03*data_range)**2
    return ((2*mp*mt+c1)*(2*cov+c2) / ((mp.square()+mt.square()+c1)*(vp+vt+c2)))


def ssim3d(prediction, target, data_range=.04, window=11):
    return ssim3d_map(prediction, target, data_range, window).flatten(1).mean(1)


def normalized_cross_correlation(prediction, target):
    p, t = prediction.flatten(1), target.flatten(1)
    p = p-p.mean(1, keepdim=True); t = t-t.mean(1, keepdim=True)
    return (p*t).sum(1)/(p.square().sum(1)*t.square().sum(1)).clamp_min(1e-16).sqrt()


def central_crop(volume, size):
    if size is None:return volume
    start = [(s-size)//2 for s in volume.shape[-3:]]
    if min(start)<0:raise ValueError('Crop exceeds volume')
    return volume[..., start[0]:start[0]+size, start[1]:start[1]+size, start[2]:start[2]+size]


def registration_loss(warped, fixed, predicted, truth, geo_weight=.02,
                      translation_scale_mm=10., rotation_scale_deg=5., data_range=.04,
                      crop_size=64, ssim_weight=1., ncc_weight=1., ssim_window=7):
    # A fixed physical central ROI removes artificial empty padding from the loss.
    # Warp is computed on the larger canvas before cropping, so displaced anatomy is retained.
    p, t = central_crop(warped, crop_size), central_crop(fixed, crop_size)
    ncc = 1-normalized_cross_correlation(p,t)
    structural = 1-ssim3d(p,t,data_range=data_range,window=ssim_window)
    relative = relative_rotation(predicted,truth)
    rot = rotation_angle(relative)/math.radians(rotation_scale_deg)
    trans = torch.linalg.vector_norm(predicted[...,:3,3]-truth[...,:3,3],dim=-1)/translation_scale_mm
    geo = rot+trans
    total = ncc_weight*ncc + ssim_weight*structural + geo_weight*geo
    return total.mean(), dict(loss=total.detach().mean(), ncc=ncc.detach().mean(),
                               dssim=structural.detach().mean(), geodesic=geo.detach().mean())
