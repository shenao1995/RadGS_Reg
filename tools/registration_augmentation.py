"""Training input perturbations in physical units, without changing pose targets."""
import math

import torch
import torch.nn.functional as F

from tools.registration_geometry import euler_xyz_matrix, rigid_matrix, warp_volume


def gaussian_blur_mm(volume, sigma_mm, spacing_mm):
    """Symmetric physical blur; retain the input grid and the gradient to RecM.

    Used only for the fixed input to the pose encoder during training. Image
    losses still compare the actual warped CT with the original RecM field.
    """
    sigma_mm = float(sigma_mm)
    spacing_mm = tuple(float(value) for value in spacing_mm)
    if not math.isfinite(sigma_mm) or sigma_mm < 0:
        raise ValueError('Blur sigma must be finite and nonnegative')
    if len(spacing_mm) != 3 or any(not math.isfinite(value) or value <= 0 for value in spacing_mm):
        raise ValueError('Expected three positive finite physical voxel spacings')
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError('Expected a single-channel [B,1,X,Y,Z] field')
    if sigma_mm == 0:
        return volume
    result = volume
    for axis, spacing in enumerate(spacing_mm):
        sigma = sigma_mm / spacing
        radius = max(1, math.ceil(3 * sigma))
        positions = torch.arange(-radius, radius + 1, device=volume.device, dtype=volume.dtype)
        kernel = torch.exp(-.5 * (positions / sigma).square())
        kernel = kernel / kernel.sum()
        shape = [1, 1, 1, 1, 1]
        shape[axis + 2] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = radius
        result = F.conv3d(result, kernel.reshape(shape), padding=padding)
    return result


def augment_fixed_input(fixed, spacing_mm, sigmas_mm):
    """Choose one listed physical sigma per training batch (Torch RNG)."""
    if len(sigmas_mm) == 1:
        sigma = sigmas_mm[0]
    else:
        index = int(torch.randint(len(sigmas_mm), (), device=fixed.device))
        sigma = sigmas_mm[index]
    return gaussian_blur_mm(fixed, sigma, spacing_mm), float(sigma)


def jitter_fixed_input(fixed,affine,rotation_deg=0.,translation_mm=0.):
    """Corrupt the measured encoder input, retaining the original loss/pose frame.

    This represents reconstruction uncertainty. It never changes the known
    moving-CT perturbation, true pose, or the uncorrupted fixed image-loss target.
    No perturbation is applied during validation or inference.
    """
    if any(not math.isfinite(value) or value<0 for value in (rotation_deg,translation_mm)):
        raise ValueError('Fixed-input pose noise must be finite and nonnegative')
    if rotation_deg==0 and translation_mm==0:return fixed,None
    noise=torch.rand((len(fixed),6),device=fixed.device,dtype=fixed.dtype)*2-1
    angles=noise[:,:3]*math.radians(rotation_deg)
    translation=noise[:,3:]*translation_mm
    matrix=rigid_matrix(euler_xyz_matrix(angles),translation)
    return warp_volume(fixed,matrix,affine,affine),matrix
