"""Compact original-1mm pairs for standalone RegM resolution experiments."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from tools.registration_data import sampled_transform, stable_seed
from tools.registration_geometry import invert_rigid, warp_volume


class NativeRegistrationDataset(Dataset):
    def __init__(self, root, split='train', limit=0):
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        self.protocol = manifest['protocol']
        if self.protocol.get('cache_format') != 'ct_rec_native128_float16':
            raise ValueError('Expected an explicitly identified native128 cache')
        if json.loads((self.root/'audit.json').read_text()).get('status') != 'verified':
            raise ValueError('Native cache must pass its complete dataset audit before training')
        if self.protocol['cached_shape'] != [192]*3 or self.protocol['cached_spacing_mm'] != [1.]*3:
            raise ValueError('Native pairs require the original 192mm canvas at 1mm')
        self.samples = [row for row in manifest['samples'] if row['split'] == split]
        if limit:
            self.samples = self.samples[:limit]
        self.affine = torch.tensor(self.protocol['affine_ras_mm'], dtype=torch.float32)
        self.split = split

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]; name = Path(row['filename']).stem
        values = np.load(self.root/'arrays_native128'/f'{name}.npy')
        if values.shape != (2, 128, 128, 128) or values.dtype != np.float16:
            raise ValueError('Invalid native CT/RecM array')
        # Keep host transfers compact; prepare_native_pair pads in FP32 on the device.
        return dict(canonical=torch.from_numpy(values[0:1]), fixed_rec=torch.from_numpy(values[1:2]),
                    transform=torch.tensor(row['moving_to_fixed'], dtype=torch.float32),
                    landmarks=torch.from_numpy(np.load(self.root/'landmarks'/f'{name}.npy').astype(np.float32)),
                    name=name, index=index)


def prepare_native_pair(batch, device, affine, epoch=0, augment=False, seed=42,
                        fixed_mode='rec', curriculum_epochs=10):
    canonical = F.pad(batch['canonical'].to(device=device, dtype=torch.float32), (32,)*6)
    rec = F.pad(batch['fixed_rec'].to(device=device, dtype=torch.float32), (32,)*6)
    truth = batch['transform'].to(device=device, dtype=torch.float32)
    if augment:
        truth = torch.stack([sampled_transform(stable_seed(f'{name}:{epoch}', seed))[2]
                             for name in batch['name']]).to(device)
    moving = warp_volume(canonical, invert_rigid(truth), affine, affine)
    if fixed_mode == 'ct':
        fixed = canonical
    elif fixed_mode == 'rec':
        fixed = rec
    elif fixed_mode == 'curriculum':
        alpha = min(1., max(0., epoch/max(1, curriculum_epochs)))
        fixed = canonical*(1-alpha)+rec*alpha
    else:
        raise ValueError(fixed_mode)
    return moving, fixed, truth
