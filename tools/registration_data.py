"""RecM-matched registration pairs, in a shared centered RAS physical canvas."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tools.registration_geometry import centered_affine, euler_xyz_matrix, rigid_matrix, invert_rigid, warp_volume


def sampled_transform(seed, rotation_deg=5., translation_mm=10.):
    rng=np.random.default_rng(seed)
    angles=rng.uniform(-rotation_deg,rotation_deg,3)
    translation=rng.uniform(-translation_mm,translation_mm,3)
    matrix=rigid_matrix(euler_xyz_matrix(torch.tensor(np.deg2rad(angles))),torch.tensor(translation))
    return angles,translation,matrix.float()


def stable_seed(name, seed=42):
    return (int(hashlib.sha256(name.encode()).hexdigest()[:12],16)+seed) % (2**32)


class RegistrationDataset(Dataset):
    def __init__(self,root,split='train',limit=0):
        self.root=Path(root)
        manifest=json.loads((self.root/'manifest.json').read_text())
        self.protocol=manifest['protocol']
        self.samples=[r for r in manifest['samples'] if r['split']==split]
        if limit:self.samples=self.samples[:limit]
        self.affine=torch.tensor(self.protocol['affine_ras_mm'],dtype=torch.float32)
        self.split=split

    def __len__(self):return len(self.samples)

    def __getitem__(self,index):
        row=self.samples[index];name=Path(row['filename']).stem
        values=torch.from_numpy(np.load(self.root/'arrays'/f'{name}.npy').astype(np.float32))
        return dict(canonical=values[0:1],moving=values[1:2],fixed_rec=values[2:3],
                    transform=torch.tensor(row['moving_to_fixed'],dtype=torch.float32),
                    landmarks=torch.from_numpy(np.load(self.root/'landmarks'/f'{name}.npy').astype(np.float32)),
                    name=name,index=index)


def prepare_pair(batch,device,affine,epoch=0,augment=False,seed=42,fixed_mode='rec',curriculum_epochs=10):
    canonical=batch['canonical'].to(device,non_blocking=True)
    rec=batch['fixed_rec'].to(device,non_blocking=True)
    truth=batch['transform'].to(device,non_blocking=True)
    if augment:
        truth=torch.stack([sampled_transform(stable_seed(f'{name}:{epoch}',seed))[2] for name in batch['name']]).to(device)
        moving=warp_volume(canonical,invert_rigid(truth),affine,affine)
    else:moving=batch['moving'].to(device,non_blocking=True)
    if fixed_mode=='ct':fixed=canonical
    elif fixed_mode=='rec':fixed=rec
    elif fixed_mode=='curriculum':
        alpha=min(1.,max(0.,epoch/max(1,curriculum_epochs)))
        fixed=canonical*(1-alpha)+rec*alpha
    else:raise ValueError(fixed_mode)
    return moving,fixed,truth
