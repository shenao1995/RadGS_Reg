"""Export online RecM + RegM predictions for fixed simulation cases.

The saved RecM volume is queried from the checkpoint's Gaussian field. The
registered CT is a rigid resampling of the simulated moving CT; no image decoder
or frozen reconstruction cache is used. All volumes use physical attenuation
(1/mm) and centred RAS coordinates in millimetres.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from monai.data import list_data_collate

from networks.DVGSNet import DVGSNet
from networks.pointRegM import make_regm
from Reg3D_infer import save_volume
from Reg3D_train import atomic_json
from tools.data_contract import read_split_pairs
from gs_train import move_batch, network_input, physical_gaussians
from joint_train import pair_metadata, query_batch, registration_pair
from tools.registration_geometry import centered_affine, invert_rigid, pose_matrix, pose_errors, warp_volume
from tools.registration_losses import ssim3d
from r2_gaussian.dataset.GSDataset import GSDataset


def save_preview(path, volumes, canonical):
    """Common planes and window for all rows; never normalize each prediction."""
    from PIL import Image, ImageDraw

    points = (canonical[0, 0] > 0).nonzero()
    centre = points.float().mean(0).round().long().tolist()
    width = canonical.shape[-1]
    panel = Image.new('RGB', (3 * width, len(volumes) * (width + 22) + 24), 'black')
    draw = ImageDraw.Draw(panel)
    for column, label in enumerate(('XY (axial)', 'XZ (coronal)', 'YZ (sagittal)')):
        draw.text((column * width + 5, 4), label, fill='white')
    for row, (label, volume) in enumerate(volumes.items()):
        x = volume[0, 0].detach().float().cpu().numpy()
        planes = (x[:, :, centre[2]], x[:, centre[1], :], x[centre[0], :, :])
        top = 24 + row * (width + 22)
        draw.text((5, top + 3), label + ' [0, 0.04] 1/mm', fill='white')
        for column, plane in enumerate(planes):
            pixels = (np.clip(plane.T / .04, 0, 1) * 255).astype(np.uint8)
            panel.paste(Image.fromarray(pixels).convert('RGB'), (column * width, top + 22))
    panel.save(path)


@torch.no_grad()
def run(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.cuda.set_device(0)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    checkpoint_path = Path(args.checkpoint).resolve()
    # Keep one file descriptor open so an atomic checkpoint replacement cannot
    # change the weights between hashing and loading them.
    with checkpoint_path.open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        handle.seek(0)
        state = torch.load(handle, map_location='cpu')
    config = state['config']
    recm = DVGSNet(**state['recm_model_config']).cuda().eval()
    recm.load_state_dict(state['recm'])
    regm = make_regm(state['regm_model_config']).cuda().eval()
    regm.load_state_dict(state['regm'])
    epoch = state['epoch']
    del state
    contract = SimpleNamespace(**config['reconstruction_contract'])
    options = SimpleNamespace(registration_loss_spacing_mm=config.get('registration_loss_spacing_mm', 2.),reg_input_size=regm.input_size)
    manifest = json.loads((Path(args.reg_data_dir) / 'manifest.json').read_text())
    if Path(manifest['protocol']['source_data']).resolve() != Path(args.data_dir).resolve():
        raise ValueError('RecM and registration data sources differ')
    lookup = {Path(row['filename']).stem: row for row in manifest['samples']}
    lookup['_root'] = args.reg_data_dir
    paths, points = read_split_pairs(args.data_dir, args.split)
    dataset = GSDataset(paths, points, load_points=False)
    if args.case:
        requested = set(args.case)
        indices = [i for i, path in enumerate(paths) if Path(path).stem in requested]
        missing = requested - {Path(paths[i]).stem for i in indices}
        if missing:
            raise ValueError(f'Cases absent from {args.split} split: {sorted(missing)}')
    else:
        indices = np.linspace(0, len(dataset) - 1, min(args.cases, len(dataset)), dtype=int).tolist()
    affine128 = centered_affine((128,) * 3, 1., device='cuda')
    affine192 = centered_affine((192,) * 3, 1., device='cuda')
    affine_net = centered_affine((regm.input_size,) * 3, 192./regm.input_size, device='cuda')
    rows = []
    for index in indices:
        batch = move_batch(list_data_collate([dataset[index]]), 'cuda')
        attributes = physical_gaussians(recm(network_input(batch, contract)), contract)
        reconstructed, canonical = query_batch(attributes, batch)
        names, truth, landmarks = pair_metadata(batch, lookup, 'cuda', epoch, False, config.get('seed', 42))
        moving, fixed, *_ = registration_pair(reconstructed, canonical, truth, affine_net, options)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=config.get('amp_reg', True)):
            pose = regm(moving, fixed)
        matrix = pose_matrix(pose.float())
        # Export at original 1mm resolution, with the full 192mm motion margin.
        canonical192 = F.pad(canonical, (32,) * 6)
        moving192 = warp_volume(canonical192, invert_rigid(truth), affine192, affine192)
        registered192 = warp_volume(moving192, matrix, affine192, affine192)
        name = names[0]
        case_dir = output / name
        case_dir.mkdir(exist_ok=True)
        reference = None
        if args.show_frozen_reference:
            reference_path = Path(args.reg_data_dir) / 'rec128' / (name + '.npy')
            reference = torch.from_numpy(np.load(reference_path).astype(np.float32))[None, None].cuda()
            if reference.shape != reconstructed.shape:
                raise ValueError('Frozen reference and online reconstruction grids differ')
            save_volume(case_dir / 'frozen_reconstruction.nii.gz', reference, affine128)
        for filename, volume, affine in (
            ('ct_gt.nii.gz', canonical, affine128),
            ('reconstruction.nii.gz', reconstructed, affine128),
            ('moving_ct.nii.gz', moving192, affine192),
            ('registered_ct.nii.gz', registered192, affine192),
        ):
            save_volume(case_dir / filename, volume, affine)
        errors = {key: float(value[0]) for key, value in pose_errors(matrix, truth, landmarks).items()}
        errors['reconstruction_ssim3d'] = float(ssim3d(reconstructed, canonical)[0])
        foreground = (canonical[0, 0] > 0).nonzero()
        lo = (foreground.min(0).values - 5).clamp_min(0)
        hi = (foreground.max(0).values + 6).clamp_max(128)
        roi = (slice(None), slice(None), *(slice(int(a), int(b)) for a, b in zip(lo, hi)))
        errors['reconstruction_ssim3d_foreground_roi'] = float(ssim3d(reconstructed[roi], canonical[roi])[0])
        record = dict(name=name, metrics=errors, moving_to_fixed_ras_mm=matrix[0].cpu().tolist(),
                      ground_truth_moving_to_fixed_ras_mm=truth[0].cpu().tolist(),
                      pose_axis_angle_rad_translation_mm=pose[0].float().cpu().tolist(),
                      affine128_voxel_to_ras_mm=affine128.cpu().tolist(),
                      affine192_voxel_to_ras_mm=affine192.cpu().tolist(),
                      registration_input_spacing_mm=2.,
                      training_registration_loss_spacing_mm=options.registration_loss_spacing_mm,
                      export_spacing_mm=1., intensity_unit='1/mm',
                      frozen_reference_checkpoint=manifest['protocol']['reconstruction_checkpoint'] if reference is not None else None,
                      export_note='1mm rigid CT resampling on the full 192mm canvas; network input follows the checkpoint training protocol')
        atomic_json(case_dir / 'transform.json', record)
        if not args.no_preview:
            preview = {'GT CT': canonical192}
            if reference is not None:preview['Frozen RecM'] = F.pad(reference, (32,) * 6)
            preview.update({'Joint RecM': F.pad(reconstructed, (32,) * 6), 'Moving CT': moving192, 'Registered CT': registered192})
            save_preview(case_dir / 'comparison.png', preview, canonical192)
        rows.append(record)
        print(json.dumps(dict(case=name, metrics=errors)), flush=True)
    atomic_json(output / 'manifest.json', dict(checkpoint=str(checkpoint_path), checkpoint_sha256=checkpoint_hash,
                epoch=epoch, split=args.split, exported_cases=len(rows),
                selection='explicit names or evenly spaced deterministic split indices; no quality-based case selection',
                quantitative_evaluation='These exports are qualitative examples; use joint_evaluate.py for complete-split metrics',
                cases=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data-dir', 'reg-data-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    parser.add_argument('--cases', type=int, default=5)
    parser.add_argument('--case', action='append', help='Exact sample stem; repeat for multiple cases')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--show-frozen-reference', action='store_true',help='Add the cached frozen RecM reference to validation/test exports; prediction still uses the online joint model')
    args = parser.parse_args()
    if args.cases < 1:
        parser.error('--cases must be positive')
    if args.show_frozen_reference and args.split=='train':
        parser.error('The full-resolution frozen reference is available only for val/test')
    run(args)


if __name__ == '__main__':
    main()
