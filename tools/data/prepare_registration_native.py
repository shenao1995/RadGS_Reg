"""Preserve all original members/poses while caching original-1mm CT and frozen RecM."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch

from tools.registration_geometry import centered_affine


def atomic_json(path, value):
    path = Path(path); temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2)); temporary.replace(path)


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def initialize(source_root, target):
    assert source_root.resolve() != target.resolve(), 'Native cache must have its own output directory'
    original_path = source_root/'manifest.json'
    original = json.loads(original_path.read_text())
    actual = Counter(row['split'] for row in original['samples'])
    if not actual or dict(actual) != {k:v for k,v in original['counts'].items() if v}:
        raise ValueError('Registration manifest counts disagree with its samples')
    target.mkdir(parents=True, exist_ok=True)
    for name in ['arrays_native128', 'receipts', 'progress']:
        (target/name).mkdir(exist_ok=True)
    for name in ['landmarks', 'splits', 'rec128']:
        link = target/name
        if not link.exists():
            link.symlink_to((source_root/name).resolve(), target_is_directory=True)
        assert link.resolve() == (source_root/name).resolve()
    manifest = json.loads(json.dumps(original))
    manifest['protocol'].update(schema='radgs-regm-native-pair-v1',
        cache_format='ct_rec_native128_float16', stored_shape=[128]*3,
        stored_channels=['canonical_ct', 'frozen_recm'], cached_shape=[192]*3,
        cached_spacing_mm=[1.]*3, affine_ras_mm=centered_affine((192,)*3, 1.).tolist(),
        source_registration_cache=str(source_root.resolve()), source_manifest_sha256=digest(original_path),
        cache_downsampling='None: store original128^3 at1mm; pad32per side on device, warp moving at1mm',
        full_resolution_reconstruction_saved_for=['train', 'val', 'test'],
        cache_use='Standalone training only; online joint training must query its live RecM network',
        old_loader_compatibility='Separate arrays_native128 directory prevents treating CT/RecM as the old three-channel cache')
    path = target/'manifest.json'
    if path.exists():
        assert json.loads(path.read_text()) == manifest, 'Refuse to replace a different native dataset'
    else:
        atomic_json(path, manifest)
    return manifest


@torch.no_grad()
def export(source_root, target, resume):
    from tools.data.prepare_registration import fast_input, load_recm
    from r2_gaussian.gaussian import net_query
    manifest = json.loads((target/'manifest.json').read_text())
    protocol = manifest['protocol']; checkpoint = Path(protocol['reconstruction_checkpoint'])
    assert digest(checkpoint) == protocol['checkpoint_sha256']
    assert digest(source_root/'manifest.json') == protocol['source_manifest_sha256']
    torch.set_num_threads(2); torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    model, config = load_recm(str(checkpoint))
    for key in ['model_config', 'input_scale', 'density_unit']:
        assert config[key] == protocol['reconstruction_config'][key], f'Frozen RecM contract changed: {key}'
    source = Path(protocol['source_data']); started = time.time(); verified = False
    counts = Counter(); reused = 0; skipped = 0
    for index, row in enumerate(manifest['samples']):
        name = Path(row['filename']).stem
        output = target/'arrays_native128'/f'{name}.npy'
        receipt = target/'receipts'/f'{name}.json'
        if resume and output.exists() and receipt.exists():
            record = json.loads(receipt.read_text())
            assert record['array_sha256'] == digest(output) and record['source_filename'] == row['filename']
            skipped += 1; counts[row['split']] += 1
        else:
            with (source/row['filename']).open('rb') as handle:
                data = pickle.load(handle)
            assert np.array_equal(np.asarray(data['nVoxel']), [128]*3)
            assert np.allclose(data['dVoxel'], [1.]*3, atol=1e-8)
            gt = np.asarray(data['image'], dtype=np.float32)
            assert gt.shape == (128,)*3 and np.isfinite(gt).all()
            old_full = source_root/'rec128'/f'{name}.npy'
            if old_full.exists():
                prediction = np.load(old_full)
                assert prediction.shape == gt.shape and prediction.dtype == np.float16
                reused += 1; reconstruction_source = 'exact_existing_native128_cache'
            else:
                assert row['split'] == 'train', 'Held-out original native cache is missing'
                pair, center, extent = fast_input(data)
                attributes = model(pair*config['input_scale']); attributes['density'] /= config['density_unit']
                attributes = {key: value[0].contiguous() for key, value in attributes.items()}
                predicted = net_query(attributes, center, torch.tensor([128]*3, device='cuda'), extent)['vol']
                if not verified:
                    from r2_gaussian.dataset.GSDataset import GSDataset
                    from monai.data import list_data_collate
                    from gs_train import query_volume, move_batch, network_input, physical_gaussians
                    reference = move_batch(list_data_collate([GSDataset([str(source/row['filename'])], load_points=False)[0]]), 'cuda')
                    expected_attributes = physical_gaussians(model(network_input(reference, SimpleNamespace(**config))), SimpleNamespace(**config))
                    expected_prediction, expected_gt = query_volume(expected_attributes, reference, 0, 128)
                    torch.testing.assert_close(pair, reference['input_imgs'].squeeze(2), rtol=1e-6, atol=1e-8)
                    torch.testing.assert_close(predicted, expected_prediction[0, 0], rtol=1e-5, atol=1e-7)
                    np.testing.assert_array_equal(gt, expected_gt[0, 0].cpu().numpy())
                    atomic_json(target/'progress/input_contract_verified.json', dict(status='passed', sample=name))
                    verified = True
                    del reference, expected_attributes, expected_prediction, expected_gt
                prediction = predicted.cpu().numpy().astype(np.float16)
                reconstruction_source = 'frozen_RecM68_original128_query'
            array = np.stack((gt.astype(np.float16), prediction))
            assert np.isfinite(array).all()
            temporary = output.with_suffix('.npy.tmp')
            with temporary.open('wb') as handle:
                np.save(handle, array)
            temporary.replace(output)
            atomic_json(receipt, dict(source_filename=row['filename'], split=row['split'],
                array_sha256=digest(output), shape=list(array.shape), dtype='float16',
                reconstruction_source=reconstruction_source, reconstruction_checkpoint_sha256=protocol['checkpoint_sha256']))
            counts[row['split']] += 1
        if (index+1) % 50 == 0 or index+1 == len(manifest['samples']):
            status = dict(status='running', completed=index+1, total=4303, counts=dict(counts),
                          reused_existing_native=reused, skipped=skipped, seconds=time.time()-started,
                          cuda_device=torch.cuda.get_device_name())
            atomic_json(target/'progress/export.json', status); print(json.dumps(status), flush=True)
    atomic_json(target/'progress/export.json', dict(status='completed_awaiting_audit', completed=4303, total=4303,
        counts=dict(counts), reused_existing_native=reused, skipped=skipped, seconds=time.time()-started,
        cuda_device=torch.cuda.get_device_name()))


def audit(source_root, target):
    original = json.loads((source_root/'manifest.json').read_text())
    manifest = json.loads((target/'manifest.json').read_text())
    assert original['samples'] == manifest['samples'] and original['counts'] == manifest['counts']
    assert digest(source_root/'manifest.json') == manifest['protocol']['source_manifest_sha256']
    assert {path.stem for path in (target/'arrays_native128').glob('*.npy')} == {
        Path(row['filename']).stem for row in manifest['samples']}
    counts = Counter(); levels = {}; groups = {}
    for row in manifest['samples']:
        name = Path(row['filename']).stem; path = target/'arrays_native128'/f'{name}.npy'
        values = np.load(path, mmap_mode='r'); receipt = json.loads((target/'receipts'/f'{name}.json').read_text())
        assert values.shape == (2, 128, 128, 128) and values.dtype == np.float16 and np.isfinite(values).all()
        assert receipt['array_sha256'] == digest(path)
        assert receipt['source_filename'] == row['filename'] and receipt['split'] == row['split']
        assert receipt['reconstruction_checkpoint_sha256'] == manifest['protocol']['checkpoint_sha256']
        with (Path(manifest['protocol']['source_data'])/row['filename']).open('rb') as handle:
            original_data = pickle.load(handle)
        np.testing.assert_array_equal(values[0], np.asarray(original_data['image'], dtype=np.float16))
        assert (target/'landmarks'/f'{name}.npy').resolve() == (source_root/'landmarks'/f'{name}.npy').resolve()
        if row['split'] != 'train':
            np.testing.assert_array_equal(values[1], np.load(source_root/'rec128'/f'{name}.npy'))
        counts[row['split']] += 1; levels.setdefault(row['level'], Counter())[row['split']] += 1
        assert groups.setdefault(row['case_id'], row['split']) == row['split']
        assert max(abs(x) for x in row['rotation_xyz_deg']) <= 5 and max(abs(x) for x in row['translation_ras_mm']) <= 10
    assert dict(counts) == manifest['counts'] and sum(counts.values()) == 4303
    result = dict(status='verified', counts=dict(counts), levels={k: dict(v) for k, v in sorted(levels.items())},
                  ct_groups=dict(Counter(groups.values())), original_members_poses_and_splits_identical=True,
                  stored_shape=[2, 128, 128, 128], stored_spacing_mm=[1.]*3,
                  canvas_shape=[192]*3, canvas_spacing_mm=[1.]*3,
                  original_val_test_reconstructions_bitwise_preserved=True,
                  all_canonical_voxels_equal_original_ct_after_declared_float16_storage=True,
                  reconstruction_checkpoint_sha256=manifest['protocol']['checkpoint_sha256'],
                  source_manifest_sha256=manifest['protocol']['source_manifest_sha256'],
                  total_array_bytes=sum((target/'arrays_native128'/(Path(row['filename']).stem+'.npy')).stat().st_size for row in manifest['samples']),
                  test_quality_metrics_computed=False, ready_for_training=True)
    atomic_json(target/'audit.json', result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-registration', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=['initialize', 'export', 'audit'], required=True)
    parser.add_argument('--resume', action='store_true'); args = parser.parse_args()
    source_root, target = Path(args.source_registration), Path(args.output)
    if args.mode == 'initialize':
        initialize(source_root, target)
    elif args.mode == 'export':
        export(source_root, target, args.resume)
    else:
        audit(source_root, target)
