"""Generate exact RecM splits and bounded rigid pairs; export frozen RecM volumes."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from networks.DVGSNet import DVGSNet
from tools.data_contract import input_view_indices, volume_is_query_aligned
from tools.registration_data import sampled_transform, stable_seed
from tools.registration_geometry import centered_affine, invert_rigid, transform_points, warp_volume
from tools.registration_losses import ssim3d
from gs_train import atomic_json
from r2_gaussian.gaussian import net_query


def initialize(args):
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    for name in ('arrays','landmarks','rec128','progress','splits'): (root/name).mkdir(exist_ok=True)
    source=Path(args.data_dir)
    rows=json.loads((source/'samples_manifest.json').read_text())['samples']
    identities=set();ct_splits={}
    for row in rows:
        assert row['filename'] not in identities;identities.add(row['filename'])
        previous=ct_splits.setdefault(row['case_id'],row['split']);assert previous==row['split']
        angles,translation,matrix=sampled_transform(stable_seed(row['filename'],args.seed))
        row.update(rotation_xyz_deg=angles.tolist(),translation_ras_mm=translation.tolist(),moving_to_fixed=matrix.tolist())
    for split in ('train','val','test'):
        expected=[line.strip() for line in (source/'splits'/f'{split}.txt').read_text().splitlines() if line.strip()]
        assert set(expected)=={r['filename'] for r in rows if r['split']==split}
        (root/'splits'/f'{split}.txt').write_text('\n'.join(expected)+'\n')
    config=Path(args.checkpoint).parent/'config.json'
    manifest=dict(protocol=dict(schema='radgs-regm-rigid-v1',source_data=str(source.resolve()),
        reconstruction_checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=hashlib.file_digest(open(args.checkpoint,'rb'),'sha256').hexdigest(),
        reconstruction_config=json.loads(config.read_text()),tensor_axes='XYZ',physical_frame='centered_RAS',
        transform_direction='moving_to_fixed',pose_rotation='active Rz@Ry@Rx for sampling; network uses axis-angle radians',
        rotation_each_axis_deg=[-5,5],translation_each_axis_mm=[-10,10],seed=args.seed,
        raw_ct_spacing_mm=[1.,1.,1.],original_shape=[128]*3,canvas_mm=[192.]*3,
        cached_shape=[96]*3,cached_spacing_mm=[2.]*3,affine_ras_mm=centered_affine((96,)*3,2.).tolist(),
        cache_downsampling='128^3 1mm CT or RecM padded 32/side to 192^3, trilinear resize to 96^3; no anatomy-dependent recentering',
        intensity_unit='linear attenuation 1/mm; fixed shared intensity scale .04; no separate max normalization',
        perturbation='moving=warp(canonical,inverse(T_gt)); fixed=canonical or RecM reconstruction',
        split_policy='Exact original RecM filenames/CT groups, no resplitting',
        full_resolution_reconstruction_saved_for=['val','test']),samples=rows,
        counts=dict(Counter(r['split'] for r in rows)))
    atomic_json(root/'manifest.json',manifest)
    print(json.dumps(dict(counts=manifest['counts'],case_groups=len(ct_splits)),indent=2),flush=True)


def load_recm(checkpoint):
    config=json.loads((Path(checkpoint).parent/'config.json').read_text())
    model=DVGSNet(**config['model_config']).cuda().eval()
    state=torch.load(checkpoint,map_location='cpu')
    model.load_state_dict(state.get('model_state_dict',state),strict=True)
    del state
    return model,config


def fast_input(data,device='cuda'):
    assert volume_is_query_aligned(data.get('metadata',{})), 'Unrecognized source geometry'
    spacing=np.array(data['dVoxel']);shape=np.array(data['nVoxel'])
    scale=2/max(shape*spacing/1000.)
    ap,la=input_view_indices(data['metadata'],data['numTrain'])
    pair=torch.as_tensor(np.asarray(data['train']['projections'])[[ap,la]],device=device).float()[:,None]*scale
    pair=F.interpolate(pair,size=(128,128),mode='bilinear',align_corners=False)[:,0][None]
    center=torch.as_tensor(np.array(data['offOrigin'])/1000*scale,device=device,dtype=torch.float32)
    extent=torch.as_tensor(shape*spacing/1000*scale,device=device,dtype=torch.float32)
    return pair,center,extent


def export(args):
    torch.set_num_threads(2);torch.cuda.set_device(args.worker % torch.cuda.device_count())
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    root=Path(args.output);manifest=json.loads((root/'manifest.json').read_text())
    model,config=load_recm(args.checkpoint)
    rows=[r for i,r in enumerate(manifest['samples']) if i%args.workers==args.worker]
    affine=centered_affine((96,)*3,2.,device='cuda')
    counts=Counter();started=time.time();verified=False
    for index,row in enumerate(rows):
        name=Path(row['filename']).stem;path=Path(args.data_dir)/row['filename']
        output=root/'arrays'/f'{name}.npy';landmark_path=root/'landmarks'/f'{name}.npy'
        if args.resume and output.is_file() and landmark_path.is_file():
            counts['skipped']+=1;continue
        with path.open('rb') as f:data=pickle.load(f)
        with torch.no_grad():
            pair,center,extent=fast_input(data)
            attributes=model(pair*config['input_scale']);attributes['density']/=config['density_unit']
            attr={k:v[0].contiguous() for k,v in attributes.items()}
            prediction=net_query(attr,center,torch.tensor([128]*3,device='cuda'),extent)['vol'][None,None]
            gt=torch.as_tensor(data['image'],device='cuda').float()[None,None]
            if not verified:
                from r2_gaussian.dataset.GSDataset import GSDataset
                from monai.data import list_data_collate
                from gs_train import query_volume,move_batch,network_input,physical_gaussians
                reference=move_batch(list_data_collate([GSDataset([str(path)],load_points=False)[0]]),'cuda')
                torch.testing.assert_close(pair,reference['input_imgs'].squeeze(2),rtol=1e-6,atol=1e-8)
                reference_attributes=physical_gaussians(model(network_input(reference,SimpleNamespace(**config))),SimpleNamespace(**config))
                reference_prediction,reference_gt=query_volume(reference_attributes,reference,0,128)
                torch.testing.assert_close(prediction,reference_prediction,rtol=1e-5,atol=1e-7)
                torch.testing.assert_close(gt,reference_gt,rtol=0,atol=0)
                atomic_json(root/'progress'/f'input_verified_{args.worker}.json',dict(sample=name,verified=True))
                verified=True;del reference,reference_attributes,reference_prediction,reference_gt
            def canvas(value):return F.interpolate(F.pad(value,(32,)*6),size=(96,)*3,mode='trilinear',align_corners=False)
            canonical=canvas(gt);rec=canvas(prediction)
            matrix=torch.tensor(row['moving_to_fixed'],device='cuda')
            moving=warp_volume(canonical,invert_rigid(matrix),affine,affine)
            points=np.argwhere(data['mask']>0).astype(np.float32)-63.5
            assert len(points)>0
            # Prove all foreground centres stay inside the common canvas after perturbation.
            moved=transform_points(invert_rigid(matrix),torch.tensor(points,device='cuda'))
            margin=95.5-float(moved.abs().max());assert margin>1.,(name,margin)
            rng=np.random.default_rng(stable_seed(name,args.seed))
            landmarks=points[rng.choice(len(points),64,replace=len(points)<64)]
            np.save(landmark_path,landmarks)
            stacked=torch.cat((canonical,moving,rec),dim=1)[0].cpu().numpy().astype(np.float16)
            assert np.isfinite(stacked).all()
            np.save(output,stacked)
            if row['split']!='train':np.save(root/'rec128'/f'{name}.npy',prediction[0,0].cpu().numpy().astype(np.float16))
            # Fixed, physically shared data range, and a GT maximum shared by both arrays.
            if row['split']=='val':
                full_ssim=float(ssim3d(prediction,gt,data_range=.04))
                shared_ssim=float(ssim3d(prediction/gt.max().clamp_min(1e-8),gt/gt.max().clamp_min(1e-8),data_range=1))
                atomic_json(root/'progress'/f'metric_{name}.json',dict(sample=name,ssim3d_fixed_range=full_ssim,ssim3d_shared_gt_peak=shared_ssim,canvas_foreground_margin_mm=margin))
        counts[row['split']]+=1
        if (index+1)%10==0 or index+1==len(rows):
            status=dict(worker=args.worker,completed=index+1,total=len(rows),counts=dict(counts),seconds=time.time()-started)
            atomic_json(root/'progress'/f'worker_{args.worker}.json',status);print(json.dumps(status),flush=True)
    atomic_json(root/'progress'/f'worker_{args.worker}.json',dict(status='completed',worker=args.worker,completed=len(rows),total=len(rows),counts=dict(counts)))


def audit(args):
    root=Path(args.output);manifest=json.loads((root/'manifest.json').read_text());counts=Counter();table=defaultdict(Counter)
    for row in manifest['samples']:
        name=Path(row['filename']).stem
        arr=np.load(root/'arrays'/f'{name}.npy',mmap_mode='r')
        assert arr.shape==(3,96,96,96) and arr.dtype==np.float16
        assert (root/'landmarks'/f'{name}.npy').is_file()
        assert max(abs(v) for v in row['rotation_xyz_deg'])<=5 and max(abs(v) for v in row['translation_ras_mm'])<=10
        counts[row['split']]+=1;table[row['level']][row['split']]+=1
    assert dict(counts)==manifest['counts']
    metrics=[json.loads(p.read_text()) for p in (root/'progress').glob('metric_*.json')]
    assert len(metrics)==counts['val']
    summary=dict(status='verified',counts=dict(counts),levels={k:dict(v) for k,v in sorted(table.items())},
        baseline_val_reconstruction={k:float(np.mean([m[k] for m in metrics])) for k in ('ssim3d_fixed_range','ssim3d_shared_gt_peak')},
        min_canvas_foreground_margin_mm=min(m['canvas_foreground_margin_mm'] for m in metrics))
    atomic_json(root/'audit.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-dir',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['initialize','export','audit'],required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--resume',action='store_true')
    a=p.parse_args();{'initialize':initialize,'export':export,'audit':audit}[a.mode](a)
