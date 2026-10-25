"""Evaluate RegM on matched held-out pairs, or rigidly register two physical volumes."""
import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader,default_collate

from Reg3D_train import checkpoint_model,evaluate,atomic_json
from tools.registration_data import RegistrationDataset
from tools.registration_geometry import centered_affine, pose_matrix, warp_volume


def network_canvas(model,device):
    size=int(model.config.get('input_size',96))
    if size<=0 or size!=model.input_size:
        raise ValueError('Invalid or inconsistent saved RegM input grid')
    shape=(size,)*3
    return shape,centered_affine(shape,192./size,device=device)


def read_volume(path,device):
    import SimpleITK as sitk
    image=sitk.ReadImage(str(path))
    array=sitk.GetArrayFromImage(image).transpose(2,1,0).copy()
    affine=np.eye(4)
    affine[:3,:3]=np.array(image.GetDirection()).reshape(3,3)@np.diag(image.GetSpacing())
    affine[:3,3]=image.GetOrigin()
    affine=np.diag([-1.,-1.,1.,1.])@affine # ITK LPS -> physical RAS
    return torch.tensor(array,device=device,dtype=torch.float32)[None,None],torch.tensor(affine,device=device,dtype=torch.float32)


def save_volume(path,volume,affine):
    import SimpleITK as sitk
    affine=affine.detach().cpu().numpy();lps=np.diag([-1.,-1.,1.,1.])@affine
    spacing=np.linalg.norm(lps[:3,:3],axis=0)
    image=sitk.GetImageFromArray(volume[0,0].detach().cpu().numpy().transpose(2,1,0).copy())
    image.SetSpacing(spacing.tolist());image.SetOrigin(lps[:3,3].tolist())
    image.SetDirection((lps[:3,:3]/spacing[None]).reshape(-1).tolist())
    sitk.WriteImage(image,str(path))


@torch.no_grad()
def register_files(args,model):
    device='cuda';moving,ma=read_volume(args.moving,device);fixed,fa=read_volume(args.fixed,device)
    # Network transforms rotate around the fixed ROI centre; export transforms in global RAS.
    centre=(fa[:3,:3]@torch.tensor([(n-1)/2 for n in fixed.shape[-3:]],device=device))+fa[:3,3]
    ma_local=ma.clone();ma_local[:3,3]-=centre
    fa_local=fa.clone();fa_local[:3,3]-=centre
    shape,ca=network_canvas(model,device);identity=torch.eye(4,device=device)
    moving_canvas=warp_volume(moving,identity,ma_local,ca,output_shape=shape)
    fixed_canvas=warp_volume(fixed,identity,fa_local,ca,output_shape=shape)
    if moving_canvas.max()<=0 or fixed_canvas.max()<=0:
        raise ValueError('Empty input or no physical overlap with the fixed 192mm canvas; provide volumes in a common initial RAS frame')
    if moving_canvas.max()>.2 or fixed_canvas.max()>.2:
        raise ValueError('Expected linear attenuation volumes in 1/mm, not raw HU or independently normalized intensities')
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=not args.no_amp):pose=model(moving_canvas,fixed_canvas)
    local=pose_matrix(pose.float())[0];global_matrix=local.clone()
    global_matrix[:3,3]=local[:3,3]+centre-local[:3,:3]@centre
    warped=warp_volume(moving,global_matrix,ma,fa,output_shape=fixed.shape[-3:])
    save_volume(Path(args.output_dir)/'registered_moving.nii.gz',warped,fa)
    atomic_json(Path(args.output_dir)/'transform.json',dict(moving_to_fixed_ras_mm=global_matrix.cpu().tolist(),
        local_pose_axis_angle_rad_translation_mm=pose[0].cpu().tolist(),rotation_centre_ras_mm=centre.cpu().tolist(),
        moving=str(Path(args.moving).resolve()),fixed=str(Path(args.fixed).resolve()),checkpoint=args.checkpoint,
        network_input_shape=list(shape),network_spacing_mm=192./shape[0],
        intensity_unit='1/mm',registered_volume='registered_moving.nii.gz'))


@torch.no_grad()
def evaluate_split(args,model):
    protocol=json.loads((Path(args.data_dir)/'manifest.json').read_text())['protocol']
    native=protocol.get('cache_format')=='ct_rec_native128_float16'
    if native:
        from tools.registration_native_data import NativeRegistrationDataset, prepare_native_pair
        from registration_native_train import evaluate_native
        dataset=NativeRegistrationDataset(args.data_dir,args.split,args.limit)
        evaluator=evaluate_native
    else:
        if model.input_size!=96:
            raise ValueError('A high-resolution model requires its original1mm cache or physical input files')
        dataset=RegistrationDataset(args.data_dir,args.split,args.limit)
        evaluator=evaluate
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    metrics,rows=evaluator(model,loader,'cuda',dataset.affine.cuda(),SimpleNamespace(amp=not args.no_amp),fixed_mode=args.fixed_mode)
    output=Path(args.output_dir)
    with (output/'cases.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    atomic_json(output/'cases.json',rows)
    atomic_json(output/'evaluation.json',dict(checkpoint=str(Path(args.checkpoint).resolve()),split=args.split,cases=len(rows),fixed_mode=args.fixed_mode,metrics=metrics,
        pose_error='Absolute translation differences in RAS mm; absolute xyz Euler components of R_pred R_gt^T in degrees',
        mTRE='64 fixed foreground physical landmarks, transformed to moving by inverse(T_gt), then by T_pred',
        dataset_protocol=dataset.protocol))
    print(json.dumps(metrics,indent=2),flush=True)
    # Optional actual rigidly warped CT export; no image decoder prediction.
    for index in range(min(args.export_cases,len(dataset))):
        sample=dataset[index]
        affine=dataset.affine.cuda()
        if native:
            moving,fixed,_=prepare_native_pair(default_collate([sample]),'cuda',affine,fixed_mode=args.fixed_mode)
        else:
            moving=sample['moving'][None].cuda()
            fixed=sample['fixed_rec' if args.fixed_mode=='rec' else 'canonical'][None].cuda()
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=not args.no_amp):pose=model(moving,fixed)
        matrix=pose_matrix(pose.float())
        warped=warp_volume(moving,matrix,affine,affine)
        save_volume(output/f'{sample["name"]}_registered.nii.gz',warped,affine)
        atomic_json(output/f'{sample["name"]}_transform.json',dict(moving_to_fixed=matrix[0].cpu().tolist(),ground_truth=sample['transform'].tolist()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--data-dir');p.add_argument('--split',choices=['train','val','test'],default='val')
    p.add_argument('--moving');p.add_argument('--fixed');p.add_argument('--fixed-mode',choices=['ct','rec'],default='rec')
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--limit',type=int,default=0);p.add_argument('--export-cases',type=int,default=0);p.add_argument('--no-amp',action='store_true')
    args=p.parse_args()
    if bool(args.moving)!=bool(args.fixed):p.error('--moving and --fixed must be provided together')
    if not args.moving and not args.data_dir:p.error('Provide --data-dir or --moving/--fixed')
    Path(args.output_dir).mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    model,_=checkpoint_model(args.checkpoint);model.eval()
    if args.moving:register_files(args,model)
    else:evaluate_split(args,model)


if __name__=='__main__':main()
