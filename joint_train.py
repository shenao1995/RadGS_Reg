"""Online RecM + RegM optimization with projection and physical registration losses.

RecM(AP,LA) -> Gaussian field -> differentiable voxelizer -> RegM -> rigid CT warp.
Projection chunks accumulate gradients at Gaussian attributes, then one exact
vector-Jacobian product transfers all projection AND registration gradients to
RecM. The RecM image-gradient path remains online; staged modes explicitly stop
the pose-target gradient and isolate the RegM substep, without a cached RecM field.
"""
import argparse
import hashlib
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader,DistributedSampler,Subset
from torch.utils.tensorboard import SummaryWriter
from monai.data import list_data_collate

from networks.DVGSNet import DVGSNet
from Reg3D_train import checkpoint_model,atomic_json,atomic_save
from tools.data_contract import read_split_pairs
from gs_train import (seed_everything,move_batch,network_input,physical_gaussians,
                      query_volume,render_chunk,rng_state,restore_rng,detail_metrics)
from tools.registration_data import RegistrationDataset, sampled_transform, stable_seed
from tools.registration_geometry import centered_affine, invert_rigid, pose_matrix, pose_errors, warp_volume
from tools.registration_losses import registration_loss, ssim3d, central_crop
from tools.registration_augmentation import jitter_fixed_input
from r2_gaussian.dataset.GSDataset import GSDataset
from r2_gaussian.utils.loss_utils import ssim
from tools.joint_goal import gate as goal_gate


def load_recm(checkpoint,device):
    # A stopped joint run can supply its already-trained RecM to a new RegM.
    state=torch.load(checkpoint,map_location='cpu',mmap=True)
    if 'recm_model_config' in state:
        config=state['config']['reconstruction_contract']
        model=DVGSNet(**state['recm_model_config']).to(device)
        model.load_state_dict(state['recm']);del state
        return model,config
    config=json.loads((Path(checkpoint).parent/'config.json').read_text())
    model=DVGSNet(**config['model_config']).to(device)
    model.load_state_dict(state.get('model_state_dict',state));del state
    return model,config


def canvas(volume):
    # Identical to the standalone dataset exporter; maintains differentiability.
    return F.interpolate(F.pad(volume,(32,)*6),size=(96,)*3,mode='trilinear',align_corners=False)


def pair_metadata(batch,lookup,device,epoch,augment,seed):
    names=[item[0] if isinstance(item,(tuple,list)) else item for item in batch['sample_name']]
    # MONAI list_data_collate transposes nested one-element name lists.
    if len(names)!=batch['input_imgs'].shape[0]:
        names=list(batch['sample_name'][0])
    names=[Path(name).name.removesuffix('.pickle') for name in names]
    rows=[lookup[name] for name in names]
    if augment:
        matrix=torch.stack([sampled_transform(stable_seed(f'{name}:{epoch}',seed))[2] for name in names]).to(device)
    else:matrix=torch.tensor([r['moving_to_fixed'] for r in rows],device=device,dtype=torch.float32)
    # Fixed physical foreground landmarks are cached from the original 1mm mask.
    landmarks=torch.stack([torch.from_numpy(np.load(Path(lookup['_root'])/'landmarks'/f'{name}.npy')) for name in names]).to(device)
    return names,matrix,landmarks


def query_batch(attributes,batch):
    values=[query_volume(attributes,batch,i,128) for i in range(batch['input_imgs'].shape[0])]
    return torch.cat([v for v,g in values],0),torch.cat([g for v,g in values],0)


def reg_forward(regm,moving,fixed,affine,args):
    fixed_input=fixed
    rotation_noise=getattr(args,'fixed_input_rotation_noise_deg',0.)
    translation_noise=getattr(args,'fixed_input_translation_noise_mm',0.)
    if getattr(regm,'training',False) and (rotation_noise or translation_noise):
        # Corrupt only the encoder measurement; all image losses retain the
        # original field, and the input warp keeps its gradient attached to RecM.
        fixed_input,_=jitter_fixed_input(fixed,affine,rotation_noise,translation_noise)
    with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.amp_reg):pose=regm(moving,fixed_input)
    matrix=pose_matrix(pose.float());warped=warp_volume(moving,matrix,affine,affine)
    return matrix,warped


def registration_pair(reconstructed,canonical,truth,affine,args):
    """Same ResNet input FOV; optional original 1mm image loss and rigid resampling."""
    spacing=getattr(args,'registration_loss_spacing_mm',2.)
    network_size=getattr(args,'reg_input_size',96)
    if network_size not in (96,192):raise ValueError('Supported RegM network grids are96 or192 over the192mm canvas')
    if spacing==1.:
        fixed_loss=F.pad(reconstructed,(32,)*6)
        canonical_loss=F.pad(canonical,(32,)*6)
        affine_loss=centered_affine((192,)*3,1.,device=reconstructed.device,dtype=reconstructed.dtype)
        moving_loss=warp_volume(canonical_loss,invert_rigid(truth),affine_loss,affine_loss)
        moving_net=moving_loss if network_size==192 else F.interpolate(moving_loss,size=(96,)*3,mode='trilinear',align_corners=False)
        fixed_net=fixed_loss if network_size==192 else F.interpolate(fixed_loss,size=(96,)*3,mode='trilinear',align_corners=False)
        return moving_net,fixed_net,moving_loss,fixed_loss,affine_loss,128,11
    if spacing!=2.:raise ValueError('Registration loss spacing must be 1 or 2 mm')
    if network_size!=96:raise ValueError('Native192 RegM requires original1mm image fields')
    fixed_net=canvas(reconstructed)
    moving_net=warp_volume(canvas(canonical),invert_rigid(truth),affine,affine)
    return moving_net,fixed_net,moving_net,fixed_net,affine,64,7


def joint_step(recm,regm,batch,optimizer,contract,args,lookup,affine,epoch,gradient_audit=False):
    if getattr(args,'optimization_mode','simultaneous')!='simultaneous':
        from tools.joint_optimization import staged_joint_step
        return staged_joint_step(recm,regm,batch,optimizer,contract,args,lookup,affine,epoch,gradient_audit)
    optimizer.zero_grad(set_to_none=True)
    outputs=physical_gaussians(recm(network_input(batch,contract)),contract)
    # This boundary releases rendering graphs chunk by chunk; gradients are explicitly
    # transferred to the ORIGINAL outputs below. It does not freeze RecM or detach V_rec.
    leaves={k:v.detach().requires_grad_(True) for k,v in outputs.items()}
    rec_loss=batch['input_imgs'].new_zeros(());l1_sum=rec_loss.clone();dssim_sum=rec_loss.clone()
    if not gradient_audit:
        views=batch['render_imgs'].shape[1]
        for begin in range(0,views,args.view_chunk):
            end=min(begin+args.view_chunk,views)
            prediction,target=render_chunk(leaves,batch,begin,end)
            prediction=prediction*contract.projection_scale;target=target*contract.projection_scale
            l1=(prediction-target).abs().mean();dssim=1-ssim(prediction,target);fraction=(end-begin)/views
            value=(l1+contract.lambda_dssim*dssim)*fraction
            value.backward();rec_loss+=value.detach();l1_sum+=l1.detach()*fraction;dssim_sum+=dssim.detach()*fraction
    reconstructed,canonical=query_batch(leaves,batch)
    names,truth,landmarks=pair_metadata(batch,lookup,reconstructed.device,epoch,True,args.seed)
    moving,fixed,moving_loss,fixed_loss,affine_loss,crop_size,window=registration_pair(reconstructed,canonical,truth,affine,args)
    matrix,warped=reg_forward(regm,moving,fixed,affine,args)
    warped_loss=warp_volume(moving_loss,matrix,affine_loss,affine_loss) if crop_size==128 else warped
    loss,parts=registration_loss(warped_loss,fixed_loss,matrix,truth,geo_weight=args.geo_weight,crop_size=crop_size,ssim_window=window)
    (args.lambda_reg*loss).backward()
    gradients=[]
    for key,value in leaves.items():
        if value.grad is None or not torch.isfinite(value.grad).all():raise RuntimeError(f'Missing/nonfinite Gaussian gradient {key}')
        gradients.append(value.grad)
    torch.autograd.backward(tuple(outputs.values()),tuple(gradients))
    raw_recm=recm.module if isinstance(recm,DDP) else recm
    raw_regm=regm.module if isinstance(regm,DDP) else regm
    if gradient_audit:
        # Evidence at actual RecM parameters, with projection losses disabled entirely.
        checked={}
        for name,p in [('RecM_encoder',next(raw_recm.encoder.parameters())),('RecM_Gaussian_head',raw_recm.fc_layers[-1].weight),('RegM_pose_head',raw_regm.pose_head.layers[-1].weight)]:
            if p.grad is None or not torch.isfinite(p.grad).all() or not bool(p.grad.abs().sum()>0):raise RuntimeError(f'L_reg does not reach {name}')
            checked[name]=dict(norm=float(p.grad.norm()),max_abs=float(p.grad.abs().max()))
        optimizer.zero_grad(set_to_none=True)
        return dict(status='passed',projection_loss_disabled=True,loss=float(loss),parameters=checked,
                    fixed_requires_grad=fixed.requires_grad,gaussian_gradient_norms={k:float(v.grad.norm()) for k,v in leaves.items()})
    norm_rec=torch.nn.utils.clip_grad_norm_(raw_recm.parameters(),args.grad_clip,error_if_nonfinite=True)
    norm_reg=torch.nn.utils.clip_grad_norm_(raw_regm.parameters(),args.grad_clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(rec_loss+args.lambda_reg*loss.detach()),reconstruction_loss=float(rec_loss),
                projection_l1=float(l1_sum),projection_dssim=float(dssim_sum),registration_loss=float(loss),
                reg_ncc=float(parts['ncc']),reg_dssim=float(parts['dssim']),reg_geodesic=float(parts['geodesic']),
                recm_gradient_norm=float(norm_rec),regm_gradient_norm=float(norm_reg))


@torch.no_grad()
def validate(recm,regm,loader,contract,args,lookup,affine,epoch,writer=None,distributed=False):
    recm.eval();regm.eval();rows=[]
    for step,batch in enumerate(loader):
        batch=move_batch(batch,affine.device)
        attributes=physical_gaussians(recm(network_input(batch,contract)),contract)
        volume,gt=query_batch(attributes,batch)
        names,truth,landmarks=pair_metadata(batch,lookup,volume.device,epoch,False,args.seed)
        moving,fixed,moving_loss,fixed_loss,affine_loss,crop_size,window=registration_pair(volume,gt,truth,affine,args)
        matrix,warped=reg_forward(regm,moving,fixed,affine,args)
        errors=pose_errors(matrix,truth,landmarks)
        if not args.skip_projection_metrics:
            views=batch['render_imgs'].shape[1];n=volume.shape[0]
            projection={k:volume.new_zeros(n) for k in ('projection_ssim2d','projection_psnr2d','projection_l1')}
            for begin in range(0,views,args.view_chunk):
                end=min(begin+args.view_chunk,views)
                p,t=render_chunk(attributes,batch,begin,end)
                ps,ts=p*contract.projection_scale,t*contract.projection_scale
                projection['projection_ssim2d']+=ssim(ps,ts,size_average=False).reshape(n,end-begin).sum(1)/views
                mse=(ps-ts).square().flatten(1).mean(1).clamp_min(1e-12)
                projection['projection_psnr2d']+=(10*torch.log10(contract.metric_data_range**2/mse)).reshape(n,end-begin).sum(1)/views
                projection['projection_l1']+=(ps-ts).abs().flatten(1).mean(1).reshape(n,end-begin).sum(1)/views
            errors.update(projection)
        errors['reconstruction_ssim3d']=ssim3d(volume,gt,data_range=.04)
        errors['reconstruction_psnr3d']=10*torch.log10(.04**2/(volume-gt).square().flatten(1).mean(1).clamp_min(1e-16))
        shared=gt.flatten(1).max(1).values[:,None,None,None,None].clamp_min(1e-8)
        errors['reconstruction_ssim3d_shared_gt_peak']=ssim3d(volume/shared,gt/shared,data_range=1.)
        network_crop=128 if moving.shape[-1]==192 else 64
        network_window=11 if moving.shape[-1]==192 else 7
        errors['registration_ssim3d']=ssim3d(central_crop(warped,network_crop),central_crop(fixed,network_crop),window=network_window)
        if crop_size==128:
            warped_loss=warp_volume(moving_loss,matrix,affine_loss,affine_loss)
            errors['registration_ssim3d_original_1mm']=ssim3d(central_crop(warped_loss,128),central_crop(fixed_loss,128),window=11)
        for i,name in enumerate(names):
            nonzero=(gt[i,0]>0).nonzero()
            lo=(nonzero.min(0).values-5).clamp_min(0);hi=(nonzero.max(0).values+6).clamp_max(128)
            slices=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi))
            roi_ssim=ssim3d(volume[(slice(i,i+1),slice(None),*slices)],gt[(slice(i,i+1),slice(None),*slices)],data_range=.04)
            row=dict(name=name,reconstruction_ssim3d_foreground_roi=float(roi_ssim),**{k:float(v[i]) for k,v in errors.items()})
            rows.append(row)
        if writer is not None and step==0:
            for tag,value in [('rec_volume',volume),('ct_gt',gt),('fixed_canvas',fixed),('moving',moving),('registered',warped)]:
                image=value[0,0,:,:,value.shape[-1]//2].clamp(0,.04)/.04
                writer.add_image(f'val/{tag}',image[None],epoch)
            writer.add_histogram('val/physical_reconstruction_density',volume,epoch)
        if args.max_val_steps and step+1>=args.max_val_steps:break
    if distributed:
        gathered=[None for _ in range(dist.get_world_size())];dist.all_gather_object(gathered,rows)
        rows=[row for group in gathered for row in group]
    keys=[k for k in rows[0] if k!='name']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys},rows


def train(args):
    if not 0<=args.regm_warmup_epochs<=args.epochs:
        raise ValueError('RegM warmup must lie within the total training')
    if not 0<args.point_trim_fraction<=1 or not math.isfinite(args.point_loss_weight) or args.point_loss_weight<0:
        raise ValueError('Invalid point registration loss settings')
    if not math.isfinite(args.recm_target_foreground_weight) or not 0<=args.recm_target_foreground_weight<=1:
        raise ValueError('RecM target foreground weight must be in [0,1]')
    if not math.isfinite(args.recm_target_roi_margin_mm) or args.recm_target_roi_margin_mm<0:
        raise ValueError('RecM target ROI margin must be finite/nonnegative')
    if args.recm_target_foreground_weight and args.optimization_mode=='simultaneous':
        raise ValueError('Foreground target loss is a staged RecM-substep option')
    if any(not math.isfinite(getattr(args,key,0.)) or getattr(args,key,0.)<0
           for key in ('fixed_input_rotation_noise_deg','fixed_input_translation_noise_mm')):
        raise ValueError('Fixed-input measurement noise must be finite and nonnegative')
    world=int(os.environ.get('WORLD_SIZE',1));rank=int(os.environ.get('RANK',0));local=int(os.environ.get('LOCAL_RANK',0))
    torch.cuda.set_device(local);device=torch.device('cuda',local);torch.set_num_threads(args.cpu_threads)
    if world>1:dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    seed_everything(args.seed);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    raw_rec,rec_config=load_recm(args.recm_checkpoint,device);contract=SimpleNamespace(**rec_config)
    raw_reg,reg_state=checkpoint_model(args.regm_checkpoint,device);del reg_state
    reg_backbone=raw_reg.config['architecture']
    if args.regm_warmup_epochs and args.optimization_mode!='alternating':
        raise ValueError('RegM warmup requires alternating optimization')
    frozen_regm=args.optimization_mode=='frozen_regm'
    if frozen_regm:
        raw_reg.requires_grad_(False);raw_reg.eval()
    args.reg_input_size=raw_reg.input_size
    if args.fixed_input_rotation_noise_deg or args.fixed_input_translation_noise_mm:
        # Each rank samples independent corruptions; exact resume restores the
        # saved per-rank CUDA RNG state below.
        torch.cuda.manual_seed(args.seed+rank)
    recm=DDP(raw_rec,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 else raw_rec
    regm=DDP(raw_reg,device_ids=[local],broadcast_buffers=False,gradient_as_bucket_view=True) if world>1 and not frozen_regm else raw_reg
    if args.recm_encoder_lr is None:
        groups=[dict(params=raw_rec.parameters(),lr=args.recm_lr,name='RecM')]
    else:
        groups=[dict(params=raw_rec.encoder.parameters(),lr=args.recm_encoder_lr,name='RecM_encoder'),
                dict(params=[p for name,p in raw_rec.named_parameters() if not name.startswith('encoder.')],lr=args.recm_lr,name='RecM_Gaussian_head')]
    if not frozen_regm:groups.append(dict(params=raw_reg.parameters(),lr=args.regm_lr,name='RegM'))
    initial_lrs=[group['lr'] for group in groups]
    optimizer=torch.optim.Adam(groups)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=0.)
    manifest=json.loads((Path(args.reg_data_dir)/'manifest.json').read_text())
    if Path(manifest['protocol']['source_data']).resolve()!=Path(args.data_dir).resolve():raise ValueError('RecM and RegM dataset source mismatch')
    lookup={Path(r['filename']).stem:r for r in manifest['samples']};lookup['_root']=args.reg_data_dir
    train_paths,train_pts=read_split_pairs(args.data_dir,'train');val_paths,val_pts=read_split_pairs(args.data_dir,'val')
    train_ds=GSDataset(train_paths,train_pts,load_points=False);val_ds=GSDataset(val_paths,val_pts,load_points=False)
    sampler=DistributedSampler(train_ds,num_replicas=world,rank=rank,shuffle=True,seed=args.seed)
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,sampler=sampler,num_workers=0,collate_fn=list_data_collate)
    val_loader=DataLoader(Subset(val_ds,list(range(rank,len(val_ds),world))),batch_size=args.batch_size,num_workers=0,collate_fn=list_data_collate)
    affine=centered_affine((args.reg_input_size,)*3,192./args.reg_input_size,device=device)
    start=0;best_score=math.inf;global_step=0;best_reconstruction_roi=None;best_goal_roi=None
    if args.resume:
        state=torch.load(args.resume,map_location='cpu')
        if state['world_size']!=world:raise ValueError('Exact resume requires same world size')
        previous=state['config']
        if args.optimization_mode!=previous.get('optimization_mode','simultaneous'):
            raise ValueError('Resume must preserve the optimization mode')
        if args.recm_target_pose!=previous.get('recm_target_pose','predicted'):
            raise ValueError('Resume must preserve the saved RecM target pose source')
        for key,default in [('recm_target_foreground_weight',0.),('recm_target_roi_margin_mm',5.)]:
            if getattr(args,key)!=previous.get(key,default):raise ValueError('Resume must preserve '+key)
        for key,default in [('point_loss_weight',1.),('point_trim_fraction',.9),('regm_warmup_epochs',0)]:
            if getattr(args,key)!=previous.get(key,default):raise ValueError('Resume must preserve '+key)
        current_reg_config=dict(raw_reg.config,pose_head_activation=raw_reg.config.get('pose_head_activation','relu'))
        saved_reg_config=dict(state['regm_model_config'],pose_head_activation=state['regm_model_config'].get('pose_head_activation','relu'))
        if current_reg_config!=saved_reg_config or rec_config['model_config']!=state['recm_model_config']:
            raise ValueError('Resume must preserve both saved model configurations and RegM input resolution')
        if args.epochs<=state['epoch']:raise ValueError('Requested total epochs must exceed the resume epoch')
        if args.registration_loss_spacing_mm!=previous.get('registration_loss_spacing_mm',2.):
            raise ValueError('Resume must keep the saved registration loss spacing; set --registration-loss-spacing-mm explicitly')
        if args.seed!=previous.get('seed',42):raise ValueError('Resume must keep the saved augmentation seed')
        for key in ('fixed_input_rotation_noise_deg','fixed_input_translation_noise_mm'):
            if getattr(args,key)!=previous.get(key,0.):raise ValueError(f'Resume must keep saved {key}; pass the saved input-noise settings explicitly')
        for key in ('data_dir','reg_data_dir'):
            if Path(getattr(args,key)).resolve()!=Path(previous[key]).resolve():raise ValueError(f'Resume {key} differs from the saved dataset')
        raw_rec.load_state_dict(state['recm']);raw_reg.load_state_dict(state['regm']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];global_step=state['global_step'];best_score=state['best_score'];best_reconstruction_roi=state.get('best_reconstruction_roi');best_goal_roi=state.get('best_goal_roi');restore_rng(state['rng_states'][rank])
        if rank==0:
            # A best model may be newer than the last periodic optimizer state.
            # Retain it and its threshold when recovering or extending the run.
            source=Path(args.resume).resolve().parent
            for label,metadata in [('best','best_validation'),('best_reconstruction','best_reconstruction_validation'),('best_goal','best_goal_validation')]:
                if source!=out.resolve():
                    for filename in (f'{label}.pth',f'{metadata}.json'):
                        old=source/filename;target=out/filename
                        if old.is_file() and not target.exists():shutil.copy2(old,target)
                info=out/f'{metadata}.json'
                if info.is_file() and (out/f'{label}.pth').is_file():
                    retained=json.loads(info.read_text())
                    if label=='best':best_score=min(best_score,retained['score'])
                    elif label=='best_reconstruction':best_reconstruction_roi=max(best_reconstruction_roi if best_reconstruction_roi is not None else -math.inf,retained['metrics']['reconstruction_ssim3d_foreground_roi'])
                    else:best_goal_roi=max(best_goal_roi if best_goal_roi is not None else -math.inf,retained['metrics']['reconstruction_ssim3d_foreground_roi'])
            old_cases=source/'best_validation_cases.json'
            if source!=out.resolve() and old_cases.is_file() and not (out/old_cases.name).exists():shutil.copy2(old_cases,out/old_cases.name)
        if world>1:
            thresholds=[(best_score,best_reconstruction_roi,best_goal_roi)]
            dist.broadcast_object_list(thresholds,src=0)
            best_score,best_reconstruction_roi,best_goal_roi=thresholds[0]
        if args.epochs>state['config']['epochs']:
            for group,lr in zip(optimizer.param_groups,initial_lrs):group['lr']=lr;group['initial_lr']=lr
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs-start,eta_min=0.)
        del state
    else:seed_everything(args.seed+rank)
    baseline_metrics=None
    if args.baseline_validation_metrics:
        saved_baseline=json.loads(Path(args.baseline_validation_metrics).read_text())
        baseline_metrics=saved_baseline['results']['baseline']['metrics']
    import xray_gaussian_rasterization_voxelization as native
    native_path=Path(native._C.__file__).resolve()
    config=dict(vars(args),world_size=world,global_batch_size=args.batch_size*world,optimizer_groups=[dict(name=group.get('name',str(i)),initial_lr=lr) for i,(group,lr) in enumerate(zip(optimizer.param_groups,initial_lrs))],
        geometry_precision='float32 physical matrices/grids with TF32 disabled locally; network precision unchanged',
        train_count=len(train_ds),val_count=len(val_ds),native_module=str(native_path),native_sha256=hashlib.file_digest(native_path.open('rb'),'sha256').hexdigest(),recm_model_config=rec_config['model_config'],regm_model_config=raw_reg.config,
        reconstruction_contract=rec_config,loss='group12 projection L1 + .2 DSSIM; stopped CT target NCC+SSIM; RegM warped-volume NCC+SSIM + normalized SE3; point RegM additionally trimmed symmetric point Huber',
        gradient_transfer='chunked attribute VJP; online V_rec stays attached; L_reg-only parameter gradient audit mandatory',
        optimization_protocol=dict(mode=args.optimization_mode,teacher_input_noise=False if args.optimization_mode!='simultaneous' else None,
            recm_steps_per_batch=1,regm_steps_per_batch=0 if frozen_regm else 1,
            alternating_uses_fresh_recm=args.optimization_mode=='alternating',target_gradient_stopped=args.optimization_mode!='simultaneous'),
        recm_target_roi=dict(foreground_weight=args.recm_target_foreground_weight,margin_mm=args.recm_target_roi_margin_mm,
            source='Stopped aligned moving CT >0, per-case bbox; minimum SSIM window; empty target falls back to full crop',
            loss='Convex combination of full-field and per-case ROI NCC+3DSSIM; RegM substep and all evaluation metrics unchanged'),
        partial_run=bool(args.max_train_steps or args.max_val_steps),
        diagnostic_only=False,
        baseline_validation_metrics_values=baseline_metrics,
        selection='validation only: best.pth uses maximum axis MAE with full SSIM penalty; best_reconstruction.pth maximizes foreground ROI SSIM; best_goal.pth requires paired full-volume SSIM relative gain >=1%, foreground gain and each pose axis<1')
    if rank==0:atomic_json(out/'config.json',config)
    writer=SummaryWriter(str(out/'tensorboard')) if rank==0 else None
    audit_done=False;started=time.time()
    for epoch in range(start+1,args.epochs+1):
        sampler.set_epoch(epoch);recm.train();regm.train(not frozen_regm)
        warmup=epoch<=args.regm_warmup_epochs
        # Preserve reliable pretraining running moments at the smaller per-rank batch.
        if not args.update_reg_bn:
            for module in raw_reg.modules():
                if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):module.eval()
        totals={};seen=0;epoch_start=time.time()
        for step,batch in enumerate(train_loader):
            batch=move_batch(batch,device)
            if not audit_done:
                saved_rng=rng_state()
                saved_buffers=[(buffer,buffer.clone()) for model in (raw_rec,raw_reg) for buffer in model.buffers()]
                try:audit=joint_step(recm,regm,batch,optimizer,contract,args,lookup,affine,epoch,True)
                finally:
                    restore_rng(saved_rng)
                    with torch.no_grad():
                        for buffer,value in saved_buffers:buffer.copy_(value)
                if rank==0:atomic_json(out/'joint_gradient_audit.json',audit)
                audit_done=True
                if args.gradient_check_only:
                    if writer:writer.close()
                    if world>1:dist.destroy_process_group()
                    return
            if warmup:
                from tools.joint_optimization import registration_warmup_step
                result=registration_warmup_step(recm,regm,batch,optimizer,contract,args,lookup,affine,epoch)
            else:result=joint_step(recm,regm,batch,optimizer,contract,args,lookup,affine,epoch)
            n=batch['input_imgs'].shape[0];seen+=n;global_step+=1
            for k,v in result.items():totals[k]=totals.get(k,0.)+v*n
            if writer is not None and global_step%args.log_interval==0:
                for k,v in result.items():writer.add_scalar(f'train/{k}',v,global_step)
                heartbeat=dict(status='running',phase='train',epoch=epoch-1,current_epoch=epoch,epochs=args.epochs,
                    step_in_epoch=step+1,steps_per_epoch=len(train_loader),global_step=global_step,
                    elapsed_seconds=time.time()-started,latest_train=result)
                atomic_json(out/'progress.json',heartbeat)
                print(json.dumps(dict(event='train_progress',**heartbeat)),flush=True)
            if args.max_train_steps and step+1>=args.max_train_steps:break
        packed=torch.tensor([*totals.values(),seen],device=device,dtype=torch.float64)
        if world>1:dist.all_reduce(packed)
        averaged={k:float(packed[i]/packed[-1]) for i,k in enumerate(totals)}
        scheduler.step();record=dict(epoch=epoch,global_step=global_step,train=averaged,epoch_seconds=time.time()-epoch_start)
        improved=False;improved_reconstruction=False;improved_goal=False
        if epoch==1 or epoch%args.val_interval==0 or epoch==args.epochs:
            if rank==0:atomic_json(out/'progress.json',dict(status='running',phase='validation',epoch=epoch-1,current_epoch=epoch,epochs=args.epochs,global_step=global_step,elapsed_seconds=time.time()-started))
            metrics,rows=validate(raw_rec,raw_reg,val_loader,contract,args,lookup,affine,epoch,writer,world>1)
            record['validation']=metrics
            pose_score=max(metrics[f'translation_{a}_mm'] for a in 'xyz')
            pose_score=max(pose_score,*(metrics[f'rotation_{a}_deg'] for a in 'xyz'))
            score=pose_score+max(0.,.9-metrics['reconstruction_ssim3d'])*100
            roi=metrics['reconstruction_ssim3d_foreground_roi']
            if best_reconstruction_roi is None or roi>best_reconstruction_roi:
                best_reconstruction_roi=roi;improved_reconstruction=True
            if not config['diagnostic_only'] and goal_gate(metrics,baseline_metrics)['passed'] and (best_goal_roi is None or roi>best_goal_roi):
                best_goal_roi=roi;improved_goal=True
            if score<best_score:
                best_score=score;improved=True
                if rank==0:
                    atomic_json(out/'best_validation.json',dict(epoch=epoch,score=score,metrics=metrics))
                    atomic_json(out/'best_validation_cases.json',rows)
            if writer:
                for k,v in metrics.items():writer.add_scalar(f'val/{k}',v,epoch)
        if rank==0:
            if improved:
                atomic_save(out/'best.pth',dict(recm=raw_rec.state_dict(),regm=raw_reg.state_dict(),recm_model_config=rec_config['model_config'],
                    regm_model_config=raw_reg.config,backbone=reg_backbone,epoch=epoch,validation=record['validation'],config=config))
            if improved_reconstruction or improved_goal:
                snapshot=dict(recm=raw_rec.state_dict(),regm=raw_reg.state_dict(),recm_model_config=rec_config['model_config'],
                    regm_model_config=raw_reg.config,backbone=reg_backbone,epoch=epoch,validation=record['validation'],config=config)
                for label,selected in [('best_reconstruction',improved_reconstruction),('best_goal',improved_goal)]:
                    if selected:
                        atomic_save(out/f'{label}.pth',snapshot)
                        atomic_json(out/f'{label}_validation.json',dict(epoch=epoch,metrics=record['validation']))
            print(json.dumps(record),flush=True)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            atomic_json(out/'progress.json',dict(status='completed' if epoch==args.epochs else 'running',epoch=epoch,epochs=args.epochs,global_step=global_step,
                elapsed_seconds=time.time()-started,best_score=best_score,latest=record))
            if writer:writer.flush()
        if epoch%args.checkpoint_interval==0 or epoch==args.epochs:
            states=[None for _ in range(world)] if world>1 else [rng_state()]
            if world>1:dist.all_gather_object(states,rng_state())
            if rank==0:atomic_save(out/'latest.pt',dict(recm=raw_rec.state_dict(),regm=raw_reg.state_dict(),recm_model_config=rec_config['model_config'],regm_model_config=raw_reg.config,backbone=reg_backbone,
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,global_step=global_step,best_score=best_score,best_reconstruction_roi=best_reconstruction_roi,best_goal_roi=best_goal_roi,world_size=world,rng_states=states,config=config))
        if world>1:dist.barrier()
    if writer:writer.close()
    if world>1:dist.destroy_process_group()


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-dir','reg-data-dir','recm-checkpoint','regm-checkpoint','output-dir'):p.add_argument('--'+name,required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--optimization-mode',choices=['simultaneous','frozen_regm','alternating'],default='simultaneous')
    p.add_argument('--recm-target-pose',choices=['predicted'],default='predicted',help='Use the predicted moving-to-fixed pose for reconstruction supervision')
    p.add_argument('--recm-target-foreground-weight',type=float,default=0.,help='Staged RecM image loss: convex foreground ROI fraction [0,1]; 0 preserves the full-field control exactly')
    p.add_argument('--recm-target-roi-margin-mm',type=float,default=5.,help='Aligned CT foreground bbox margin; no prediction-derived mask or intensity normalization')
    p.add_argument('--baseline-validation-metrics',help='Paired frozen baseline evaluation.json for the 1%% volume SSIM goal')
    p.add_argument('--recm-lr',type=float,default=1e-6);p.add_argument('--regm-lr',type=float,default=1e-5)
    p.add_argument('--recm-encoder-lr',type=float,default=None,help='Optional encoder LR; Gaussian head retains --recm-lr, with all end-to-end gradients enabled')
    p.add_argument('--lambda-reg',type=float,default=1.);p.add_argument('--geo-weight',type=float,default=.02)
    p.add_argument('--point-loss-weight',type=float,default=1.)
    p.add_argument('--point-trim-fraction',type=float,default=.9)
    p.add_argument('--regm-warmup-epochs',type=int,default=0,help='First N of total epochs: train new RegM with RecM fixed')
    p.add_argument('--amp-reg',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--update-reg-bn',action='store_true')
    p.add_argument('--fixed-input-rotation-noise-deg',type=float,default=0.,help='Training-only fixed encoder input uncertainty; real CT pose/image target and evaluation inputs remain unchanged')
    p.add_argument('--fixed-input-translation-noise-mm',type=float,default=0.,help='Training-only fixed encoder input uncertainty in physical millimetres')
    p.add_argument('--view-chunk',type=int,default=25);p.add_argument('--grad-clip',type=float,default=5.)
    p.add_argument('--registration-loss-spacing-mm',type=float,choices=[1.,2.],default=1.,help='Physical spacing for the registration image loss; point inputs share a 192mm canvas')
    p.add_argument('--seed',type=int,default=42);p.add_argument('--cpu-threads',type=int,default=2)
    p.add_argument('--val-interval',type=int,default=2);p.add_argument('--checkpoint-interval',type=int,default=5);p.add_argument('--log-interval',type=int,default=20)
    p.add_argument('--max-train-steps',type=int,default=0);p.add_argument('--max-val-steps',type=int,default=0)
    p.add_argument('--gradient-check-only',action='store_true');p.add_argument('--skip-projection-metrics',action='store_true');p.add_argument('--resume')
    return p.parse_args()


if __name__=='__main__':train(parse_args())
