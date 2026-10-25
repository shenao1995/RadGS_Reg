"""Standalone RegM training: point-cloud rigid registration with CT or frozen RecM pairs.

No synthetic output volume, independent split, CAL, Chamfer or volume-shape loss.
The predicted SE(3) matrix rigidly resamples moving CT in physical millimetres.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from networks.pointRegM import PointRegM, make_regm
from tools.point_registration import point_volume_registration_loss
from tools.registration_data import RegistrationDataset, prepare_pair
from tools.registration_geometry import pose_matrix, pose_errors, warp_volume
from tools.registration_losses import registration_loss, central_crop, ssim3d
from tools.registration_augmentation import augment_fixed_input, jitter_fixed_input


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def atomic_save(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)


def seed_everything(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def checkpoint_model(path,device='cuda'):
    state=torch.load(path,map_location='cpu')
    # Early joint latest.pt files omitted the backbone label but contain both
    # named model states and the explicit RegM constructor configuration.
    legacy_joint=all(isinstance(state.get(key),dict) for key in ('recm','regm','regm_model_config'))
    if state.get('backbone') not in ('point_residual34',) and not (state.get('backbone') is None and legacy_joint):
        raise ValueError('Expected a point RegM checkpoint')
    config=state['model_config'] if 'model_config' in state else state['regm_model_config']
    weights=state['model'] if 'model' in state else state['regm']
    model=make_regm(config).to(device);model.load_state_dict(weights,strict=True)
    return model,state


def summarize_rows(rows):
    keys=[k for k in rows[0] if k!='name']
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}


@torch.no_grad()
def evaluate(model,loader,device,affine,args,fixed_mode='rec',writer=None,epoch=0):
    model.eval();rows=[]
    for step,batch in enumerate(loader):
        moving,fixed,truth=prepare_pair(batch,device,affine,fixed_mode=fixed_mode)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.amp):
            pose=model(moving,fixed)
        matrix=pose_matrix(pose.float());warped=warp_volume(moving,matrix,affine,affine)
        errors=pose_errors(matrix,truth,batch['landmarks'].to(device))
        errors['aligned_ssim3d']=ssim3d(central_crop(warped,64),central_crop(fixed,64),window=7)
        canonical=batch['canonical'].to(device)
        errors['aligned_to_ct_ssim3d']=ssim3d(central_crop(warped,64),central_crop(canonical,64),window=7)
        for i,name in enumerate(batch['name']):
            rows.append(dict(name=name,**{key:float(value[i]) for key,value in errors.items()}))
        if writer is not None and step==0:
            for tag,value in [('fixed',fixed),('moving',moving),('warped',warped),('canonical',canonical),('abs_error',(warped-fixed).abs())]:
                # Same fixed physical window for all panels and all epochs.
                img=value[0,0,:,:,value.shape[-1]//2].float().clamp(0,.04)/.04
                writer.add_image(f'val/{tag}_RAS_axial',img[None],epoch)
    return summarize_rows(rows),rows


def train(args):
    if not args.fixed_input_blur_mm or any(not math.isfinite(value) or value<0 for value in args.fixed_input_blur_mm):
        raise ValueError('Fixed-input blur sigmas must be finite and nonnegative')
    if any(not math.isfinite(value) or value<0 for value in (args.fixed_input_rotation_noise_deg,args.fixed_input_translation_noise_mm)):
        raise ValueError('Fixed-input pose noise must be finite and nonnegative')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('RegM training requires CUDA; submit a GPU job')
    torch.set_num_threads(args.cpu_threads);torch.cuda.set_device(0);seed_everything(args.seed)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=True
    device=torch.device('cuda')
    train_ds=RegistrationDataset(args.data_dir,'train',args.limit_train)
    val_ds=RegistrationDataset(args.data_dir,'val',args.limit_val)
    generator=torch.Generator().manual_seed(args.seed)
    options=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=True)
    if args.workers:options.update(persistent_workers=True,prefetch_factor=2)
    train_loader=DataLoader(train_ds,shuffle=True,generator=generator,**options)
    val_loader=DataLoader(val_ds,shuffle=False,**options)
    affine=train_ds.affine.to(device)
    loaded=None
    if args.init or args.resume:
        model,loaded=checkpoint_model(args.resume or args.init,device)
    else:
        model=PointRegM(input_size=96).to(device)
    if not args.resume:
        del loaded
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.05)
    start=0;best=math.inf;global_step=0
    if args.resume:
        state=loaded;optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];global_step=state['global_step'];best=state['best_score']
        generator.set_state(state['loader_rng']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        # A continuation in a new output directory retains the already achieved
        # best checkpoint even if no later validation improves it.
        source_dir=Path(args.resume).resolve().parent
        if source_dir!=out.resolve():
            for filename in ('best.pth','best_validation.json','best_validation_cases.json'):
                source=source_dir/filename;destination=out/filename
                if source.is_file() and not destination.exists():shutil.copy2(source,destination)
        previous_total=state['config']['epochs']
        if args.epochs<start:raise ValueError('Requested total epochs precedes the resume checkpoint')
        if args.epochs>previous_total:
            # An authorized extension gets its own cosine period. Loading the old
            # T_max would otherwise make the LR rise again after the old endpoint.
            for group in optimizer.param_groups:group['lr']=args.lr;group['initial_lr']=args.lr
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs-start,eta_min=args.lr*.05)
    config=dict(vars(args),backbone='point_residual34',model_config=model.config,
                geometry_precision='float32 physical matrices/grids with TF32 disabled locally; encoder precision unchanged',
                residual_depths=[3,4,6,3],parameters=sum(p.numel() for p in model.parameters()),
                train_count=len(train_ds),val_count=len(val_ds),protocol=train_ds.protocol,
                pose_units='axis-angle rad + RAS mm; active moving -> fixed; inverse map for sampling',
                selection='validation maximum of six per-axis MAEs / target (1mm or 1degree); test never used for checkpoint selection',
                image_loss='NCC + true 3D Gaussian SSIM(7^3) on central 128mm cube of 192mm canvas; fixed range .04',
                geodesic='SO(3) angle/(5deg) + Euclidean translation/(10mm)',
                augmentation='new bounded known pose per training case and epoch; fixed known val/test poses',
                fixed_input_quality_augmentation=dict(sigmas_mm=args.fixed_input_blur_mm,training_only=True,
                    image_loss_uses_original_fixed=True,validation_and_inference_unmodified=True),
                fixed_input_pose_noise=dict(rotation_each_axis_deg=args.fixed_input_rotation_noise_deg,
                    translation_each_axis_mm=args.fixed_input_translation_noise_mm,training_only=True,
                    moving_CT_and_true_pose_unmodified=True,image_loss_uses_original_fixed=True),
                compute=dict(gpu=torch.cuda.get_device_name(0),torch=torch.__version__,amp_bfloat16=args.amp))
    atomic_json(out/'config.json',config)
    history_path=out/'history.jsonl';started=time.time()
    with SummaryWriter(str(out/'tensorboard'),purge_step=start if args.resume else None) as writer:
        for epoch in range(start+1,args.epochs+1):
            model.train();totals={};seen=0;epoch_start=time.time()
            for step,batch in enumerate(train_loader):
                moving,fixed,truth=prepare_pair(batch,device,affine,epoch=epoch,augment=not args.static_train,
                    seed=args.seed,fixed_mode=args.fixed_mode,curriculum_epochs=args.curriculum_epochs)
                fixed_input,blur_sigma=augment_fixed_input(fixed,train_ds.protocol['cached_spacing_mm'],args.fixed_input_blur_mm)
                fixed_input,_=jitter_fixed_input(fixed_input,affine,args.fixed_input_rotation_noise_deg,args.fixed_input_translation_noise_mm)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=args.amp):pose=model(moving,fixed_input)
                matrix=pose_matrix(pose.float());warped=warp_volume(moving,matrix,affine,affine)
                loss,parts=point_volume_registration_loss(model,moving,fixed,affine,warped,fixed,matrix,truth,args,64,7)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite RegM loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip,error_if_nonfinite=True)
                optimizer.step();global_step+=1;n=len(moving);seen+=n
                for k,v in parts.items():totals[k]=totals.get(k,0.)+float(v)*n
                if global_step%args.log_interval==0:
                    for k,v in parts.items():writer.add_scalar(f'train/{k}',float(v),global_step)
                    writer.add_scalar('train/gradient_norm',float(norm),global_step)
                    writer.add_scalar('train/fixed_input_blur_sigma_mm',blur_sigma,global_step)
                    heartbeat=dict(status='running',phase='train',epoch=epoch-1,current_epoch=epoch,epochs=args.epochs,
                        step_in_epoch=step+1,steps_per_epoch=len(train_loader),global_step=global_step,
                        elapsed_seconds=time.time()-started,best_validation_score=best if math.isfinite(best) else None,latest_train={k:float(v) for k,v in parts.items()})
                    atomic_json(out/'progress.json',heartbeat)
                    print(json.dumps(dict(event='train_progress',**heartbeat)),flush=True)
            scheduler.step();values={f'train_{k}':v/seen for k,v in totals.items()}
            record=dict(epoch=epoch,global_step=global_step,lr=scheduler.get_last_lr()[0],epoch_seconds=time.time()-epoch_start,**values)
            writer.add_scalar('train/epoch_loss',values['train_loss'],epoch)
            validation=None
            if epoch%args.val_interval==0 or epoch==args.epochs or epoch==1:
                # Always select against reconstructed fixed inputs, including clean-CT pretraining.
                validation,rows=evaluate(model,val_loader,device,affine,args,writer=writer,epoch=epoch)
                record['validation']=validation
                for key,value in validation.items():writer.add_scalar(f'val/{key}',value,epoch)
                pose_keys=[f'translation_{a}_mm' for a in 'xyz']+[f'rotation_{a}_deg' for a in 'xyz']
                score=max(validation[k] for k in pose_keys)
                if score<best:
                    best=score
                    atomic_save(out/'best.pth',dict(model=model.state_dict(),model_config=model.config,backbone='point_residual34',epoch=epoch,validation=validation,config=config))
                    atomic_json(out/'best_validation.json',dict(epoch=epoch,score=score,metrics=validation))
                    atomic_json(out/'best_validation_cases.json',rows)
            with history_path.open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            state=dict(model=model.state_dict(),model_config=model.config,backbone='point_residual34',epoch=epoch,
                optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),global_step=global_step,best_score=best,
                config=config,loader_rng=generator.get_state(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
            atomic_save(out/'latest.pt',state)
            atomic_json(out/'progress.json',dict(status='running' if epoch<args.epochs else 'completed',epoch=epoch,epochs=args.epochs,global_step=global_step,
                elapsed_seconds=time.time()-started,best_validation_score=best,latest_validation=validation))
            writer.flush()
    return best


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--fixed-mode',choices=['ct','rec','curriculum'],default='curriculum');p.add_argument('--curriculum-epochs',type=int,default=10)
    p.add_argument('--point-loss-weight',type=float,default=1.)
    p.add_argument('--point-trim-fraction',type=float,default=.9)
    p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--weight-decay',type=float,default=1e-5)
    p.add_argument('--geo-weight',type=float,default=.02);p.add_argument('--ssim-weight',type=float,default=1.);p.add_argument('--ncc-weight',type=float,default=1.)
    p.add_argument('--workers',type=int,default=2);p.add_argument('--cpu-threads',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--val-interval',type=int,default=2);p.add_argument('--log-interval',type=int,default=20);p.add_argument('--grad-clip',type=float,default=5.)
    p.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--static-train',action='store_true')
    p.add_argument('--fixed-input-blur-mm',nargs='+',type=float,default=[0.],
                   help='Training-only fixed encoder input blur: choose a listed physical sigma each batch; image loss/validation/inference use the original fixed field')
    p.add_argument('--fixed-input-rotation-noise-deg',type=float,default=0.,
                   help='Training-only reconstruction-input uncertainty, uniform per-axis degrees; never changes the true CT pose or image-loss target')
    p.add_argument('--fixed-input-translation-noise-mm',type=float,default=0.,
                   help='Training-only reconstruction-input uncertainty, uniform per-axis mm; validation/inference use original fields')
    p.add_argument('--init');p.add_argument('--resume');p.add_argument('--limit-train',type=int,default=0);p.add_argument('--limit-val',type=int,default=0)
    p.add_argument('--native-1mm-cache',action='store_true',help='Use audited native128 CT/RecM pairs, original1mm warp/image loss and optional single-node DDP')
    p.add_argument('--input-size',type=int,choices=[96,192],default=None,help='Native-cache training only; preserve the checkpoint value when omitted')
    p.add_argument('--max-train-steps',type=int,default=0,help='Limit native-cache training steps; checkpoints record partial_run')
    p.add_argument('--max-val-steps',type=int,default=0,help='Limit native-cache training steps; checkpoints record partial_run')
    return p.parse_args()


if __name__=='__main__':
    args=parse_args()
    if args.native_1mm_cache:
        from registration_native_train import train_native
        train_native(args)
    else:
        if int(os.environ.get('WORLD_SIZE','1'))>1 or args.input_size is not None or args.max_train_steps or args.max_val_steps:
            raise ValueError('Distributed/input-size/smoke options require --native-1mm-cache')
        train(args)
