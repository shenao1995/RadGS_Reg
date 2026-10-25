"""Standalone native-1mm RegM with point residual blocks, DDP and exact validation membership."""
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter

from networks.pointRegM import PointRegM
from tools.point_registration import point_volume_registration_loss
from Reg3D_train import atomic_json, atomic_save, checkpoint_model, seed_everything, summarize_rows
from tools.registration_augmentation import augment_fixed_input, jitter_fixed_input
from tools.registration_geometry import pose_matrix, pose_errors, warp_volume
from tools.registration_losses import central_crop, registration_loss, ssim3d
from tools.registration_native_data import NativeRegistrationDataset, prepare_native_pair


def capture_rank_rng(generator):
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(), loader=generator.get_state())


def restore_rank_rng(state, generator):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state(state['cuda'])
    generator.set_state(state['loader'])


def validation_indices(length, rank, world):
    # Validation must never use the padding/duplication of DistributedSampler.
    return list(range(rank, length, world))


@torch.no_grad()
def evaluate_native(model, loader, device, affine, args, fixed_mode='rec', writer=None, epoch=0):
    model.eval(); rows=[]
    for step, batch in enumerate(loader):
        moving, fixed, truth = prepare_native_pair(batch, device, affine, fixed_mode=fixed_mode)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=args.amp):
            pose = model(moving, fixed)
        matrix=pose_matrix(pose.float()); warped=warp_volume(moving, matrix, affine, affine)
        errors=pose_errors(matrix, truth, batch['landmarks'].to(device))
        canonical=F.pad(batch['canonical'].to(device=device, dtype=torch.float32), (32,)*6)
        errors['aligned_ssim3d']=ssim3d(central_crop(warped,128), central_crop(fixed,128), window=11)
        errors['aligned_to_ct_ssim3d']=ssim3d(central_crop(warped,128), central_crop(canonical,128), window=11)
        for i, name in enumerate(batch['name']):
            rows.append(dict(name=name, **{key:float(value[i]) for key,value in errors.items()}))
        if writer is not None and step==0:
            for tag,value in [('fixed',fixed),('moving',moving),('warped',warped),('canonical',canonical),('abs_error',(warped-fixed).abs())]:
                image=value[0,0,:,:,value.shape[-1]//2].float().clamp(0,.04)/.04
                writer.add_image(f'val/{tag}_RAS_axial',image[None],epoch)
        if getattr(args,'max_val_steps',0) and step+1>=args.max_val_steps:
            break
    if dist.is_initialized():
        gathered=[None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered,rows)
        rows=[row for group in gathered for row in group]
    if not rows or len({row['name'] for row in rows})!=len(rows):
        raise RuntimeError('Empty or duplicated native validation cases')
    return summarize_rows(rows), rows


def train_native(args):
    if not args.native_1mm_cache:
        raise ValueError('Native trainer requires --native-1mm-cache')
    if not args.fixed_input_blur_mm or any(not math.isfinite(v) or v<0 for v in args.fixed_input_blur_mm):
        raise ValueError('Invalid blur sigma')
    for key in ('fixed_input_rotation_noise_deg','fixed_input_translation_noise_mm'):
        if not math.isfinite(getattr(args,key)) or getattr(args,key)<0:
            raise ValueError('Invalid fixed-input uncertainty')
    world=int(os.environ.get('WORLD_SIZE','1')); rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    if int(os.environ.get('SLURM_JOB_NUM_NODES','1'))!=1:
        raise ValueError('Native experiment requires a single node')
    torch.set_num_threads(args.cpu_threads); torch.cuda.set_device(local); device=torch.device('cuda',local)
    if world>1:
        dist.init_process_group('nccl',timeout=timedelta(minutes=30))
    seed_everything(args.seed)
    torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    if not args.resume and ((out/'latest.pt').exists() or (out/'history.jsonl').exists()):
        raise RuntimeError('Existing training output requires explicit resume')
    train_ds=NativeRegistrationDataset(args.data_dir,'train',args.limit_train)
    val_ds=NativeRegistrationDataset(args.data_dir,'val',args.limit_val)
    state=None
    if args.init or args.resume:
        raw,state=checkpoint_model(args.resume or args.init,device)
    else:
        raw=PointRegM().to(device)
    requested=args.input_size if args.input_size is not None else raw.input_size
    if args.resume and requested!=raw.input_size:
        raise ValueError('Resume must preserve network resolution')
    raw.input_size=requested; raw.config['input_size']=requested
    model=DDP(raw,device_ids=[local],broadcast_buffers=True,gradient_as_bucket_view=True) if world>1 else raw
    # Initialization is shared; measurement corruptions must be independent per rank.
    seed_everything(args.seed+rank)
    generator=torch.Generator().manual_seed(args.seed+rank)
    sampler=DistributedSampler(train_ds,num_replicas=world,rank=rank,shuffle=True,seed=args.seed)
    options=dict(batch_size=args.batch_size,num_workers=args.workers,pin_memory=True,generator=generator)
    if args.workers:
        options.update(persistent_workers=True,prefetch_factor=2)
    train_loader=DataLoader(train_ds,sampler=sampler,**options)
    val_loader=DataLoader(Subset(val_ds,validation_indices(len(val_ds),rank,world)),shuffle=False,**options)
    affine=train_ds.affine.to(device)
    optimizer=torch.optim.AdamW(raw.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs,eta_min=args.lr*.05)
    start=0;global_step=0;best=math.inf
    if args.resume:
        if state.get('world_size')!=world or len(state.get('rank_rng_states',[]))!=world:
            raise ValueError('Native resume requires same world size and all saved rank states; use --init for old formats')
        previous=state['config']
        for key in ('data_dir','batch_size','seed','fixed_mode','curriculum_epochs','static_train',
                    'geo_weight','ssim_weight','ncc_weight','point_loss_weight','point_trim_fraction','fixed_input_blur_mm',
                    'fixed_input_rotation_noise_deg','fixed_input_translation_noise_mm',
                    'limit_train','limit_val','max_train_steps','max_val_steps','amp','weight_decay','grad_clip'):
            if getattr(args,key)!=previous.get(key):
                raise ValueError('Resume changes the saved experiment: '+key)
        if train_ds.protocol!=previous['protocol']:
            raise ValueError('Resume changes the saved dataset protocol')
        if raw.config!=state['model_config']:
            raise ValueError('Resume changes the saved model configuration')
        optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        start=state['epoch'];global_step=state['global_step'];best=state['best_score']
        if args.epochs<=start or args.epochs<previous['epochs']:
            raise ValueError('Resume total epochs must preserve or extend the original schedule')
        restore_rank_rng(state['rank_rng_states'][rank],generator)
        if args.epochs>previous['epochs']:
            for group in optimizer.param_groups:
                group['lr']=args.lr;group['initial_lr']=args.lr
            scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs-start,eta_min=args.lr*.05)
        elif args.lr!=previous['lr']:
            raise ValueError('Resume must preserve LR unless explicitly extending epochs')
        if rank==0:
            source=Path(args.resume).resolve().parent
            for name in ('best.pth','best_validation.json','best_validation_cases.json'):
                if source!=out.resolve() and (source/name).is_file() and not (out/name).exists():
                    shutil.copy2(source/name,out/name)
            if (out/'best_validation.json').exists():
                best=min(best,json.loads((out/'best_validation.json').read_text())['score'])
        if world>1:
            values=[best];dist.broadcast_object_list(values,src=0);best=values[0]
    del state
    config=dict(vars(args),backbone='point_residual34',model_config=raw.config,world_size=world,
                global_batch_size=args.batch_size*world,normalization='LayerNorm',
                residual_depths=[3,4,6,3],parameters=sum(p.numel() for p in raw.parameters()),
                train_count=len(train_ds),val_count=len(val_ds),train_sampler_padding=world*len(sampler)-len(train_ds),
                train_steps_per_epoch=min(len(train_loader),args.max_train_steps) if args.max_train_steps else len(train_loader),
                protocol=train_ds.protocol,partial_run=bool(args.limit_train or args.limit_val or args.max_train_steps or args.max_val_steps),
                image_loss='NCC + Gaussian11^3 SSIM on central128^3 of192^3 original1mm canvas',
                geometry_precision='FP32 physical transforms/grids, TF32 disabled locally; network precision unchanged',
                selection='Complete validation membership; minimum maximum of six axis MAEs; no test selection',
                independent_rank_noise=True,compute=dict(gpu=torch.cuda.get_device_name(local),torch=torch.__version__,
                    node=os.environ.get('SLURM_JOB_NODELIST'),job_id=os.environ.get('SLURM_JOB_ID')))
    if rank==0:
        atomic_json(out/'config.json',config)
    writer=SummaryWriter(str(out/'tensorboard'),purge_step=global_step if args.resume else None) if rank==0 else None
    started=time.time()
    try:
        for epoch in range(start+1,args.epochs+1):
            sampler.set_epoch(epoch);model.train();totals={};seen=0;epoch_start=time.time()
            for step,batch in enumerate(train_loader):
                moving,fixed,truth=prepare_native_pair(batch,device,affine,epoch=epoch,augment=not args.static_train,
                    seed=args.seed,fixed_mode=args.fixed_mode,curriculum_epochs=args.curriculum_epochs)
                fixed_input,sigma=augment_fixed_input(fixed,(1.,)*3,args.fixed_input_blur_mm)
                fixed_input,_=jitter_fixed_input(fixed_input,affine,args.fixed_input_rotation_noise_deg,args.fixed_input_translation_noise_mm)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=args.amp):
                    pose=model(moving,fixed_input)
                matrix=pose_matrix(pose.float());warped=warp_volume(moving,matrix,affine,affine)
                loss,parts=point_volume_registration_loss(raw,moving,fixed,affine,warped,fixed,matrix,truth,args,128,11)
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite native RegM loss')
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(raw.parameters(),args.grad_clip,error_if_nonfinite=True)
                optimizer.step();global_step+=1;seen+=len(moving)
                for key,value in parts.items():
                    totals[key]=totals.get(key,0.)+float(value)*len(moving)
                if rank==0 and global_step%args.log_interval==0:
                    for key,value in parts.items():writer.add_scalar('train/'+key,float(value),global_step)
                    writer.add_scalar('train/gradient_norm',float(norm),global_step)
                    writer.add_scalar('train/fixed_input_blur_sigma_mm',sigma,global_step)
                    progress=dict(status='running',phase='train',epoch=epoch-1,current_epoch=epoch,
                        epochs=args.epochs,step_in_epoch=step+1,steps_per_epoch=config['train_steps_per_epoch'],
                        global_step=global_step,elapsed_seconds=time.time()-started)
                    atomic_json(out/'progress.json',progress);print(json.dumps(dict(event='train_progress',**progress)),flush=True)
                if args.max_train_steps and step+1>=args.max_train_steps:
                    break
            packed=torch.tensor([*totals.values(),seen],device=device,dtype=torch.float64)
            if world>1:dist.all_reduce(packed)
            averaged={key:float(packed[i]/packed[-1]) for i,key in enumerate(totals)}
            scheduler.step();record=dict(epoch=epoch,global_step=global_step,observations=int(packed[-1]),lr=scheduler.get_last_lr()[0],
                epoch_seconds=time.time()-epoch_start,**{'train_'+key:value for key,value in averaged.items()})
            if epoch==1 or epoch%args.val_interval==0 or epoch==args.epochs:
                validation,rows=evaluate_native(raw,val_loader,device,affine,args,writer=writer,epoch=epoch)
                if not args.max_val_steps:
                    expected={Path(row['filename']).stem for row in val_ds.samples}
                    if {row['name'] for row in rows}!=expected or len(rows)!=len(val_ds):
                        raise RuntimeError('Native validation differs from the exact original membership')
                record['validation']=validation
                score=max([validation[f'translation_{a}_mm'] for a in 'xyz']+[validation[f'rotation_{a}_deg'] for a in 'xyz'])
                if score<best:
                    best=score
                    if rank==0:
                        atomic_save(out/'best.pth',dict(model=raw.state_dict(),model_config=raw.config,
                            backbone='point_residual34',epoch=epoch,validation=validation,config=config))
                        atomic_json(out/'best_validation.json',dict(epoch=epoch,score=score,metrics=validation))
                        atomic_json(out/'best_validation_cases.json',rows)
                if writer:
                    for key,value in validation.items():writer.add_scalar('val/'+key,value,epoch)
            buffer_hash=hashlib.sha256()
            for name,value in raw.named_buffers():
                buffer_hash.update(name.encode());buffer_hash.update(value.detach().cpu().numpy().tobytes())
            buffer_hashes=[buffer_hash.hexdigest()]
            if world>1:
                buffer_hashes=[None for _ in range(world)]
                dist.all_gather_object(buffer_hashes,buffer_hash.hexdigest())
                if len(set(buffer_hashes))!=1:
                    raise RuntimeError('SyncBatchNorm buffers differ between ranks')
            record['rank_buffer_sha256']=buffer_hashes
            rng=capture_rank_rng(generator);states=[rng]
            if world>1:
                states=[None for _ in range(world)];dist.all_gather_object(states,rng)
            if rank==0:
                with (out/'history.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
                atomic_save(out/'latest.pt',dict(model=raw.state_dict(),model_config=raw.config,backbone='point_residual34',
                    optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),epoch=epoch,global_step=global_step,
                    best_score=best,world_size=world,rank_rng_states=states,config=config))
                atomic_json(out/'progress.json',dict(status='completed' if epoch==args.epochs else 'running',
                    epoch=epoch,epochs=args.epochs,global_step=global_step,elapsed_seconds=time.time()-started,
                    best_validation_score=best,latest=record))
                print(json.dumps(record),flush=True);writer.flush()
            if world>1:dist.barrier()
    finally:
        if writer:writer.close()
        if world>1 and dist.is_initialized():dist.destroy_process_group()
    return best
