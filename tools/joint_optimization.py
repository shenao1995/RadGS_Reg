"""Stopped-target reconstruction and alternating rigid registration updates."""
from contextlib import contextmanager
import math

import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from tools.registration_losses import central_crop, normalized_cross_correlation, ssim3d


@contextmanager
def teacher_mode(model):
    """Prediction uses fixed running statistics; restore mixed BN/train flags."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.no_grad():
            yield
    finally:
        for module, training in modes:
            module.training = training


def foreground_boxes(target, margin_voxels, minimum_size):
    """Detached aligned-CT foreground bounds, with margin and valid SSIM windows."""
    boxes=[]
    shape=target.shape[-3:]
    if any(size<minimum_size for size in shape):raise ValueError('SSIM window exceeds the image field')
    for sample in target.detach():
        coordinates=(sample>0).any(dim=0).nonzero()
        if not len(coordinates):
            boxes.append(None);continue
        low=coordinates.min(0).values.tolist();high=(coordinates.max(0).values+1).tolist()
        limits=[]
        for first,last,size in zip(low,high,shape):
            first=max(0,first-margin_voxels);last=min(size,last+margin_voxels)
            if last-first<minimum_size:
                first=max(0,min((first+last-minimum_size)//2,size-minimum_size));last=first+minimum_size
            limits.append(slice(first,last))
        boxes.append(tuple(limits))
    return boxes


def image_target_loss(volume, target, crop_size, window, foreground_weight=0.,
                      roi_margin_mm=5., spacing_mm=1.):
    if target.requires_grad:
        raise ValueError('RecM target must be detached from the pose estimator')
    if volume.shape!=target.shape or volume.ndim!=5:
        raise ValueError('Target image loss requires equal [B,C,X,Y,Z] fields')
    if not math.isfinite(foreground_weight) or not 0<=foreground_weight<=1:
        raise ValueError('Foreground weight must lie in [0,1]')
    if not math.isfinite(roi_margin_mm) or roi_margin_mm<0 or not math.isfinite(spacing_mm) or spacing_mm<=0:
        raise ValueError('Invalid foreground ROI physical geometry')
    prediction, reference = central_crop(volume, crop_size), central_crop(target, crop_size)
    def image_parts(prediction,reference):
        return (1-normalized_cross_correlation(prediction,reference)).mean(),(1-ssim3d(prediction,reference,window=window)).mean()
    if foreground_weight==0:
        ncc,structural=image_parts(prediction,reference)
        return ncc+structural,dict(ncc=ncc.detach(),dssim=structural.detach())
    boxes=foreground_boxes(reference,math.ceil(roi_margin_mm/spacing_mm),window)
    # An empty sample uses the whole field in the per-case ROI loop below.
    full_parts=image_parts(prediction,reference) if foreground_weight<1 else None
    roi_parts=[];fractions=[]
    for i,box in enumerate(boxes):
        slices=(slice(i,i+1),slice(None),*(box or (slice(None),)*3))
        roi_parts.append(image_parts(prediction[slices],reference[slices]))
        fractions.append(math.prod(prediction[slices].shape[-3:])/math.prod(prediction.shape[-3:]))
    roi_ncc=torch.stack([p[0] for p in roi_parts]).mean()
    roi_structural=torch.stack([p[1] for p in roi_parts]).mean()
    if foreground_weight==1:
        ncc,structural=roi_ncc,roi_structural
    else:
        ncc=(1-foreground_weight)*full_parts[0]+foreground_weight*roi_ncc
        structural=(1-foreground_weight)*full_parts[1]+foreground_weight*roi_structural
    parts=dict(ncc=ncc.detach(),dssim=structural.detach(),roi_ncc=roi_ncc.detach(),
               roi_dssim=roi_structural.detach(),roi_field_fraction=prediction.new_tensor(sum(fractions)/len(fractions)))
    if full_parts is not None:parts.update(full_ncc=full_parts[0].detach(),full_dssim=full_parts[1].detach())
    return ncc+structural,parts


def stopped_ct_target(model, moving, fixed, moving_loss, affine, affine_loss,
                      truth, crop_size, args, backend):
    """Align moving CT with a detached prediction from the point RegM."""
    if getattr(args, 'recm_target_pose', 'predicted') != 'predicted':
        raise ValueError('RecM supervision requires the predicted RegM pose')
    with teacher_mode(model):
        matrix, warped = backend.reg_forward(model, moving.detach(), fixed.detach(), affine, args)
        return backend.warp_volume(moving_loss, matrix, affine_loss, affine_loss) if crop_size == 128 else warped


def parameter_evidence(model, head):
    checked = {}
    for name, parameter in [('encoder', next(model.encoder.parameters())), ('head', head)]:
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or not bool(parameter.grad.abs().sum()>0):
            raise RuntimeError('Missing/nonfinite staged gradient: '+name)
        checked[name] = float(parameter.grad.norm())
    return checked


def staged_joint_step(recm, regm, batch, optimizer, contract, args, lookup, affine, epoch, gradient_audit=False):
    # Imported here to reuse the exact dataset, physical operators and rendering contract.
    import joint_train as backend
    raw_rec = recm.module if isinstance(recm, DDP) else recm
    raw_reg = regm.module if isinstance(regm, DDP) else regm
    alternating = args.optimization_mode == 'alternating'
    optimizer.zero_grad(set_to_none=True)
    outputs = backend.physical_gaussians(recm(backend.network_input(batch, contract)), contract)
    leaves = {key: value.detach().requires_grad_(True) for key, value in outputs.items()}
    rec_loss = batch['input_imgs'].new_zeros(()); l1_sum=rec_loss.clone(); dssim_sum=rec_loss.clone()
    if not gradient_audit:
        views = batch['render_imgs'].shape[1]
        for begin in range(0, views, args.view_chunk):
            end=min(begin+args.view_chunk,views)
            prediction,target=backend.render_chunk(leaves,batch,begin,end)
            prediction=prediction*contract.projection_scale; target=target*contract.projection_scale
            l1=(prediction-target).abs().mean(); dssim=1-backend.ssim(prediction,target)
            fraction=(end-begin)/views
            ((l1+contract.lambda_dssim*dssim)*fraction).backward()
            rec_loss+=(l1.detach()+contract.lambda_dssim*dssim.detach())*fraction
            l1_sum+=l1.detach()*fraction; dssim_sum+=dssim.detach()*fraction
    reconstructed, canonical = backend.query_batch(leaves,batch)
    _,truth,_ = backend.pair_metadata(batch,lookup,reconstructed.device,epoch,True,args.seed)
    moving,fixed,moving_loss,fixed_loss,affine_loss,crop,window = backend.registration_pair(reconstructed,canonical,truth,affine,args)
    # Teacher inputs remain uncorrupted; fixed-input uncertainty belongs only
    # to the trainable RegM step. Use the current online RecM output.
    target=stopped_ct_target(raw_reg,moving,fixed,moving_loss,affine,affine_loss,truth,crop,args,backend)
    image_loss,parts=image_target_loss(fixed_loss,target,crop,window,
        foreground_weight=getattr(args,'recm_target_foreground_weight',0.),
        roi_margin_mm=getattr(args,'recm_target_roi_margin_mm',5.),
        spacing_mm=args.registration_loss_spacing_mm)
    if not torch.isfinite(image_loss):raise FloatingPointError('Nonfinite stopped-target image loss')
    (args.lambda_reg*image_loss).backward()
    if any(p.grad is not None for p in raw_reg.parameters()):
        raise RuntimeError('RecM step leaked gradients into RegM')
    gradients=[]
    for key,value in leaves.items():
        if value.grad is None or not torch.isfinite(value.grad).all():raise RuntimeError('Invalid Gaussian gradient: '+key)
        gradients.append(value.grad)
    torch.autograd.backward(tuple(outputs.values()),tuple(gradients))
    rec_evidence = parameter_evidence(raw_rec,raw_rec.fc_layers[-1].weight) if gradient_audit else None
    norm_rec=torch.nn.utils.clip_grad_norm_(raw_rec.parameters(),args.grad_clip,error_if_nonfinite=True)
    if not gradient_audit:optimizer.step()  # RegM grads are None: Adam does not update it.
    result=dict(loss=float(rec_loss+args.lambda_reg*image_loss.detach()),reconstruction_loss=float(rec_loss),
                projection_l1=float(l1_sum),projection_dssim=float(dssim_sum),
                registration_loss=float(image_loss),reg_ncc=float(parts['ncc']),reg_dssim=float(parts['dssim']),
                reg_geodesic=0.,recm_gradient_norm=float(norm_rec),regm_gradient_norm=0.,
                target_requires_grad=target.requires_grad)
    result.update({'target_'+key:float(value) for key,value in parts.items() if key not in ('ncc','dssim')})
    del outputs,leaves,gradients,reconstructed,fixed,fixed_loss,moving,moving_loss,target,image_loss
    optimizer.zero_grad(set_to_none=True)
    reg_evidence=None
    if alternating:
        # Use the UPDATED RecM on the same batch; raw/no_grad avoids an extra DDP
        # forward that would otherwise expect a RecM backward in the RegM step.
        with torch.no_grad():
            attributes=backend.physical_gaussians(raw_rec(backend.network_input(batch,contract)),contract)
            reconstructed,canonical=backend.query_batch(attributes,batch)
            moving,fixed,moving_loss,fixed_loss,affine_loss,crop,window=backend.registration_pair(reconstructed,canonical,truth,affine,args)
        matrix,warped=backend.reg_forward(regm,moving,fixed,affine,args)
        warped_loss=backend.warp_volume(moving_loss,matrix,affine_loss,affine_loss) if crop==128 else warped
        from tools.point_registration import point_volume_registration_loss
        reg_loss,reg_parts=point_volume_registration_loss(regm,moving,fixed,affine,warped_loss,fixed_loss,matrix,truth,args,crop,window)
        if not torch.isfinite(reg_loss):raise FloatingPointError('Nonfinite alternating RegM loss')
        (args.lambda_reg*reg_loss).backward()
        if any(p.grad is not None for p in raw_rec.parameters()):
            raise RuntimeError('RegM step leaked gradients into RecM')
        reg_evidence=parameter_evidence(raw_reg,raw_reg.pose_head.layers[-1].weight) if gradient_audit else None
        norm_reg=torch.nn.utils.clip_grad_norm_(raw_reg.parameters(),args.grad_clip,error_if_nonfinite=True)
        if not gradient_audit:optimizer.step()
        result.update(regm_gradient_norm=float(norm_reg),regm_step_loss=float(reg_loss),
                      regm_step_ncc=float(reg_parts['ncc']),regm_step_dssim=float(reg_parts['dssim']),
                      regm_step_geodesic=float(reg_parts['geodesic']))
        if 'point_huber' in reg_parts:result['regm_point_huber']=float(reg_parts['point_huber'])
    if gradient_audit:
        optimizer.zero_grad(set_to_none=True)
        return dict(status='passed',mode=args.optimization_mode,projection_loss_disabled=True,
                    target_pose_source=getattr(args,'recm_target_pose','predicted'),
                    image_loss_parts={key:float(value) for key,value in parts.items()},
                    recm_gradient=rec_evidence,regm_gradient=reg_evidence,
                    target_requires_grad=False,recm_step_regm_gradients_absent=True,
                    regm_step_recm_gradients_absent=True if alternating else None,
                    optimizer_steps=0)
    return result


def registration_warmup_step(recm,regm,batch,optimizer,contract,args,lookup,affine,epoch):
    """Train the new pose encoder before exposing RecM to its stopped targets."""
    import joint_train as backend
    from tools.point_registration import point_volume_registration_loss
    raw=recm.module if isinstance(recm,DDP) else recm
    optimizer.zero_grad(set_to_none=True)
    with teacher_mode(raw):
        attributes=backend.physical_gaussians(raw(backend.network_input(batch,contract)),contract)
        reconstructed,canonical=backend.query_batch(attributes,batch)
        _,truth,_=backend.pair_metadata(batch,lookup,reconstructed.device,epoch,True,args.seed)
        moving,fixed,moving_loss,fixed_loss,affine_loss,crop,window=backend.registration_pair(reconstructed,canonical,truth,affine,args)
    matrix,warped=backend.reg_forward(regm,moving,fixed,affine,args)
    warped_loss=backend.warp_volume(moving_loss,matrix,affine_loss,affine_loss) if crop==128 else warped
    loss,parts=point_volume_registration_loss(regm,moving,fixed,affine,warped_loss,fixed_loss,matrix,truth,args,crop,window)
    (args.lambda_reg*loss).backward()
    if any(p.grad is not None for p in raw.parameters()):raise RuntimeError('Warmup changed RecM gradients')
    norm=torch.nn.utils.clip_grad_norm_(regm.parameters(),args.grad_clip,error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss),regm_step_loss=float(loss),regm_step_ncc=float(parts['ncc']),
                regm_step_dssim=float(parts['dssim']),regm_step_geodesic=float(parts['geodesic']),
                regm_point_huber=float(parts.get('point_huber',loss.new_zeros(()))),
                regm_gradient_norm=float(norm),recm_gradient_norm=0.,regm_warmup=1.)
