"""Physical point-cloud extraction, sampling, and robust registration losses."""
import torch
import torch.nn.functional as F


@torch.no_grad()
def volume_surface_points(volume, affine, count=2048, threshold=.005, random=False):
    if volume.ndim!=5 or volume.shape[1]!=1:
        raise ValueError('Expected [B,1,X,Y,Z] attenuation volume')
    fields=volume.detach().float();mask=fields>threshold
    interior=-F.max_pool3d(-mask.float(),3,stride=1,padding=1)>.5
    # Explicitly treat the outer grid border as exterior.
    interior[:,:,:1]=False;interior[:,:,-1:]=False
    interior[:,:,:,:1]=False;interior[:,:,:,-1:]=False
    interior[:,:,:,:,:1]=False;interior[:,:,:,:,-1:]=False
    boundary=mask & ~interior
    shape=fields.shape[-3:];result=[]
    if affine.ndim==2:affine=affine[None].expand(len(fields),-1,-1)
    for b in range(len(fields)):
        indices=boundary[b,0].nonzero()
        if not len(indices):
            # Empty predictions remain finite without borrowing a GT mask.
            indices=torch.zeros(1,3,dtype=torch.long,device=fields.device)
        if random:
            order=torch.randperm(len(indices),device=fields.device)
        else:
            linear=(indices[:,0]*shape[1]+indices[:,1])*shape[2]+indices[:,2]
            order=torch.argsort((linear*2654435761) & 2147483647,stable=True)
        selected=indices[order[torch.arange(count,device=fields.device)%len(indices)]]
        world=selected.float()@affine[b,:3,:3].float().T+affine[b,:3,3].float()
        density=fields[b,0,selected[:,0],selected[:,1],selected[:,2]][:,None]
        normal=[]
        for axis in range(3):
            low=selected.clone();high=selected.clone()
            low[:,axis]=(low[:,axis]-1).clamp_min(0);high[:,axis]=(high[:,axis]+1).clamp_max(shape[axis]-1)
            difference=fields[b,0,high[:,0],high[:,1],high[:,2]]-fields[b,0,low[:,0],low[:,1],low[:,2]]
            spacing=affine[b,:3,axis].norm().clamp_min(1e-8)
            normal.append(-difference/(2*spacing))
        normal=F.normalize(torch.stack(normal,-1)@F.normalize(affine[b,:3,:3].float(),dim=0).T,dim=-1,eps=1e-8)
        if not mask[b].any():density.zero_();normal.zero_()
        result.append(torch.cat([world,density,normal],-1))
    return torch.stack(result)


@torch.no_grad()
def farthest_indices(xyz,count):
    if count>=xyz.shape[1]:return torch.arange(xyz.shape[1],device=xyz.device)[None].expand(len(xyz),-1)
    points=xyz.float();distance=torch.full(points.shape[:2],float('inf'),device=xyz.device)
    index=((points-points.mean(1,keepdim=True))**2).sum(-1).argmax(1)
    batch=torch.arange(len(xyz),device=xyz.device);selected=[]
    for _ in range(count):
        selected.append(index)
        distance=torch.minimum(distance,(points-points[batch,index,None]).square().sum(-1))
        index=distance.argmax(1)
    return torch.stack(selected,1)


def gather(values,index):
    return values[torch.arange(len(values),device=values.device).reshape((-1,)+(1,)*(index.ndim-1)),index]


def robust_point_registration_loss(moving,fixed,matrix,trim_fraction=.9,huber_mm=2.,distance_scale_mm=10.):
    """Trimmed symmetric nearest-point Huber loss; gradients reach rigid pose.

    No assumed surface correspondences, no scale fitting, and no gradients
    through the detached surface selection. NN lookup is detached; distances
    to selected matches retain their exact coordinate/pose derivatives.
    """
    if not 0<trim_fraction<=1:raise ValueError('trim_fraction must be in (0,1]')
    warped=moving[...,:3].float()@matrix[:,:3,:3].float().transpose(1,2)+matrix[:,None,:3,3].float()
    reference=fixed[...,:3].float();losses=[]
    with torch.autocast(device_type=warped.device.type,enabled=False):
        for source,target in [(warped,reference),(reference,warped)]:
            distances=[]
            for chunk in source.split(512,dim=1):
                with torch.no_grad():nearest=torch.cdist(chunk.detach(),target.detach()).argmin(-1)
                delta=chunk-gather(target,nearest)
                distances.append(torch.linalg.vector_norm(delta,dim=-1))
            distance=torch.cat(distances,1)
            kept=distance.topk(max(1,int(distance.shape[1]*trim_fraction)),largest=False).values
            huber=torch.where(kept<huber_mm,.5*kept.square()/huber_mm,kept-.5*huber_mm)
            losses.append(huber.mean()/distance_scale_mm)
    return sum(losses)/2


def point_volume_registration_loss(regm,moving,fixed,affine,warped,fixed_loss,predicted,truth,args,crop,window):
    from networks.pointRegM import PointRegM
    from tools.registration_losses import registration_loss
    loss,parts=registration_loss(warped,fixed_loss,predicted,truth,geo_weight=args.geo_weight,ncc_weight=getattr(args,'ncc_weight',1.),ssim_weight=getattr(args,'ssim_weight',1.),crop_size=crop,ssim_window=window)
    raw=regm.module if hasattr(regm,'module') else regm
    if isinstance(raw,PointRegM):
        clouds=[volume_surface_points(v,affine,raw.num_points,raw.surface_threshold) for v in [moving,fixed]]
        point=robust_point_registration_loss(*clouds,predicted,trim_fraction=getattr(args,'point_trim_fraction',.9))
        loss=loss+getattr(args,'point_loss_weight',1.)*point
        parts['point_huber']=point.detach()
    return loss,parts
