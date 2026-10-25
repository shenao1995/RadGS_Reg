"""Held-out online joint evaluation, with a paired frozen RecM + standalone RegM baseline."""
import argparse,csv,json,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import DataLoader
from monai.data import list_data_collate
from networks.DVGSNet import DVGSNet
from networks.pointRegM import make_regm
from Reg3D_train import atomic_json,checkpoint_model
from tools.data_contract import read_split_pairs
from joint_train import load_recm,validate
from tools.registration_geometry import centered_affine
from r2_gaussian.dataset.GSDataset import GSDataset
from tools.joint_goal import GOAL_SCHEMA, gate


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--data-dir',required=True);p.add_argument('--reg-data-dir',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--baseline-recm');p.add_argument('--baseline-regm');p.add_argument('--max-steps',type=int,default=0)
    args=p.parse_args();out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    torch.cuda.set_device(0);torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    with open(args.checkpoint,'rb') as f:checkpoint_hash=hashlib.file_digest(f,'sha256').hexdigest()
    state=torch.load(args.checkpoint,map_location='cpu');config=state['config'];device='cuda'
    recm=DVGSNet(**state['recm_model_config']).to(device);recm.load_state_dict(state['recm']);recm.eval()
    regm=make_regm(state['regm_model_config']).to(device);regm.load_state_dict(state['regm']);regm.eval();epoch=state['epoch'];del state
    contract=SimpleNamespace(**config['reconstruction_contract'])
    options=SimpleNamespace(amp_reg=config.get('amp_reg',True),max_val_steps=args.max_steps,view_chunk=25,skip_projection_metrics=False,seed=config.get('seed',42),registration_loss_spacing_mm=config.get('registration_loss_spacing_mm',2.),reg_input_size=regm.input_size)
    manifest=json.loads((Path(args.reg_data_dir)/'manifest.json').read_text());lookup={Path(r['filename']).stem:r for r in manifest['samples']};lookup['_root']=args.reg_data_dir
    paths,points=read_split_pairs(args.data_dir,args.split)
    dataset=GSDataset(paths,points,load_points=False)
    loader=DataLoader(dataset,batch_size=args.batch_size,num_workers=0,collate_fn=list_data_collate)
    affine=centered_affine((regm.input_size,)*3,192./regm.input_size,device=device)
    results={}
    metrics,rows=validate(recm,regm,loader,contract,options,lookup,affine,epoch)
    results['joint']=dict(metrics=metrics,gate=gate(metrics),cases=len(rows))
    atomic_json(out/'joint_cases.json',rows)
    with (out/'joint_cases.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    del recm,regm;torch.cuda.empty_cache()
    if bool(args.baseline_recm)!=bool(args.baseline_regm):raise ValueError('Provide both baseline checkpoints')
    if args.baseline_recm:
        recm,baseline_config=load_recm(args.baseline_recm,device);regm,_=checkpoint_model(args.baseline_regm,device)
        baseline_options=SimpleNamespace(**{**vars(options),'reg_input_size':regm.input_size})
        baseline_affine=centered_affine((regm.input_size,)*3,192./regm.input_size,device=device)
        baseline_metrics,baseline_rows=validate(recm,regm,loader,SimpleNamespace(**baseline_config),baseline_options,lookup,baseline_affine,0)
        results['baseline']=dict(metrics=baseline_metrics,gate=gate(baseline_metrics),cases=len(baseline_rows))
        atomic_json(out/'baseline_cases.json',baseline_rows)
        assert [r['name'] for r in baseline_rows]==[r['name'] for r in rows]
        results['joint_minus_baseline']={key:metrics[key]-baseline_metrics[key] for key in metrics}
        # CT groups are the independent units; do not treat vertebrae from one CT as independent.
        groups={Path(r['filename']).stem:r['case_id'] for r in manifest['samples']}
        rng=np.random.default_rng(20261022);ids=sorted({groups[r['name']] for r in rows})
        for key,label in [('reconstruction_ssim3d','ssim3d_improvement_cluster_bootstrap_95ci'),
                          ('reconstruction_ssim3d_foreground_roi','foreground_ssim3d_improvement_cluster_bootstrap_95ci')]:
            diffs={g:np.array([row[key]-base[key] for row,base in zip(rows,baseline_rows) if groups[row['name']]==g]) for g in ids}
            boot=[]
            for _ in range(1000):boot.append(np.concatenate([diffs[g] for g in rng.choice(ids,len(ids),replace=True)]).mean())
            results[label]=np.percentile(boot,[2.5,97.5]).tolist()
        results['foreground_reconstruction_improved']=results['joint_minus_baseline']['reconstruction_ssim3d_foreground_roi']>0
        results['joint']['gate']=gate(metrics,baseline_metrics)
    result=dict(checkpoint=args.checkpoint,checkpoint_sha256=checkpoint_hash,epoch=epoch,split=args.split,partial_evaluation=bool(args.max_steps),results=results,
        fixed='online differentiable voxelized RecM output; never the standalone frozen cache',
        ssimmethod='true Gaussian 11^3 window sigma1.5 valid interior, fixed physical attenuation data_range=.04',
        baseline_recm=args.baseline_recm,baseline_regm=args.baseline_regm)
    result['goal_schema']=GOAL_SCHEMA
    result['diagnostic_only']=config.get('diagnostic_only',False)
    result['full_split_evaluated']=len(rows)==len(dataset)
    result['goal_metrics_passed_on_test']=bool(not result['diagnostic_only'] and args.split=='test' and len(rows)==len(dataset) and not args.max_steps and results['joint']['gate']['passed'])
    atomic_json(out/'evaluation.json',result);print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
