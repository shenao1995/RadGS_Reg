"""Merge shards, audit every payload, and write per-level/split statistics."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import pickle
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    args=parser.parse_args()
    m=json.loads(args.manifest.read_text());run=Path(m['run_dir']);out=Path(m['output_dir'])
    shards=[json.loads((run/'generation'/f'shard_{i}.json').read_text()) for i in range(8)]
    assert all(s['state']['status']=='completed' and not s['errors'] for s in shards)
    rows=sorted([r for s in shards for r in s['samples']],key=lambda r:r['filename'])
    missing=[r for s in shards for r in s['missing_labels']]
    exclusions=[r for s in shards for r in s['exclusions']]
    assert rows and len({r['filename'] for r in rows})==len(rows)
    assert len(rows)+len(missing)+len(exclusions)==5*len(m['cases'])
    assert {p.name for p in out.glob('*.pickle')}=={r['filename'] for r in rows}
    def audit(row):
        p=out/row['filename']
        meta=json.loads(p.with_suffix('.json').read_text())
        assert p.with_suffix('.ply').stat().st_size>0
        assert meta['case_id']==row['case_id'] and meta['label']==row['label']
        assert np.max(meta['geometry']['source_ct_spacing_mm'])<=1.+1e-6
        assert meta['input_view_indices']=={'ap':25,'lateral':0}
        for view in ('ap','lateral'):assert (out/'previews'/f'{p.stem}_{view}.png').is_file()
        with p.open('rb') as f:d=pickle.load(f)
        assert d['image'].shape==d['mask'].shape==(128,128,128)
        assert np.allclose(d['dVoxel'],[1.,1.,1.],rtol=0,atol=1e-6)
        assert np.isfinite(d['image']).all() and d['image'].min()>=0
        assert np.count_nonzero(d['image'][d['mask']==0])==0 and d['mask'].any()
        projs=d['train']['projections']
        assert projs.shape==(50,256,256) and np.isfinite(projs).all() and projs.min()>=0
        assert np.all(projs.max(axis=(1,2))>0)
        assert np.allclose(d['train']['angles'],np.arange(50)*np.pi/50)
        assert len(d['val']['projections'])==0
        return p.stat().st_size
    with ThreadPoolExecutor(max_workers=8) as pool: total_bytes=sum(pool.map(audit,rows))
    groups={s:{r['case_id'] for r in rows if r['split']==s} for s in ['train','val','test']}
    assert not(groups['train']&groups['val'] or groups['train']&groups['test'] or groups['val']&groups['test'])
    (out/'splits').mkdir(exist_ok=True)
    for split in groups:
        (out/'splits'/f'{split}.txt').write_text(''.join(r['filename']+'\n' for r in rows if r['split']==split))
    counts=dict(Counter(r['split'] for r in rows))
    (out/'samples_manifest.json').write_text(json.dumps({'samples':rows,'counts':counts,'snapshot':str(args.manifest)},indent=2)+'\n')
    (out/'generation_manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    stats=[]
    for level in ['L1','L2','L3','L4','L5']:
        rr=[r for r in rows if r['level']==level]
        stats.append({'level':level,'qualified_CT':len(m['cases']),
            'missing_mask':sum(r['level']==level for r in missing),
            'excluded_output_spacing':sum(r['level']==level and r['reason']=='output_spacing_gt_1mm' for r in exclusions),
            'excluded_FOV':sum(r['level']==level and r['reason']=='detector_fov' for r in exclusions),
            **{s:sum(r['split']==s for r in rr) for s in groups},'total':len(rr),'DRRs':50*len(rr)})
    with (out/'dataset_statistics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(stats[0]));w.writeheader();w.writerows(stats)
    summary={'status':'passed','raw_CT_count':m['raw_case_count'],'raw_CT_spacing_excluded':len(m['excluded_raw_cases']),
        'qualified_raw_CT':len(m['cases']),'generated_CT':len({r['case_id'] for r in rows}),
        'samples':len(rows),'DRRs':50*len(rows),'counts':counts,'split_CT_counts':{k:len(v) for k,v in groups.items()},
        'spacing_mm':[1.,1.,1.],'all_payloads_audited':len(rows),'CT_split_overlap':0,
        'missing_lumbar_labels':len(missing),'excluded_samples':exclusions,'level_statistics':stats,
        'source_CT_statistics':{src:{'downloaded':sum(c['source']==src for c in m['cases']+m['excluded_raw_cases']),
            'source_spacing_retained':sum(c['source']==src for c in m['cases'])} for src in sorted({c['source'] for c in m['cases']+m['excluded_raw_cases']})},
        'pickle_GiB':total_bytes/2**30}
    (out/'dataset_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    (run/'dataset_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
