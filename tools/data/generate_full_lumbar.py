"""One GPU shard of the complete, spacing-filtered CTSpine1K lumbar dataset."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time
import traceback

import numpy as np
import SimpleITK as sitk

from tools.data_generation import GenerationConfig, check_foreground_fov, extract_vertebra, load_pair, save_sample, spacing_within_limit, view_angles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--shards', type=int, default=8)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    run, raw, output = map(Path, (manifest['run_dir'], manifest['raw_dir'], manifest['output_dir']))
    output.mkdir(parents=True, exist_ok=True)
    config = GenerationConfig(**{**manifest['config'], 'labels': tuple(manifest['config']['labels'])})
    cases = manifest['cases'][args.shard::args.shards]
    state = dict(status='running', shard=args.shard, total_cases=len(cases), processed_cases=0,
                 generated_samples=0, excluded_samples=0, missing_labels=0, errors=0, started_at=time.time())
    records, exclusions, missing, errors = [], [], [], []
    destination = run/'generation'/f'shard_{args.shard}.json'

    def publish():
        state['updated_at'] = time.time()
        temporary = destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(state=state, samples=records, exclusions=exclusions,
                                             missing_labels=missing, errors=errors), indent=2))
        temporary.replace(destination)

    def prepare(case):
        ct, mask = load_pair(raw/case['ct'], raw/case['mask'])
        if not spacing_within_limit(ct.GetSpacing(), config.max_source_spacing_mm):
            raise ValueError(f"Source-spacing manifest mismatch: {case['case_id']} {ct.GetSpacing()}")
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(sitk.Cast(mask, sitk.sitkUInt32))
        samples = []
        for label in config.labels:
            if not stats.HasLabel(label):
                samples.append((label, 'missing', None))
                continue
            extracted = extract_vertebra(ct, mask, label, config.spacing_mm, config.margin_mm,
                                          stats, config.volume_size)
            if extracted is None:
                raise ValueError('A present label vanished during extraction')
            samples.append((label, 'present', extracted))
        return samples

    publish()
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = {i: pool.submit(prepare, case) for i, case in enumerate(cases[:2])}
        for index, case in enumerate(cases):
            state['current_case'] = case['case_id']
            try:
                samples = pending.pop(index).result()
            except Exception:
                samples = []
                errors.append(dict(case_id=case['case_id'], reason='prepare', traceback=traceback.format_exc()))
            next_index = index+2
            if next_index < len(cases):
                pending[next_index] = pool.submit(prepare, cases[next_index])
            for label, status, extracted in samples:
                row = dict(case_id=case['case_id'], source=case['source'], label=label,
                           level=f'L{label-19}', split=case['split'])
                if status == 'missing':
                    missing.append(row)
                    continue
                hu, foreground, geometry = extracted
                spacing = np.linalg.norm(np.asarray(geometry['voxel_to_centered_ras_mm'])[:3,:3], axis=0)
                if not spacing_within_limit(spacing, config.max_output_spacing_mm):
                    exclusions.append({**row, 'reason':'output_spacing_gt_1mm', 'spacing_mm':spacing.tolist()})
                    continue
                try:
                    check_foreground_fov(foreground, spacing, config, view_angles(config.num_views))
                except ValueError as error:
                    exclusions.append({**row, 'reason':'detector_fov', 'message':str(error)})
                    continue
                try:
                    ct_path, mask_path = raw/case['ct'], raw/case['mask']
                    provenance = {'ct_path':str(ct_path), 'mask_path':str(mask_path),
                        'ct_bytes':ct_path.stat().st_size, 'mask_bytes':mask_path.stat().st_size,
                        'ct_mtime_ns':ct_path.stat().st_mtime_ns, 'mask_mtime_ns':mask_path.stat().st_mtime_ns}
                    path = save_sample(output, case['case_id'], label, hu, foreground, geometry,
                                       config, device='cuda', backend='triton', provenance=provenance)
                    records.append({**row, 'filename':path.name, 'spacing_mm':spacing.tolist()})
                except Exception:
                    errors.append({**row, 'reason':'render_or_save', 'traceback':traceback.format_exc()})
                state.update(generated_samples=len(records), excluded_samples=len(exclusions),
                             missing_labels=len(missing), errors=len(errors))
                publish()
            state.update(processed_cases=index+1, generated_samples=len(records),
                         excluded_samples=len(exclusions), missing_labels=len(missing), errors=len(errors))
            publish()
            print(json.dumps(state), flush=True)
            del samples
    state['status'] = 'failed' if errors else 'completed'
    publish()
    if errors:
        raise RuntimeError(f'Shard {args.shard}: {len(errors)} unexpected errors; see {destination}')


if __name__ == '__main__':
    main()
