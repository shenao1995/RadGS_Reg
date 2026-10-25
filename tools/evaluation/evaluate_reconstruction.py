"""Evaluate a saved run using its recorded physical units, without optimizer updates."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from monai.data import list_data_collate
import gs_train
from gs_infer import build_model_from_checkpoint
from tools.data_contract import read_split_pairs
from r2_gaussian.dataset.GSDataset import GSDataset

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--data-dir', required=True)
parser.add_argument('--output-dir', required=True)
parser.add_argument('--split', choices=['val', 'test', 'train'], default='val')
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--max-val-steps', type=int, default=0)
parser.add_argument('--projection-scale', type=float, default=None, help='Optional common evaluation scale for GT and prediction, independent of training scale')
parser.add_argument('--metric-data-range', type=float, default=None)

args = parser.parse_args()
torch.set_num_threads(2)
torch.cuda.set_device(0)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
output = Path(args.output_dir)
output.mkdir(parents=True, exist_ok=True)
config_file = Path(args.checkpoint).parent / 'config.json'
config = json.loads(config_file.read_text()) if config_file.exists() else {}
model, model_config = build_model_from_checkpoint(args.checkpoint, 'cuda')
config.update(getattr(model, 'training_contract', {}))
config.update(input_scale=model.input_scale, density_unit=model.density_unit)
if args.projection_scale is not None:
    config['projection_scale'] = args.projection_scale
if args.metric_data_range is not None:
    config['metric_data_range'] = args.metric_data_range

config.update(output_dir=str(output), data_dir=args.data_dir, max_val_steps=args.max_val_steps,
              visual_volume_size=128, view_chunk=25)
config.setdefault('lambda_dssim', .2)
config.setdefault('vol_foreground_weight', 4.)
config.setdefault('metric_data_range', .1 if config.get('projection_scale', 1.) == 1 else 25.)
options = SimpleNamespace(**config)
paths, points = read_split_pairs(args.data_dir, args.split)
dataset = GSDataset(paths, points, num_points=10000)
loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, collate_fn=list_data_collate)
with SummaryWriter(str(output / 'tensorboard')) as writer:
    metrics = gs_train.validate(model, loader, 'cuda', options, writer, config.get('epochs', 0), False)
    for key, value in metrics.items():
        writer.add_scalar(f'{args.split}/{key}', value, 0)
result = {'checkpoint': args.checkpoint, 'split': args.split, 'total_cases': len(dataset),
          'max_val_steps': args.max_val_steps, 'metrics': metrics,
          'protocol': {'projection_scale': config.get('projection_scale', 1.),
                       'metric_data_range': config['metric_data_range'],
                       'input_scale': model.input_scale, 'density_unit': model.density_unit,
                       'rotation_mode': model.rotation_mode,
                       'tf32_matmul': True, 'tf32_cudnn': True,
                       'roi_views': [0,12,25,37,49], 'view_chunk': 25},
          'model_config': model_config}

gs_train.atomic_json(output / 'evaluation.json', result)
print(json.dumps(result, indent=2), flush=True)
