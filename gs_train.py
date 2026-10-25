"""DVGSNet reconstruction training, single GPU or torchrun DDP.

Group-05 architecture/physical initialization with group-12 projection-only objective.
Chamfer and training volume-shape computations/backward are commented out;
only L1 + DSSIM runs in each training step. Voxel queries remain for validation images.
All 50 views contribute each step. Renderer gradients are accumulated on
Gaussian outputs in chunks before one network/DDP backward pass.
"""
import argparse
import csv
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.utils.tensorboard import SummaryWriter
from monai.data import list_data_collate

from networks.DVGSNet import DVGSNet
# from chamfer_loss import ChamferLoss  # Disabled: projection-only experiments.
from tools.data_contract import align_gt_volume_to_query, read_split_pairs
from r2_gaussian.dataset.GSDataset import GSDataset
from r2_gaussian.gaussian import net_query, net_render
from r2_gaussian.utils.loss_utils import l1_loss, ssim


LOSS_NAMES = ("loss", "l1", "dssim")  # No computed auxiliary-loss diagnostics.


def normalize_positive_volume(volume, eps=1e-6):
    volume = torch.clamp(volume, min=0.0)
    return volume / volume.amax(dim=(-3, -2, -1), keepdim=True).clamp_min(eps)


def atomic_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def atomic_save(value, path):
    temporary = Path(str(path) + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda"])


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
            for key, value in batch.items()}


def render_chunk(gaussians, batch, begin, end):
    predictions, targets = [], []
    height, width = batch["render_imgs"].shape[-2:]
    for b in range(batch["render_imgs"].shape[0]):
        sample = {key: value[b].contiguous() for key, value in gaussians.items()}
        for view in range(begin, end):
            predictions.append(net_render(
                sample, batch["world_view_transforms"][b, view],
                batch["full_proj_transforms"][b, view], batch["camera_centers"][b, view],
                height, width, batch["FovX"][b, view], batch["FovY"][b, view])["render"])
            targets.append(batch["render_imgs"][b, view])
    return torch.stack(predictions), torch.stack(targets)


def query_volume(gaussians, batch, index, size):
    sample = {key: value[index].contiguous() for key, value in gaussians.items()}
    shape = torch.full_like(batch["nVoxel"][index].reshape(-1), size)
    prediction = net_query(sample, batch["offOrigin"][index].reshape(-1), shape,
                           batch["sVoxel"][index].reshape(-1))["vol"][None, None]
    target = align_gt_volume_to_query(batch["vol_gts"][index:index + 1],
                                     batch["gt_is_query_aligned"][index:index + 1])
    if target.shape[-3:] != prediction.shape[-3:]:
        target = F.interpolate(target, size=prediction.shape[-3:], mode="trilinear", align_corners=False)
    return prediction, target


def volume_shape_loss(prediction, target, foreground_weight):
    prediction, target = normalize_positive_volume(prediction), normalize_positive_volume(target)
    weight = 1.0 + foreground_weight * target
    return (weight * (prediction - target).abs()).sum() / weight.sum()


def physical_gaussians(outputs, args):
    """Convert network density units to physical attenuation once, for both operators."""
    unit = getattr(args, "density_unit", 1.0)
    return {**outputs, "density": outputs["density"] / unit}


def network_input(batch, args):
    return batch["input_imgs"].squeeze(2).float() * getattr(args, "input_scale", 1.0)


def projection_objective_images(prediction, target, args):
    scale = getattr(args, "projection_scale", 1.0)
    return prediction * scale, target * scale


@torch.no_grad()
def initialize_gaussian_head(model, args):
    if args.head_init == "original":
        return
    # Only changes initial values of the original head, with the same architecture.
    # Fixed seeds make all matched runs start from exactly the same parameters.
    head = model.fc_layers[-1]
    torch.nn.init.normal_(head.weight, std=1e-4)
    bias = head.bias.view(model.num_points, 11)
    bias[:, :3].uniform_(-0.7, 0.7)
    sigma = args.initial_sigma
    bias[:, 3:6] = math.log((sigma - .001) / (1.001 - sigma))
    bias[:, 6:10] = 0.
    bias[:, 6] = 1.
    density = args.initial_density * args.density_unit
    bias[:, 10] = math.log(math.expm1(density)) if density < 20 else density


def train_step(model, batch, optimizer, args):
    optimizer.zero_grad(set_to_none=True)
    outputs = physical_gaussians(model(network_input(batch, args)), args)
    leaves = {key: value.detach().requires_grad_(True) for key, value in outputs.items()}
    views = batch["render_imgs"].shape[1]
    values = torch.zeros(len(LOSS_NAMES), device=batch["input_imgs"].device, dtype=torch.float64)
    for begin in range(0, views, args.view_chunk):
        end = min(begin + args.view_chunk, views)
        prediction, target = render_chunk(leaves, batch, begin, end)
        prediction, target = projection_objective_images(prediction, target, args)
        l1 = l1_loss(prediction, target)
        dssim = 1.0 - ssim(prediction, target)
        fraction = (end - begin) / views
        ((l1 + args.lambda_dssim * dssim) * fraction).backward()
        values[1] += l1.detach() * fraction
        values[2] += dssim.detach() * fraction
        del prediction, target, l1, dssim
    # Disabled by request: no point-distance/voxelizer forward or backward in training.
    # cd = chamfer(leaves["xyz"], batch["gt_gs"])
    # (args.lambda_chamfer * cd).backward()
    # values[3] = cd.detach()
    # count = min(args.vol_loss_samples, batch["render_imgs"].shape[0])
    # for index in torch.randperm(batch["render_imgs"].shape[0], device="cpu")[:count].tolist():
    #     prediction, target = query_volume(leaves, batch, index, args.vol_loss_size)
    #     loss = volume_shape_loss(prediction, target, args.vol_foreground_weight)
    #     (args.lambda_volume * loss / count).backward()
    #     values[4] += loss.detach() / count
    #     del prediction, target, loss
    gradients = []
    for key, value in leaves.items():
        if value.grad is None:
            raise RuntimeError(f"Missing Gaussian gradient: {key}")
        gradients.append(value.grad)
    # One ordinary backward through the network invokes every DDP reducer hook.
    torch.autograd.backward(tuple(outputs.values()), tuple(gradients))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip or math.inf,
                                          error_if_nonfinite=True)
    optimizer.step()
    values[0] = values[1] + args.lambda_dssim * values[2]
    # + args.lambda_chamfer * values[3] + args.lambda_volume * values[4]
    if not torch.isfinite(values).all():
        raise FloatingPointError(f"Non-finite training losses: {values}")
    return values, float(norm)


def write_visualizations(writer, model_outputs, batch, args, epoch):
    selected = sorted({0, 25})
    gaussians = {key: value[:1] for key, value in model_outputs.items()}
    first = {key: value[:1] if torch.is_tensor(value) else value for key, value in batch.items()}
    images = []
    for view in selected:
        prediction, target = render_chunk(gaussians, first, view, view + 1)
        maximum = target.max().clamp_min(1e-6)
        images.extend([target[0] / maximum, prediction[0] / maximum,
                       (prediction[0] - target[0]).abs() / maximum])
    writer.add_images("val_images", torch.stack(images).clamp(0, 1).cpu(), epoch)
    prediction, target = query_volume(model_outputs, batch, 0, args.visual_volume_size)
    # shape_loss = volume_shape_loss(prediction, target, args.vol_foreground_weight)
    physical_prediction, physical_target = prediction[0, 0], target[0, 0]
    prediction, target = normalize_positive_volume(prediction)[0, 0], normalize_positive_volume(target)[0, 0]
    images = []
    for index in np.linspace(0, target.shape[-1] - 1, 7).round().astype(int)[1:-1]:
        images.extend([target[..., index], prediction[..., index],
                       (prediction[..., index] - target[..., index]).abs()])
    writer.add_images("reconstruction/slice-gt_pred_diff", torch.stack(images)[:, None].clamp(0, 1).cpu(), epoch)
    # writer.add_scalar("val/vol_l1", float(shape_loss), epoch)
    # Preserve unwindowed numerical output for fair comparisons; no clipping here.
    if getattr(args, "output_dir", None):
        destination = Path(args.output_dir) / "previews"
        destination.mkdir(exist_ok=True)
        np.savez_compressed(destination / f"epoch_{epoch:03d}.npz",
                            volume_prediction=physical_prediction.cpu().numpy(), volume_target=physical_target.cpu().numpy(),
                            shape_prediction=prediction.cpu().numpy(), shape_target=target.cpu().numpy())
        for view in (0, 12, 25, 37, 49):
            pred, gt = render_chunk(gaussians, first, view, view + 1)
            np.savez_compressed(destination / f"epoch_{epoch:03d}_view_{view:02d}.npz",
                                prediction=pred.cpu().numpy(), target=gt.cpu().numpy())


@torch.no_grad()
def detail_metrics(prediction, target):
    """Per-view foreground and ROI metrics, with the SAME GT-derived scale for both."""
    rows = []
    for pred, gt in zip(prediction, target):
        peak = gt.max().clamp_min(1e-8)
        mask = gt[0] > .02 * peak
        positions = mask.nonzero()
        if not len(positions):
            continue
        lo = (positions.amin(0) - 8).clamp_min(0)
        hi = positions.amax(0) + 9
        y0, x0, y1, x1 = int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])
        a = pred[None, :, y0:y1, x0:x1] / peak
        b = gt[None, :, y0:y1, x0:x1] / peak
        roi_ssim = ssim(a, b)
        foreground_rel_l1 = (pred[0][mask] - gt[0][mask]).abs().sum() / gt[0][mask].abs().sum().clamp_min(1e-8)
        gradient_error = gradient_size = dot = aa = bb = 0.
        for dim in (-1, -2):
            da, db = torch.diff(a, dim=dim), torch.diff(b, dim=dim)
            gradient_error += (da - db).abs().sum()
            gradient_size += db.abs().sum()
            dot += (da * db).sum()
            aa += da.square().sum()
            bb += db.square().sum()
        edge_cosine = dot / (aa * bb).sqrt().clamp_min(1e-8)
        rows.append(torch.stack([roi_ssim, foreground_rel_l1, gradient_error / gradient_size.clamp_min(1e-8), edge_cosine]))
    return torch.stack(rows).sum(0) if rows else prediction.new_zeros(4), len(rows)


@torch.no_grad()
def validate(model, loader, device, args, writer, epoch, distributed):
    model.eval()
    # Sums of per-view quantities plus view count; no padded validation cases.
    totals = torch.zeros(10, device=device, dtype=torch.float64)
    for step, batch in enumerate(loader):
        if args.max_val_steps and step >= args.max_val_steps:
            break
        batch = move_batch(batch, device)
        outputs = physical_gaussians(model(network_input(batch, args)), args)
        views = batch["render_imgs"].shape[1]
        for begin in range(0, views, args.view_chunk):
            prediction, target = render_chunk(outputs, batch, begin, min(begin + args.view_chunk, views))
            n = prediction.shape[0]
            # Five fixed angles per case, in physical units; evaluate detail independently
            # of the whole-frame SSIM used by the historical trainer.
            detail_indices = [b * (min(begin + args.view_chunk, views) - begin) + (view - begin)
                              for b in range(batch["render_imgs"].shape[0])
                              for view in (0, 12, 25, 37, 49) if begin <= view < min(begin + args.view_chunk, views)]
            if detail_indices:
                sums, count = detail_metrics(prediction[detail_indices], target[detail_indices])
                totals[5:9] += sums.double()
                totals[9] += count
            prediction, target = projection_objective_images(prediction, target, args)
            l1 = l1_loss(prediction, target)
            similarity = ssim(prediction, target)
            mse = (prediction - target).square().mean(dim=(1, 2, 3)).clamp_min(1e-12)
            psnr = 10.0 * torch.log10(args.metric_data_range ** 2 / mse)
            totals[:4] += torch.stack([l1, 1.0 - similarity, similarity, psnr.mean()]).double() * n
            totals[4] += n
        if writer is not None and step == 0:
            write_visualizations(writer, outputs, batch, args, epoch)
    if distributed:
        dist.all_reduce(totals)
    if totals[4] == 0:
        raise RuntimeError("Empty validation split")
    means = (totals[:4] / totals[4]).tolist()
    return {"loss": means[0] + args.lambda_dssim * means[1], "l1": means[0],
            "dssim": means[1], "ssim_2d": means[2], "psnr_2d": means[3],
            **dict(zip(("roi_ssim", "foreground_rel_l1", "edge_rel_l1", "edge_cosine"),
                       (totals[5:9] / totals[9].clamp_min(1)).tolist()))}


def training(model_dir, tensorboard_writer=None, args=None):
    args = args or parse_args([])
    world_size, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    distributed = world_size > 1
    if not torch.cuda.is_available():
        raise RuntimeError("This training path requires the Gaussian CUDA rasterizer/voxelizer.")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if distributed:
        dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)
    output = Path(model_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    train_paths, train_points = read_split_pairs(args.data_dir, "train")
    val_paths, val_points = read_split_pairs(args.data_dir, "val")
    train_ds = GSDataset(train_paths, train_points, num_points=args.num_points, load_points=False)
    val_ds = GSDataset(val_paths, val_points, num_points=args.num_points, load_points=False)
    sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed, drop_last=False)
    val_subset = Subset(val_ds, list(range(rank, len(val_ds), world_size)))
    # NetScene currently creates CUDA cameras; forked DataLoader workers are unsafe.
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=0, collate_fn=list_data_collate)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, collate_fn=list_data_collate)
    model_config = dict(in_channels=2, base_channels=32, num_points=args.num_points,
                        img_size=128, fc_rank=1024, rotation_mode=args.rotation_mode)
    raw_model = DVGSNet(**model_config).to(device)
    initialize_gaussian_head(raw_model, args)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_epochs, gamma=0.5)
    start_epoch, global_step, best_metric, best_epoch = 0, 0, -math.inf, 0
    resumed_rng = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if checkpoint.get("args", {}).get("loss_mode") != args.loss_mode:
            raise ValueError("Resume requires a checkpoint from the same projection-only training path")
        if checkpoint["world_size"] != world_size:
            raise ValueError("Resume requires the original world size for exact RNG restoration")
        raw_model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]
        best_metric, best_epoch = checkpoint["best_metric"], checkpoint["best_epoch"]
        resumed_rng = checkpoint["rng_states"][rank]
        del checkpoint
    model = DDP(raw_model, device_ids=[local_rank], output_device=local_rank,
                broadcast_buffers=False, gradient_as_bucket_view=True) if distributed else raw_model
    seed_everything(args.seed + rank)
    if resumed_rng is not None:
        restore_rng(resumed_rng)
    writer = tensorboard_writer
    if rank == 0 and writer is None:
        writer = SummaryWriter(str(output / "tensorboard"))
    # chamfer = ChamferLoss(chunk_size=1024).to(device)
    history_path = output / "metrics.csv"
    config = {**vars(args), "world_size": world_size, "global_batch_size": args.batch_size * world_size,
              "train_samples": len(train_ds), "val_samples": len(val_ds), "model_config": model_config,
              "train_steps_per_epoch": len(train_loader),
              "train_sampler_padding": len(sampler) * world_size - len(train_ds),
              "parameter_count": sum(p.numel() for p in raw_model.parameters()),
              "gpu": torch.cuda.get_device_name(device), "torch": torch.__version__}
    if rank == 0:
        atomic_json(output / "config.json", config)
        writer.add_text("args", json.dumps(config, indent=2), global_step=global_step)
        print(json.dumps(config, indent=2), flush=True)
    atomic_json(output / f"rank_{rank}.json", {"rank": rank, "local_rank": local_rank,
        "world_size": world_size, "gpu": torch.cuda.get_device_name(device), "pid": os.getpid()})
    started = time.time()
    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_started = time.time()
            sampler.set_epoch(epoch)
            model.train()
            sums = torch.zeros(len(LOSS_NAMES) + 1, device=device, dtype=torch.float64)
            for step, batch in enumerate(train_loader):
                if args.max_train_steps and step >= args.max_train_steps:
                    break
                batch = move_batch(batch, device)
                values, grad_norm = train_step(model, batch, optimizer, args)
                n = batch["input_imgs"].shape[0]
                sums[:-1] += values * n
                sums[-1] += n
                global_step += 1
                if (step + 1) % args.log_interval == 0 or step == 0:
                    current = torch.cat([values * n, values.new_tensor([n])])
                    if distributed:
                        dist.all_reduce(current)
                    metrics = dict(zip(LOSS_NAMES, (current[:-1] / current[-1]).tolist()))
                    if rank == 0:
                        for name, value in metrics.items():
                            writer.add_scalar(f"train_step/{name}", value, global_step)
                        writer.add_scalar("train_step/grad_norm_rank0", grad_norm, global_step)
                        status = {"status": "running", "epoch": epoch + 1, "epochs": args.epochs,
                                  "step": step + 1, "steps_per_epoch": len(train_loader),
                                  "global_step": global_step, "losses": metrics,
                                  "elapsed_seconds": time.time() - started,
                                  "peak_gpu_memory_GiB_rank0": torch.cuda.max_memory_allocated(device) / 2**30}
                        atomic_json(output / "progress.json", status)
                        print(f"epoch={epoch+1}/{args.epochs} step={step+1}/{len(train_loader)} "
                              + " ".join(f"{k}={v:.6f}" for k, v in metrics.items()), flush=True)
                        writer.flush()
            if distributed:
                dist.all_reduce(sums)
            train_metrics = dict(zip(LOSS_NAMES, (sums[:-1] / sums[-1]).tolist()))
            scheduler.step()
            val_metrics = {}
            if (epoch + 1) % args.val_interval == 0 or epoch + 1 == args.epochs:
                val_metrics = validate(raw_model, val_loader, device, args, writer if rank == 0 else None,
                                       epoch + 1, distributed)
            if rank == 0 and val_metrics:
                quality_path = output / "best_detail.json"
                previous = json.loads(quality_path.read_text()) if quality_path.exists() else {"roi_ssim": -math.inf}
                if val_metrics["roi_ssim"] > previous["roi_ssim"]:
                    atomic_save(raw_model.state_dict(), output / "best_detail.pth")
                    atomic_json(quality_path, {"epoch": epoch + 1, **val_metrics})
            improved = bool(val_metrics and val_metrics["ssim_2d"] > best_metric)
            if improved:
                best_metric, best_epoch = val_metrics["ssim_2d"], epoch + 1
                if rank == 0:
                    atomic_save(raw_model.state_dict(), output / "model1.pth")
            save = (epoch + 1) % args.checkpoint_interval == 0 or epoch + 1 == args.epochs
            if save:
                states = [None] * world_size
                if distributed:
                    dist.all_gather_object(states, rng_state())
                else:
                    states[0] = rng_state()
                if rank == 0:
                    atomic_save({"epoch": epoch + 1, "global_step": global_step,
                        "model_state_dict": raw_model.state_dict(), "model_config": model_config,
                        "optimizer_state_dict": optimizer.state_dict(), "scheduler_state_dict": scheduler.state_dict(),
                        "best_metric": best_metric, "best_epoch": best_epoch, "world_size": world_size,
                        "rng_states": states, "args": vars(args)}, output / "latest.pt")
            checksum_difference = 0.0
            if distributed:
                checksum = torch.stack([next(raw_model.parameters()).double().sum(),
                                        raw_model.fc_layers[-1].bias.double().sum()])
                checksums = [torch.zeros_like(checksum) for _ in range(world_size)]
                dist.all_gather(checksums, checksum)
                checksum_difference = float((torch.stack(checksums) - checksums[0]).abs().max())
                if checksum_difference > 1e-7:
                    raise RuntimeError(f"DDP parameters diverged: checksum difference {checksum_difference}")
            if rank == 0:
                for name, value in train_metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch + 1)
                for name, value in val_metrics.items():
                    writer.add_scalar(f"val/{name}", value, epoch + 1)
                writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], epoch + 1)
                row = {"epoch": epoch + 1, "global_step": global_step,
                       "seconds": time.time() - epoch_started,
                       **{f"train_{k}": v for k, v in train_metrics.items()},
                       **{f"val_{k}": val_metrics.get(k, "") for k in ("loss", "l1", "dssim", "ssim_2d", "psnr_2d", "roi_ssim", "foreground_rel_l1", "edge_rel_l1", "edge_cosine")}}
                exists = history_path.is_file()
                with history_path.open("a", newline="") as handle:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(row))
                    if not exists:
                        csv_writer.writeheader()
                    csv_writer.writerow(row)
                status = {"status": "completed" if epoch + 1 == args.epochs else "running",
                          "completed_epochs": epoch + 1, "epochs": args.epochs,
                          "global_step": global_step, "train": train_metrics, "val": val_metrics,
                          "best_ssim_2d": best_metric, "best_epoch": best_epoch,
                          "ddp_checksum_max_difference": checksum_difference,
                          "epoch_seconds": time.time() - epoch_started,
                          "elapsed_seconds": time.time() - started,
                          "peak_gpu_memory_GiB_rank0": torch.cuda.max_memory_allocated(device) / 2**30}
                atomic_json(output / "progress.json", status)
                print(json.dumps(status), flush=True)
                writer.flush()
            if distributed:
                dist.barrier()
    except BaseException:
        atomic_json(output / f"failure_rank_{rank}.json", {"status": "failed", "rank": rank,
                    "traceback": traceback.format_exc(), "global_step": global_step})
        raise
    finally:
        if rank == 0 and writer is not None:
            writer.close()
        if distributed:
            dist.destroy_process_group()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=os.environ.get("RADGS_DATA_DIR", "data/vertebra_dataset"))
    parser.add_argument("--output-dir", default=os.environ.get("RADGS_OUTPUT_DIR", "gs_model/DVGSNet_baseline"))
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch-size", type=int, default=4, help="Samples per GPU; global size = this * world size")
    parser.add_argument("--view-chunk", type=int, default=25)
    parser.add_argument("--num-points", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr-step-epochs", type=int, default=20)
    parser.add_argument("--input-scale", type=float, default=1000.)
    parser.add_argument("--projection-scale", type=float, default=1000., help="Same constant for prediction and GT before L1/SSIM; no per-view normalization")
    parser.add_argument("--density-unit", type=float, default=1000., help="Network density divided by this constant before both physical CUDA operators")
    parser.add_argument("--rotation-mode", choices=("unit", "legacy"), default="unit")
    parser.add_argument("--head-init", choices=("original", "physical"), default="physical")
    parser.add_argument("--initial-sigma", type=float, default=.035, help="Gaussian sigma in normalized scene coordinates")
    parser.add_argument("--initial-density", type=float, default=.005, help="Physical attenuation in inverse mm")
    parser.add_argument("--lambda-dssim", type=float, default=.2)
    parser.add_argument("--lambda-chamfer", type=float, default=0.)
    parser.add_argument("--lambda-volume", type=float, default=0.)
    parser.add_argument("--vol-foreground-weight", type=float, default=4.)
    parser.add_argument("--vol-loss-size", type=int, default=48)
    parser.add_argument("--vol-loss-samples", type=int, default=1)
    parser.add_argument("--visual-volume-size", type=int, default=128)
    parser.add_argument("--metric-data-range", type=float, default=25., help="Retains the historical projection PSNR range")
    parser.add_argument("--val-interval", type=int, default=2)
    parser.add_argument("--checkpoint-interval", type=int, default=5)
    parser.add_argument("--log-interval", type=int, default=25)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--max-train-steps", type=int, default=0, help="Smoke checks only; 0 = full split")
    parser.add_argument("--max-val-steps", type=int, default=0, help="Smoke checks only; 0 = full split")
    args = parser.parse_args(argv)
    if args.lambda_chamfer != 0 or args.lambda_volume != 0:
        parser.error("Auxiliary-loss code is disabled; Chamfer and volume weights must be zero.")
    args.loss_mode = "projection_only_no_auxiliary_computation"
    for key in ("epochs", "batch_size", "view_chunk", "num_points", "vol_loss_size", "vol_loss_samples",
                "visual_volume_size", "val_interval", "checkpoint_interval", "log_interval", "cpu_threads"):
        if getattr(args, key) < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.max_train_steps < 0 or args.max_val_steps < 0 or args.metric_data_range <= 0 or args.lr <= 0:
        parser.error("Require nonnegative step limits and positive learning rate/metric range")
    args.data_dir = str(Path(args.data_dir).resolve())
    args.output_dir = str(Path(args.output_dir).resolve())
    args.grad_clip = 0.0  # Norm check only; keep the original unclipped Adam update.
    for key in ("input_scale", "projection_scale", "density_unit", "initial_density", "lr_step_epochs"):
        if getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if not .001 < args.initial_sigma < 1.001:
        parser.error("initial sigma must be inside (.001, 1.001)")
    return args


if __name__ == "__main__":
    parsed = parse_args()
    training(parsed.output_dir, args=parsed)
