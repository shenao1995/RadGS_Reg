"""Shared orientation and view conventions for legacy and NanoDRR datasets."""
from pathlib import Path

import torch


def input_view_indices(metadata, num_views):
    indices = metadata.get("input_view_indices", {"ap": 24, "lateral": 0})
    ap, lateral = int(indices["ap"]), int(indices["lateral"])
    if not (0 <= ap < num_views and 0 <= lateral < num_views) or ap == lateral:
        raise ValueError(f"Invalid AP/LA indices {indices} for {num_views} views")
    return ap, lateral


def volume_is_query_aligned(metadata):
    return (metadata.get("schema") == "radgs-nanodrr-v1"
            and metadata.get("volume_axes") == "XYZ"
            and metadata.get("volume_frame") == "centered_RAS")


def align_gt_volume_to_query(volume, query_aligned=False):
    """Keep NanoDRR XYZ volumes; preserve the historical flip for legacy data."""
    flags = torch.as_tensor(query_aligned, dtype=torch.bool, device=volume.device)
    if flags.numel() == 1:
        return volume if bool(flags.item()) else torch.rot90(volume, -2, dims=(-3, -2))
    if flags.ndim != 1 or flags.numel() != volume.shape[0]:
        raise ValueError("Alignment flags must be scalar or match the batch dimension.")
    return torch.where(flags.reshape((-1,) + (1,) * (volume.ndim - 1)), volume,
                       torch.rot90(volume, -2, dims=(-3, -2)))


def read_split_pairs(data_dir, split):
    root = Path(data_dir)
    listing = root / "splits" / f"{split}.txt"
    paths = [str(root / line.strip()) for line in listing.read_text().splitlines() if line.strip()]
    points = [str(Path(path).with_suffix(".ply")) for path in paths]
    if not paths:
        raise ValueError(f"No generated samples in {listing}")
    for path in paths + points:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    return paths, points
