"""Masked vertebra CT -> NanoDRR projections and RadGS/NAF training samples.

Use the dedicated environment from requirements-nanodrr.txt. Volume axes
are XYZ in centered RAS. Scanner lengths are mm; stored projections are the
physical NanoDRR line integrals / 1000, as required by readNAFInfo's legacy
mm-to-m and scene-scale conversions. Training targets are never min-maxed.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import itertools
import json
from pathlib import Path
import pickle
import tempfile
import time

import numpy as np
import SimpleITK as sitk
import torch

SCHEMA = "radgs-nanodrr-v1"
LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0])


@dataclass(frozen=True)
class GenerationConfig:
    labels: tuple[int, ...] = (20, 21, 22, 23, 24)  # CTSpine1K L1--L5
    spacing_mm: float = 1.0
    margin_mm: float = 5.0
    num_views: int = 50
    num_val_views: int = 0
    detector_height: int = 256
    detector_width: int = 256
    pixel_size_mm: float = 1.2
    sdd_mm: float = 1100.0
    dso_mm: float = 800.0
    n_samples: int = 500
    batch_size: int = 8
    num_points: int = 10000
    seed: int = 42
    volume_size: int | None = None
    max_source_spacing_mm: float = 1.0
    max_output_spacing_mm: float = 1.0

    def __post_init__(self):
        if not self.labels or any(label <= 0 for label in self.labels):
            raise ValueError("Provide positive foreground label IDs.")
        if self.max_source_spacing_mm <= 0 or self.max_output_spacing_mm <= 0:
            raise ValueError("Spacing filter limits must be positive.")
        if self.spacing_mm <= 0 or self.margin_mm < 0:
            raise ValueError("Require positive spacing and nonnegative crop margin.")
        if self.num_views < 2 or self.num_views % 2 or self.num_val_views < 0:
            raise ValueError("num_views must be even (exact 0/90 degrees); num_val_views >= 0.")
        if min(self.detector_height, self.detector_width, self.batch_size, self.num_points) < 1:
            raise ValueError("Detector sizes, batch_size and num_points must be positive.")
        if not self.sdd_mm > self.dso_mm > 0 or self.pixel_size_mm <= 0 or self.n_samples < 2:
            raise ValueError("Require SDD > DSO > 0, positive pixel size and n_samples >= 2.")
        if self.volume_size is not None and self.volume_size < 2:
            raise ValueError("volume_size must be at least 2.")


def angle2pose(dso_mm: float, angle: float) -> np.ndarray:
    """Same c2w as dataset_readers.angle2pose; 0=LA (+RAS-X), 90=AP (+RAS-Y).

    Camera +X=image right, +Y=image down, +Z=forward. Pixel centers are
    (u+0.5, v+0.5), consistent with the rasterizer's ndc2Pix convention.
    """
    c, s = np.cos(angle), np.sin(angle)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [[-s, 0, -c], [c, 0, -s], [0, -1, 0]]
    pose[:3, 3] = [dso_mm * c, dso_mm * s, 0]
    return pose


def view_angles(count: int, *, validation: bool = False) -> np.ndarray:
    """Uniform half-open 180-degree arc; optional validation half-step grid."""
    if count == 0:
        return np.empty(0, dtype=np.float64)
    return (np.arange(count, dtype=np.float64) + (0.5 if validation else 0.0)) * np.pi / count


def spacing_within_limit(spacing, limit_mm=1.0):
    """Require all three physical spacings to be positive and <= limit (1e-6 tolerance)."""
    values = np.asarray(spacing, dtype=float)
    return bool(values.shape == (3,) and np.isfinite(values).all()
                and np.all(values > 0) and np.all(values <= limit_mm + 1e-6))


def same_grid(a: sitk.Image, b: sitk.Image) -> bool:
    return a.GetSize() == b.GetSize() and all(
        np.allclose(getattr(a, field)(), getattr(b, field)(), atol=1e-5, rtol=0)
        for field in ("GetSpacing", "GetOrigin", "GetDirection")
    )


def load_pair(ct_path: Path, mask_path: Path) -> tuple[sitk.Image, sitk.Image]:
    ct = sitk.ReadImage(str(ct_path), sitk.sitkFloat32)
    mask = sitk.ReadImage(str(mask_path))
    if ct.GetDimension() != 3 or mask.GetDimension() != 3:
        raise ValueError("CT and mask must both be 3D images.")
    if not same_grid(ct, mask):
        mask = sitk.Resample(mask, ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    return ct, mask


def extract_vertebra(ct: sitk.Image, mask: sitk.Image, label: int,
                     spacing_mm: float, margin_mm: float, stats=None, volume_size=None):
    """Crop a label and physically resample even oblique CTs onto RAS axes.

    Image background is air, mask uses nearest-neighbour interpolation, and
    original world geometry is retained alongside the centered training frame.
    """
    if stats is None:
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(sitk.Cast(mask, sitk.sitkUInt32))
    if not stats.HasLabel(label):
        return None
    box = stats.GetBoundingBox(label)
    lo = np.asarray(box[:3], dtype=int)
    hi = lo + np.asarray(box[3:], dtype=int)
    margin = np.ceil(margin_mm / np.asarray(ct.GetSpacing())).astype(int)
    lo = np.maximum(0, lo - margin)
    hi = np.minimum(ct.GetSize(), hi + margin)
    roi_ct = sitk.RegionOfInterest(ct, (hi - lo).tolist(), lo.tolist())
    roi_mask = sitk.RegionOfInterest(mask, (hi - lo).tolist(), lo.tolist())
    size = np.asarray(roi_ct.GetSize())
    corners = np.array([
        LPS_TO_RAS @ np.asarray(roi_ct.TransformContinuousIndexToPhysicalPoint(tuple(index)))
        for index in itertools.product(*[(-0.5, float(n) - 0.5) for n in size])
    ])
    lower, upper = corners.min(axis=0), corners.max(axis=0)
    center_ras = (lower + upper) / 2
    target_size = np.maximum(1, np.ceil((upper - lower) / spacing_mm).astype(int))
    if volume_size is not None:
        # A fixed cubic canvas permits ordinary batch collation. Large crops
        # increase isotropic spacing instead of clipping the anatomy.
        spacing_mm = max(spacing_mm, float((upper - lower).max()) / volume_size)
        target_size = np.full(3, volume_size, dtype=int)
    origin_ras = center_ras - (target_size - 1) * spacing_mm / 2
    reference = sitk.Image(target_size.tolist(), sitk.sitkFloat32)
    reference.SetSpacing((spacing_mm,) * 3)
    reference.SetDirection(tuple(LPS_TO_RAS.ravel()))
    reference.SetOrigin(tuple(LPS_TO_RAS @ origin_ras))
    out_ct = sitk.Resample(roi_ct, reference, sitk.Transform(), sitk.sitkLinear, -1000, sitk.sitkFloat32)
    out_mask = sitk.Resample(roi_mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, sitk.sitkUInt16)
    # SimpleITK is ZYX; NAF metadata, Gaussian coordinates and net_query are XYZ.
    foreground = sitk.GetArrayFromImage(out_mask).transpose(2, 1, 0) == label
    if not foreground.any():
        raise ValueError(f"Label {label} disappeared at {spacing_mm} mm spacing.")
    hu_xyz = sitk.GetArrayFromImage(out_ct).transpose(2, 1, 0).copy()
    hu_xyz[~foreground] = -1000.0
    if not np.isfinite(hu_xyz).all():
        raise ValueError("CT contains non-finite values.")
    centered_affine = np.eye(4)
    centered_affine[:3, :3] *= spacing_mm
    centered_affine[:3, 3] = -(target_size - 1) * spacing_mm / 2
    original_affine = centered_affine.copy()
    original_affine[:3, 3] += center_ras
    return hu_xyz, foreground, {
        "voxel_to_centered_ras_mm": centered_affine.tolist(),
        "voxel_to_original_ras_mm": original_affine.tolist(),
        "center_in_original_ras_mm": center_ras.tolist(),
        "source_ct_size_xyz": list(ct.GetSize()),
        "source_ct_spacing_mm": list(ct.GetSpacing()),
        "source_ct_origin_lps_mm": list(ct.GetOrigin()),
        "source_ct_direction_lps": list(ct.GetDirection()),
        "crop_index_xyz": lo.tolist(), "crop_size_xyz": (hi - lo).tolist(),
        "touches_scan_boundary": bool(np.any(lo == 0) or np.any(hi == ct.GetSize())),
    }


def make_subject(hu_xyz: np.ndarray, foreground: np.ndarray, affine: np.ndarray, device: str):
    from nanodrr.data import Subject
    from nanodrr.data.preprocess import hu_to_mu
    from torchio import ScalarImage

    hu = torch.from_numpy(np.ascontiguousarray(hu_xyz, dtype=np.float32))
    mu_xyz = hu_to_mu(hu) * torch.from_numpy(foreground.astype(np.float32))
    image = ScalarImage(tensor=mu_xyz.unsqueeze(0), affine=affine)
    subject = Subject.from_images(image, convert_to_mu=False).to(device)
    return subject, mu_xyz.numpy()


def render_projections(subject, angles: np.ndarray, config: GenerationConfig,
                       device: str, backend: str) -> tuple[np.ndarray, float]:
    from nanodrr.camera import make_k_inv
    from nanodrr.drr import DRR

    if len(angles) == 0:
        return np.empty((0, config.detector_height, config.detector_width), np.float32), 0.0
    k_inv = make_k_inv(config.sdd_mm, config.pixel_size_mm, config.pixel_size_mm,
                       0.0, 0.0, config.detector_height, config.detector_width,
                       device=torch.device(device), dtype=torch.float32)
    renderer = DRR(k_inv, torch.tensor([config.sdd_mm], dtype=torch.float32, device=device),
                   config.detector_height, config.detector_width)
    projections = []
    started = time.perf_counter()
    with torch.inference_mode():
        for begin in range(0, len(angles), config.batch_size):
            poses = np.stack([angle2pose(config.dso_mm, angle)
                              for angle in angles[begin:begin + config.batch_size]])
            images = renderer(subject, torch.as_tensor(poses, dtype=torch.float32, device=device),
                              n_samples=config.n_samples, backend=backend)
            projections.append(images[:, 0].cpu().numpy())
    result = np.concatenate(projections).astype(np.float32, copy=False)
    if not np.isfinite(result).all() or result.min() < -1e-6 or result.max() <= 0:
        raise RuntimeError("Invalid/empty DRRs: check CT, labels, camera geometry and attenuation.")
    return result, time.perf_counter() - started


def point_cloud(foreground: np.ndarray, spacing_mm: float, count: int, seed: int) -> np.ndarray:
    indices = np.argwhere(foreground)
    if len(indices) == 0:
        raise ValueError("Cannot sample an empty foreground.")
    if len(indices) > count:
        indices = indices[np.random.default_rng(seed).choice(len(indices), count, replace=False)]
    extent = np.asarray(foreground.shape) * spacing_mm
    return ((indices + 0.5) * spacing_mm - extent / 2) * (2.0 / extent.max())


def _atomic_write(path: Path, writer):
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        writer(temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def check_foreground_fov(foreground, spacing, config, angles):
    """Check tissue support, excluding the air padding used for fixed-size batches."""
    indices = np.argwhere(foreground)
    if not len(indices):
        raise ValueError("Empty foreground.")
    half_size = np.asarray(foreground.shape) / 2
    lower = (indices.min(axis=0) - half_size) * spacing
    upper = (indices.max(axis=0) + 1 - half_size) * spacing
    corners = np.array(list(itertools.product(*zip(lower, upper))))
    for angle in angles:
        w2c = np.linalg.inv(angle2pose(config.dso_mm, angle))
        camera = corners @ w2c[:3, :3].T + w2c[:3, 3]
        sensor = camera[:, :2] / camera[:, 2:3] * config.sdd_mm
        half_detector = np.array([config.detector_width, config.detector_height]) * config.pixel_size_mm / 2
        if np.any(np.abs(sensor) > half_detector):
            raise ValueError("Vertebra exceeds detector FOV: increase detector/pixel size.")


def save_sample(output: Path, case_id: str, label: int, hu: np.ndarray, foreground: np.ndarray,
                geometry: dict, config: GenerationConfig, *, device: str, backend: str,
                overwrite: bool = False, preview: bool = True, provenance: dict | None = None):
    from importlib.metadata import version
    from plyfile import PlyData, PlyElement

    output.mkdir(parents=True, exist_ok=True)
    stem = f"{case_id}_L{label - 19}" if 20 <= label <= 25 else f"{case_id}_label{label}"
    paths = [output / (stem + ext) for ext in (".pickle", ".ply", ".json")]
    signature = hashlib.sha256(json.dumps({"schema": SCHEMA, "config": asdict(config),
        "label": label, "source": provenance or {}, "geometry": geometry}, sort_keys=True).encode()).hexdigest()
    if any(path.exists() for path in paths) and not overwrite:
        if all(path.exists() for path in paths) and json.loads(paths[2].read_text()).get("signature") == signature:
            print(f"SKIP {stem} (same configuration)", flush=True)
            return paths[0]
        raise FileExistsError(f"{output / stem}: partial/different output exists; choose another directory or --overwrite.")
    effective_spacing = np.linalg.norm(np.asarray(geometry["voxel_to_centered_ras_mm"])[:3, :3], axis=0)
    extent = np.asarray(hu.shape) * effective_spacing
    if np.linalg.norm(extent / 2) >= config.dso_mm:
        raise ValueError("Source must lie outside the volume; increase --dso-mm.")
    train_angles = view_angles(config.num_views)
    val_angles = view_angles(config.num_val_views, validation=True)
    if len(val_angles) and np.any(np.isclose(val_angles[:, None], train_angles[None, :], atol=1e-10, rtol=0)):
        raise ValueError("Validation grid overlaps train; use --num-val-views equal to --num-views.")
    check_foreground_fov(foreground, effective_spacing, config, np.concatenate([train_angles, val_angles]))
    subject, mu_xyz = make_subject(hu, foreground, np.asarray(geometry["voxel_to_centered_ras_mm"]), device)
    train, train_seconds = render_projections(subject, train_angles, config, device, backend)
    val, val_seconds = render_projections(subject, val_angles, config, device, backend)
    seed = config.seed + int(hashlib.sha256(stem.encode()).hexdigest()[:8], 16)
    points = point_cloud(foreground, float(effective_spacing[0]), config.num_points, seed).astype(np.float32)
    metadata = {
        "schema": SCHEMA, "signature": signature, "case_id": case_id,
        "label": label, "provenance": provenance or {}, "config": asdict(config), "geometry": geometry,
        "volume_axes": "XYZ", "volume_frame": "centered_RAS", "density_unit": "1/mm",
        "projection_storage_scale": 0.001, "projection_quantity": "line_integral_mu_mm / 1000",
        "point_frame": "readNAFInfo normalized scene", "scene_mm_to_normalized": 2.0 / extent.max(),
        "input_view_indices": {"ap": config.num_views // 2, "lateral": 0},
        "input_view_order": ["ap", "lateral"], "angle_unit": "radian",
        "arc": "[0, pi), endpoint excluded", "renderer": "nanodrr", "backend": backend,
        "nanodrr_version": version("nanodrr"), "torch_version": torch.__version__,
        "train_render_seconds": train_seconds, "val_render_seconds": val_seconds,
        "intensity_normalization": "none; PNG previews only",
    }
    data = {
        "numTrain": len(train), "numVal": len(val), "DSD": config.sdd_mm, "DSO": config.dso_mm,
        "nDetector": [config.detector_height, config.detector_width],
        "dDetector": [config.pixel_size_mm] * 2, "nVoxel": list(mu_xyz.shape),
        "dVoxel": effective_spacing.tolist(), "offOrigin": [0.0] * 3, "offDetector": [0.0] * 2,
        "accuracy": 0.5, "mode": "cone", "filter": None, "totalAngle": 180.0,
        "startAngle": 0.0, "randomAngle": False, "normalize": False, "noise": False,
        "image": np.ascontiguousarray(mu_xyz), "mask": foreground.astype(np.uint8),
        "train": {"angles": train_angles, "projections": train / 1000.0},
        "val": {"angles": val_angles, "projections": val / 1000.0}, "metadata": metadata,
    }
    def write_pickle(path):
        with path.open("wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    _atomic_write(paths[0], write_pickle)
    vertices = np.empty(len(points), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    _atomic_write(paths[1], lambda path: PlyData([PlyElement.describe(vertices, "vertex")]).write(str(path)))
    if preview:
        from PIL import Image
        preview_dir = output / "previews"
        preview_dir.mkdir(exist_ok=True)
        for name, index in metadata["input_view_indices"].items():
            img = np.round(np.clip(train[index] / float(train.max()), 0, 1) * 255).astype(np.uint8)
            Image.fromarray(img).save(preview_dir / f"{stem}_{name}.png")
    _atomic_write(paths[2], lambda path: path.write_text(json.dumps(metadata, indent=2)))
    print(f"OK {stem}: volume={tuple(mu_xyz.shape)}, DRRs={train.shape}, "
          f"{train_seconds:.2f}s, backend={backend}", flush=True)
    return paths[0]


def _stem(path: Path) -> str:
    return path.name[:-7] if path.name.endswith(".nii.gz") else path.stem


def discover_pairs(ct_dir: Path, mask_dir: Path):
    for ct in sorted([*ct_dir.rglob("*.nii.gz"), *ct_dir.rglob("*.nii")]):
        relative = ct.relative_to(ct_dir)
        parent = mask_dir / relative.parent
        candidates = [parent / f"{_stem(ct)}_seg.nii.gz", parent / relative.name,
                      parent / f"{_stem(ct)}_seg.nii"]
        mask = next((path for path in candidates if path.is_file()), None)
        if mask is None:
            raise FileNotFoundError(f"No matching mask for {ct}")
        yield ct, mask, "__".join([*relative.parts[:-1], _stem(ct)])


def generate_net_data(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, help="Single CT NIfTI (requires --mask).")
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--ct-dir", type=Path, default=Path("data/raw_data/volumes"))
    parser.add_argument("--mask-dir", type=Path, default=Path("data/raw_data/labels"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/vertebra_dataset"))
    parser.add_argument("--labels", type=int, nargs="+", default=[20, 21, 22, 23, 24])
    for name, default, kind in [
        ("spacing-mm", 1.0, float), ("max-source-spacing-mm", 1.0, float),
        ("max-output-spacing-mm", 1.0, float), ("margin-mm", 5.0, float), ("num-views", 50, int),
        ("num-val-views", 0, int), ("detector-height", 256, int), ("detector-width", 256, int),
        ("pixel-size-mm", 1.2, float), ("sdd-mm", 1100.0, float), ("dso-mm", 800.0, float),
        ("n-samples", 500, int), ("batch-size", 8, int), ("num-points", 10000, int), ("seed", 42, int),
    ]:
        parser.add_argument("--" + name, default=default, type=kind)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--backend", choices=["auto", "torch", "triton"], default="auto")
    parser.add_argument("--max-cases", type=int, help="Maximum number of CTs, not vertebrae.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--volume-size", type=int, help="Fixed cubic output size; enlarges spacing if needed to avoid clipping.")
    args = parser.parse_args(argv)
    if bool(args.ct) != bool(args.mask):
        parser.error("Use --ct and --mask together.")
    if args.max_cases is not None and args.max_cases < 1:
        parser.error("--max-cases must be positive.")
    config = GenerationConfig(**{field: tuple(args.labels) if field == "labels" else getattr(args, field)
                                  for field in GenerationConfig.__dataclass_fields__})
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    is_cuda = torch.device(device).type == "cuda"
    if is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use an allocated GPU job or --device cpu.")
    backend = ("triton" if is_cuda else "torch") if args.backend == "auto" else args.backend
    if backend == "triton" and not is_cuda:
        raise ValueError("The Triton backend requires CUDA.")
    if args.ct:
        pairs = [(args.ct, args.mask, _stem(args.ct))]
    else:
        if not args.ct_dir.is_dir() or not args.mask_dir.is_dir():
            parser.error("CT/mask directories do not exist; set --ct-dir and --mask-dir.")
        pairs = discover_pairs(args.ct_dir, args.mask_dir)
    if args.max_cases is not None:
        pairs = itertools.islice(pairs, args.max_cases)
    generated = 0
    for ct_path, mask_path, case_id in pairs:
        print(f"CASE {case_id}: {ct_path}", flush=True)
        ct, mask = load_pair(ct_path, mask_path)
        if not spacing_within_limit(ct.GetSpacing(), config.max_source_spacing_mm):
            print(f"EXCLUDE {case_id}: source spacing={ct.GetSpacing()}", flush=True)
            continue
        stats = sitk.LabelShapeStatisticsImageFilter()
        stats.Execute(sitk.Cast(mask, sitk.sitkUInt32))
        for label in config.labels:
            extracted = extract_vertebra(ct, mask, label, config.spacing_mm, config.margin_mm, stats, config.volume_size)
            if extracted is None:
                print(f"  label {label}: absent, skipped", flush=True)
                continue
            hu, foreground, geometry = extracted
            effective = np.linalg.norm(np.asarray(geometry["voxel_to_centered_ras_mm"])[:3, :3], axis=0)
            if not spacing_within_limit(effective, config.max_output_spacing_mm):
                print(f"EXCLUDE {case_id} label={label}: output spacing={effective}", flush=True)
                continue
            provenance = {"ct_path": str(ct_path.resolve()), "mask_path": str(mask_path.resolve()),
                "ct_bytes": ct_path.stat().st_size, "mask_bytes": mask_path.stat().st_size,
                "ct_mtime_ns": ct_path.stat().st_mtime_ns, "mask_mtime_ns": mask_path.stat().st_mtime_ns}
            save_sample(args.out_dir, case_id, label, hu, foreground, geometry, config,
                        device=device, backend=backend, overwrite=args.overwrite,
                        preview=not args.no_preview, provenance=provenance)
            generated += 1
    if not generated:
        raise RuntimeError("No vertebrae generated: check paths and segmentation label IDs.")
    print(f"Finished: {generated} vertebral samples -> {args.out_dir.resolve()}", flush=True)


if __name__ == "__main__":
    generate_net_data()
