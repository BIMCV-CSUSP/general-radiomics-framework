from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import SimpleITK as sitk

from radiomics_framework.config import ProjectConfig, load_project_config, resolve_project_path
from radiomics_framework.extract import load_manifest, load_roi_mask
from radiomics_framework.features import ensure_directory
from radiomics_framework.preprocessing import preprocess_image


def setup_logger(output_dir: Path) -> logging.Logger:
    ensure_directory(output_dir)
    logger = logging.getLogger("radiomics_framework.qc")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "image_qc.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _image_stats(prefix: str, image: sitk.Image) -> dict[str, Any]:
    array = sitk.GetArrayFromImage(image).astype(float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        finite = np.asarray([np.nan])
    return {
        f"{prefix}_size": "x".join(str(item) for item in image.GetSize()),
        f"{prefix}_spacing": "x".join(f"{item:.6g}" for item in image.GetSpacing()),
        f"{prefix}_min": float(np.nanmin(finite)),
        f"{prefix}_p01": float(np.nanpercentile(finite, 1)),
        f"{prefix}_mean": float(np.nanmean(finite)),
        f"{prefix}_std": float(np.nanstd(finite)),
        f"{prefix}_p99": float(np.nanpercentile(finite, 99)),
        f"{prefix}_max": float(np.nanmax(finite)),
    }


def _mask_stats(mask: sitk.Image, label: int) -> dict[str, Any]:
    mask_array = sitk.GetArrayFromImage(mask)
    mask_binary = mask_array == label
    voxel_count = int(mask_binary.sum())
    voxel_volume_mm3 = float(np.prod(mask.GetSpacing()))
    return {
        "mask_size": "x".join(str(item) for item in mask.GetSize()),
        "mask_spacing": "x".join(f"{item:.6g}" for item in mask.GetSpacing()),
        "mask_voxels": voxel_count,
        "mask_volume_ml": voxel_count * voxel_volume_mm3 / 1000.0,
    }


def _slice_index_from_mask(mask_array: np.ndarray, label: int) -> int:
    if mask_array.ndim < 3:
        return 0
    slice_counts = (mask_array == label).sum(axis=(1, 2))
    if slice_counts.max() > 0:
        return int(np.argmax(slice_counts))
    return mask_array.shape[0] // 2


def _as_slice(array: np.ndarray, slice_index: int) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return array[slice_index]
    raise ValueError(f"QC plotting supports 2D/3D images. Found array shape={array.shape}")


def _display_limits(image_slice: np.ndarray) -> tuple[float, float]:
    finite = image_slice[np.isfinite(image_slice)]
    if finite.size == 0:
        return 0.0, 1.0
    low, high = np.nanpercentile(finite, [1, 99])
    if not np.isfinite(low) or not np.isfinite(high) or low == high:
        low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
    if low == high:
        high = low + 1.0
    return float(low), float(high)


def _write_qc_panel(
    raw_image: sitk.Image,
    processed_image: sitk.Image,
    mask: sitk.Image,
    label: int,
    output_path: Path,
    title: str,
) -> None:
    plt = _setup_matplotlib()
    raw_array = sitk.GetArrayFromImage(raw_image).astype(float)
    processed_array = sitk.GetArrayFromImage(processed_image).astype(float)
    mask_array = sitk.GetArrayFromImage(mask)
    slice_index = _slice_index_from_mask(mask_array, label)

    raw_slice = _as_slice(raw_array, min(slice_index, raw_array.shape[0] - 1) if raw_array.ndim == 3 else 0)
    processed_slice = _as_slice(processed_array, slice_index)
    mask_slice = _as_slice(mask_array, slice_index) == label
    raw_low, raw_high = _display_limits(raw_slice)
    processed_low, processed_high = _display_limits(processed_slice)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(raw_slice, cmap="gray", vmin=raw_low, vmax=raw_high)
    axes[0].set_title("Raw")
    axes[1].imshow(processed_slice, cmap="gray", vmin=processed_low, vmax=processed_high)
    axes[1].set_title("Preprocessed")
    axes[2].imshow(processed_slice, cmap="gray", vmin=processed_low, vmax=processed_high)
    axes[2].imshow(np.ma.masked_where(~mask_slice, mask_slice), cmap="autumn", alpha=0.45)
    axes[2].set_title("Mask overlay")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    ensure_directory(output_path.parent)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def export_image_qc(
    config: ProjectConfig,
    *,
    output_dir: Path | None = None,
    max_cases: int = 12,
    random_state: int = 42,
) -> Path:
    """Export raw/preprocessed/mask panels and image statistics."""

    output_dir = output_dir or config.output_dir / "qc"
    output_dir = output_dir.resolve()
    logger = setup_logger(output_dir)
    image_dir = ensure_directory(output_dir / "images")
    manifest = load_manifest(config)
    if max_cases > 0 and len(manifest) > max_cases:
        manifest = manifest.sample(n=max_cases, random_state=random_state).sort_index()

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for _, row in manifest.iterrows():
        row_payload = row.to_dict()
        sample_id = str(row[config.columns.sample_id])
        for modality in config.enabled_modalities:
            for roi in config.enabled_rois:
                try:
                    image_path = resolve_project_path(config, row[modality.image_column])
                    raw_image = sitk.ReadImage(str(image_path))
                    processed_image = preprocess_image(raw_image, config.preprocessing)
                    mask = load_roi_mask(config, pd.Series(row_payload), processed_image, roi)
                    output_path = image_dir / (
                        f"{_safe_name(sample_id)}__{_safe_name(modality.name)}__{_safe_name(roi.name)}.png"
                    )
                    _write_qc_panel(
                        raw_image,
                        processed_image,
                        mask,
                        roi.label,
                        output_path,
                        f"{sample_id} | {modality.name} | {roi.name}",
                    )
                    records.append(
                        {
                            "sample_id": sample_id,
                            "modality": modality.name,
                            "roi": roi.name,
                            "image_path": str(image_path),
                            "qc_image": str(output_path),
                            **_image_stats("raw", raw_image),
                            **_image_stats("preprocessed", processed_image),
                            **_mask_stats(mask, roi.label),
                        }
                    )
                except Exception as exc:
                    failure = {
                        "sample_id": sample_id,
                        "modality": modality.name,
                        "roi": roi.name,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    logger.error("QC failed: %s", failure)
                    if not config.execution.continue_on_error:
                        raise

    stats_path = output_dir / "image_qc_stats.csv"
    pd.DataFrame(records).to_csv(stats_path, index=False)
    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "image_qc_failures.csv", index=False)
    logger.info("Saved %d QC records to %s", len(records), stats_path)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export visual QC panels for project images and masks.")
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_cases", type=int, default=12)
    parser.add_argument("--random_state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else None
    export_image_qc(
        config,
        output_dir=output_dir,
        max_cases=args.max_cases,
        random_state=args.random_state,
    )


if __name__ == "__main__":
    main()
