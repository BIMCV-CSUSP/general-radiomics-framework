"""Auto-generate PyRadiomics parameter YAMLs from a dataset fingerprint.

The goal of this module is to replace the hand-tuned
``configs/pyradiomics_default.yaml`` by one YAML per modality whose values
(``normalize``, ``resampledPixelSpacing``, ``force2D``/``force2Ddimension``,
``binWidth``, ``voxelArrayShift``, ...) are derived from the actual image
properties found in the manifest.

Workflow:

1. For each image column in the manifest, open a random subset of images
   (and masks, if any) and accumulate voxel spacing, size, and intensity
   statistics — this is the *modality fingerprint*.
2. Translate that fingerprint into PyRadiomics ``setting``/``imageType``/
   ``featureClass`` dictionaries using modality-aware heuristics
   (MR-like modalities get normalization + ``voxelArrayShift``; CT/ADC/PET
   keep the raw intensities; anisotropic volumes switch to 2D extraction
   along the correct axis).
3. Write one ``pyradiomics_<modality>.yaml`` per modality.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import SimpleITK as sitk
import yaml


LOGGER = logging.getLogger(__name__)


MR_KEYWORDS: tuple[str, ...] = (
    "t1",
    "t2",
    "flair",
    "dwi",
    "swi",
    "mprage",
    "tse",
    "mri",
    "_mr",
    "mr_",
    "pd",
)
CT_KEYWORDS: tuple[str, ...] = ("ct", "cta", "ncct")
PET_KEYWORDS: tuple[str, ...] = ("pet", "suv")
QUANTITATIVE_KEYWORDS: tuple[str, ...] = (
    "adc",
    "t1map",
    "t2map",
    "t1_map",
    "t2_map",
    "dose",
    "_map",
)


@dataclass
class GeneratedModalityParams:
    """Outputs of ``generate_params_for_modalities`` for one modality."""

    name: str
    yaml_path: Path
    fingerprint_path: Path | None
    fingerprint: "ModalityFingerprint"


@dataclass
class ModalityFingerprint:
    """Summary statistics computed from a sample of images for one modality."""

    name: str
    image_column: str
    n_sampled: int
    n_available: int
    spacing_median_xyz: tuple[float, float, float]
    spacing_min_xyz: tuple[float, float, float]
    spacing_max_xyz: tuple[float, float, float]
    size_median_xyz: tuple[int, int, int]
    is_anisotropic: bool
    slice_axis_sitk: int | None
    intensity_p01: float
    intensity_p50: float
    intensity_p99: float
    intensity_min: float
    intensity_max: float
    intensity_mean: float
    intensity_std: float
    range_p99_p01: float
    modality_kind: str  # "mr", "ct", "pet", "quantitative", "unknown"
    used_mask: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_modality_kind(name: str, image_column: str) -> str:
    """Guess the imaging modality from naming conventions."""

    text = f"{name} {image_column}".lower()
    for keyword in QUANTITATIVE_KEYWORDS:
        if keyword in text:
            return "quantitative"
    for keyword in CT_KEYWORDS:
        if f" {keyword}" in f" {text}" or text.startswith(f"{keyword}_") or text.endswith(f"_{keyword}") or text == keyword:
            return "ct"
    for keyword in PET_KEYWORDS:
        if keyword in text:
            return "pet"
    for keyword in MR_KEYWORDS:
        if keyword in text:
            return "mr"
    return "unknown"


def intensity_range_for_kind(modality_kind: str, observed_min: float, observed_max: float) -> str:
    """Return a coarse classification for reporting purposes."""

    if modality_kind == "ct":
        return "HU"
    if modality_kind == "pet":
        return "SUV"
    if modality_kind == "quantitative":
        return "quantitative"
    if modality_kind == "mr":
        return "relative"
    return "unknown"


def read_manifest_rows(manifest_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV manifest and return its columns and rows."""

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header row: {manifest_path}")
        rows = [dict(row) for row in reader]
    return list(reader.fieldnames), rows


def resolve_manifest_value(root: Path, value: str) -> Path:
    """Resolve a manifest cell against the project root."""

    path = Path(value.strip())
    if path.is_absolute():
        return path
    return (root / path).resolve()


def fingerprint_modality(
    *,
    name: str,
    image_column: str,
    rows: Sequence[dict[str, str]],
    project_root: Path,
    mask_column: str | None = None,
    max_samples: int = 20,
    intensity_sample_voxels: int = 200_000,
    random_seed: int = 42,
) -> ModalityFingerprint:
    """Sample images and compute a modality fingerprint."""

    rng = np.random.default_rng(random_seed)
    candidate_indices = [
        index
        for index, row in enumerate(rows)
        if row.get(image_column, "").strip()
    ]
    if not candidate_indices:
        raise ValueError(f"No non-empty values for image column {image_column!r}.")

    if max_samples is None or max_samples <= 0:
        sampled_indices = list(candidate_indices)
    else:
        sample_count = min(max_samples, len(candidate_indices))
        sampled_indices = rng.choice(
            candidate_indices, size=sample_count, replace=False
        ).tolist()

    spacings: list[tuple[float, float, float]] = []
    sizes: list[tuple[int, int, int]] = []
    intensity_chunks: list[np.ndarray] = []
    used_mask = False

    for index in sampled_indices:
        row = rows[index]
        image_path = resolve_manifest_value(project_root, row[image_column])
        if not image_path.exists():
            LOGGER.warning("Image not found, skipping: %s", image_path)
            continue
        try:
            image = sitk.ReadImage(str(image_path))
        except RuntimeError as exc:
            LOGGER.warning("Could not read %s: %s", image_path, exc)
            continue

        spacings.append(tuple(float(value) for value in image.GetSpacing()))
        sizes.append(tuple(int(value) for value in image.GetSize()))

        array = sitk.GetArrayFromImage(image).astype(np.float64, copy=False)

        mask_array: np.ndarray | None = None
        if mask_column:
            mask_cell = row.get(mask_column, "").strip()
            if mask_cell:
                mask_path = resolve_manifest_value(project_root, mask_cell)
                if mask_path.exists():
                    try:
                        mask_image = sitk.ReadImage(str(mask_path))
                        if mask_image.GetSize() != image.GetSize():
                            resampler = sitk.ResampleImageFilter()
                            resampler.SetReferenceImage(image)
                            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
                            mask_image = resampler.Execute(mask_image)
                        mask_array = sitk.GetArrayFromImage(mask_image)
                        used_mask = True
                    except RuntimeError as exc:
                        LOGGER.warning("Could not read mask %s: %s", mask_path, exc)
                        mask_array = None

        if mask_array is not None and np.any(mask_array > 0):
            voxels = array[mask_array > 0]
        else:
            voxels = array.ravel()

        if voxels.size == 0:
            continue
        if voxels.size > intensity_sample_voxels:
            voxels = rng.choice(voxels, size=intensity_sample_voxels, replace=False)
        intensity_chunks.append(voxels)

    if not spacings:
        raise ValueError(
            f"Could not read any image for modality {name!r} (column {image_column!r})."
        )

    spacing_array = np.asarray(spacings, dtype=np.float64)
    size_array = np.asarray(sizes, dtype=np.int64)
    intensity_pool = np.concatenate(intensity_chunks) if intensity_chunks else np.zeros(1)

    spacing_median = tuple(float(value) for value in np.median(spacing_array, axis=0))
    spacing_min = tuple(float(value) for value in np.min(spacing_array, axis=0))
    spacing_max = tuple(float(value) for value in np.max(spacing_array, axis=0))
    size_median = tuple(int(value) for value in np.median(size_array, axis=0))

    spacing_ratio = float(np.max(spacing_median) / max(float(np.min(spacing_median)), 1e-6))
    is_anisotropic = spacing_ratio >= 2.0
    slice_axis_sitk = int(np.argmax(spacing_median)) if is_anisotropic else None

    percentiles = np.percentile(intensity_pool, [1.0, 50.0, 99.0]).tolist()
    kind = detect_modality_kind(name, image_column)

    fingerprint = ModalityFingerprint(
        name=name,
        image_column=image_column,
        n_sampled=len(spacings),
        n_available=len(candidate_indices),
        spacing_median_xyz=spacing_median,
        spacing_min_xyz=spacing_min,
        spacing_max_xyz=spacing_max,
        size_median_xyz=size_median,
        is_anisotropic=is_anisotropic,
        slice_axis_sitk=slice_axis_sitk,
        intensity_p01=float(percentiles[0]),
        intensity_p50=float(percentiles[1]),
        intensity_p99=float(percentiles[2]),
        intensity_min=float(np.min(intensity_pool)),
        intensity_max=float(np.max(intensity_pool)),
        intensity_mean=float(np.mean(intensity_pool)),
        intensity_std=float(np.std(intensity_pool)),
        range_p99_p01=float(percentiles[2] - percentiles[0]),
        modality_kind=kind,
        used_mask=used_mask,
    )
    return fingerprint


def _sitk_axis_to_pyradiomics_dimension(sitk_axis: int) -> int:
    """Convert a SimpleITK spacing index to a PyRadiomics ``force2Ddimension``.

    SimpleITK's ``GetSpacing()`` returns values in ``(x, y, z)`` order, while
    ``sitk.GetArrayFromImage()`` returns ``(z, y, x)``. PyRadiomics'
    ``force2Ddimension`` uses the numpy axis index of the out-of-plane axis
    (0 = axial, 1 = coronal, 2 = sagittal).
    """

    mapping = {2: 0, 1: 1, 0: 2}
    return mapping.get(int(sitk_axis), 0)


def _round_bin_width(value: float) -> float:
    """Round a bin width to a short human-readable number."""

    if value <= 0:
        return 25.0
    if value >= 10:
        return float(round(value, 1))
    if value >= 1:
        return float(round(value, 2))
    return float(round(value, 3))


def settings_from_fingerprint(
    fingerprint: ModalityFingerprint,
    *,
    target_bin_count: int = 32,
    label: int = 1,
    geometry_tolerance: float = 1.0e-3,
    normalize_scale: float = 100.0,
) -> dict[str, Any]:
    """Translate a modality fingerprint into a PyRadiomics ``setting`` dict."""

    kind = fingerprint.modality_kind
    normalize = kind in {"mr", "unknown"}

    std = max(fingerprint.intensity_std, 1e-6)
    if normalize:
        normalized_range = (fingerprint.range_p99_p01 / std) * normalize_scale
        bin_width = _round_bin_width(normalized_range / target_bin_count)
        voxel_array_shift: int | None = int(round(3 * normalize_scale))
    else:
        effective_range = max(fingerprint.range_p99_p01, 1e-6)
        bin_width = _round_bin_width(effective_range / target_bin_count)
        voxel_array_shift = None

    spacing_xyz = list(fingerprint.spacing_median_xyz)
    if fingerprint.is_anisotropic and fingerprint.slice_axis_sitk is not None:
        force_2d = True
        force_2d_dim = _sitk_axis_to_pyradiomics_dimension(fingerprint.slice_axis_sitk)
        spacing_xyz[fingerprint.slice_axis_sitk] = 0.0
    else:
        force_2d = False
        force_2d_dim = None

    setting: dict[str, Any] = {
        "normalize": normalize,
    }
    if normalize:
        setting["normalizeScale"] = float(normalize_scale)
    setting.update(
        {
            "preCrop": True,
            "interpolator": "sitkBSpline",
            "resampledPixelSpacing": [float(round(value, 6)) for value in spacing_xyz],
            "force2D": force_2d,
        }
    )
    if force_2d_dim is not None:
        setting["force2Ddimension"] = force_2d_dim
    setting.update(
        {
            "geometryTolerance": float(geometry_tolerance),
            "correctMask": True,
            "binWidth": float(bin_width),
        }
    )
    if voxel_array_shift is not None:
        setting["voxelArrayShift"] = voxel_array_shift
    if kind == "ct":
        setting["resegmentMode"] = "absolute"
        setting["resegmentRange"] = [-1000.0, 3000.0]
    setting["label"] = int(label)
    return setting


def image_types_for_fingerprint(fingerprint: ModalityFingerprint) -> dict[str, Any]:
    """Build an ``imageType`` dictionary adapted to the modality."""

    image_types: dict[str, Any] = {
        "Original": {},
        "LoG": {"sigma": [1.0, 2.0, 3.0]},
        "Wavelet": {},
    }
    if fingerprint.modality_kind in {"mr", "unknown"}:
        image_types["Square"] = {}
        image_types["SquareRoot"] = {}
        image_types["Logarithm"] = {}
        image_types["Exponential"] = {}
    return image_types


def feature_classes_default() -> dict[str, Any]:
    """Default feature-class selection enabling all standard classes."""

    return {
        "shape": [],
        "firstorder": [],
        "glcm": [],
        "glrlm": [],
        "glszm": [],
        "ngtdm": [],
        "gldm": [],
    }


def build_params_payload(
    fingerprint: ModalityFingerprint,
    *,
    target_bin_count: int = 32,
    label: int = 1,
    geometry_tolerance: float = 1.0e-3,
    normalize_scale: float = 100.0,
) -> dict[str, Any]:
    """Return the full ``{setting, imageType, featureClass}`` payload."""

    setting = settings_from_fingerprint(
        fingerprint,
        target_bin_count=target_bin_count,
        label=label,
        geometry_tolerance=geometry_tolerance,
        normalize_scale=normalize_scale,
    )
    return {
        "setting": setting,
        "imageType": image_types_for_fingerprint(fingerprint),
        "featureClass": feature_classes_default(),
    }


def _format_header_comment(fingerprint: ModalityFingerprint) -> str:
    """Return a YAML header describing the fingerprint used to build the file."""

    lines = [
        "# PyRadiomics parameters generated by radiomics_framework.pyradiomics_params",
        f"# Modality: {fingerprint.name} (column: {fingerprint.image_column})",
        f"# Detected kind: {fingerprint.modality_kind}",
        f"# Images sampled: {fingerprint.n_sampled} of {fingerprint.n_available}",
        "# Median voxel spacing (x, y, z) mm: "
        + ", ".join(f"{value:.4f}" for value in fingerprint.spacing_median_xyz),
        "# Median image size (x, y, z): "
        + ", ".join(str(value) for value in fingerprint.size_median_xyz),
        f"# Anisotropic: {fingerprint.is_anisotropic} (slice axis sitk={fingerprint.slice_axis_sitk})",
        "# Intensity p01 / p50 / p99: "
        f"{fingerprint.intensity_p01:.2f} / {fingerprint.intensity_p50:.2f} / {fingerprint.intensity_p99:.2f}"
        f" | std={fingerprint.intensity_std:.2f}"
        f" | mask-based={fingerprint.used_mask}",
        "",
    ]
    return "\n".join(lines)


def dump_params_yaml(payload: dict[str, Any], fingerprint: ModalityFingerprint) -> str:
    """Render the PyRadiomics YAML with a fingerprint-aware header."""

    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    return _format_header_comment(fingerprint) + body


def write_params_yaml(
    payload: dict[str, Any],
    fingerprint: ModalityFingerprint,
    output_path: Path,
) -> Path:
    """Write the generated YAML to disk and return the path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dump_params_yaml(payload, fingerprint), encoding="utf-8")
    return output_path


def write_fingerprint_json(
    fingerprint: ModalityFingerprint,
    output_path: Path,
) -> Path:
    """Persist the full modality fingerprint as JSON next to the YAML."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = fingerprint.to_dict()
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    return output_path


def generate_params_for_modalities(
    *,
    manifest_path: Path,
    project_root: Path,
    modalities: Sequence[dict[str, Any]],
    output_dir: Path,
    mask_column: str | None = None,
    max_samples: int = 20,
    target_bin_count: int = 32,
    label: int = 1,
    filename_template: str = "pyradiomics_{name}.yaml",
    save_fingerprint: bool = True,
    fingerprint_filename_template: str = "pyradiomics_{name}.fingerprint.json",
) -> dict[str, GeneratedModalityParams]:
    """Fingerprint each modality and write one PyRadiomics YAML per modality.

    Parameters
    ----------
    manifest_path:
        CSV file with image paths per modality.
    project_root:
        Directory used to resolve relative paths from the manifest.
    modalities:
        Sequence of ``{"name": ..., "image_column": ...}`` dictionaries.
    output_dir:
        Directory where the generated YAMLs will be written.
    mask_column:
        Optional mask column used to restrict intensity statistics to the ROI.
    max_samples:
        Maximum number of images to open per modality.
    target_bin_count:
        Desired number of gray-level bins used to derive ``binWidth``.
    label:
        Mask label value written into the generated YAML.
    filename_template:
        Filename template for the generated files.
    """

    _, rows = read_manifest_rows(manifest_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, GeneratedModalityParams] = {}
    for modality in modalities:
        name = str(modality["name"])
        image_column = str(modality["image_column"])
        LOGGER.info("Fingerprinting modality %s (column=%s)", name, image_column)
        fingerprint = fingerprint_modality(
            name=name,
            image_column=image_column,
            rows=rows,
            project_root=project_root,
            mask_column=mask_column,
            max_samples=max_samples,
        )
        payload = build_params_payload(
            fingerprint,
            target_bin_count=target_bin_count,
            label=label,
        )
        output_path = output_dir / filename_template.format(name=name)
        write_params_yaml(payload, fingerprint, output_path)
        LOGGER.info("Wrote %s", output_path)

        fingerprint_path: Path | None = None
        if save_fingerprint:
            fingerprint_path = output_dir / fingerprint_filename_template.format(name=name)
            write_fingerprint_json(fingerprint, fingerprint_path)
            LOGGER.info("Wrote %s", fingerprint_path)

        written[name] = GeneratedModalityParams(
            name=name,
            yaml_path=output_path,
            fingerprint_path=fingerprint_path,
            fingerprint=fingerprint,
        )
    return written


def derive_preprocessing_from_fingerprints(
    fingerprints: Iterable[ModalityFingerprint],
) -> dict[str, Any]:
    """Derive a project-level preprocessing block from modality fingerprints.

    The framework applies the same preprocessing block to every modality, so
    flags that only make sense for one kind of image (N4) are enabled only
    when all modalities would benefit from them.

    - ``cast_float32`` / ``resample_mask_to_image``: always True (safe).
    - ``n4_bias_correction``: True iff every enabled modality is MR-like.
    - ``n4_shrink_factor``: 4 for typical image sizes, 2 for very small images.
    - ``denoise``: False (no reliable automatic criterion; left as opt-in).
    """

    fps = list(fingerprints)
    preprocessing: dict[str, Any] = {
        "cast_float32": True,
        "n4_bias_correction": False,
        "n4_shrink_factor": 4,
        "denoise": False,
        "denoise_time_step": 0.01875,
        "resample_mask_to_image": True,
    }
    if not fps:
        return preprocessing

    kinds = {fp.modality_kind for fp in fps}
    all_mr_like = kinds.issubset({"mr", "unknown"}) and "mr" in kinds
    preprocessing["n4_bias_correction"] = bool(all_mr_like)

    min_in_plane = min(
        min(fp.size_median_xyz[0], fp.size_median_xyz[2]) for fp in fps
    )
    preprocessing["n4_shrink_factor"] = 4 if min_in_plane >= 128 else 2
    return preprocessing


def _parse_modality_spec(spec: str) -> dict[str, str]:
    """Parse a ``name:image_column`` spec (falls back to name=column)."""

    if ":" in spec:
        name, column = spec.split(":", 1)
        return {"name": name.strip(), "image_column": column.strip()}
    return {"name": spec.strip(), "image_column": spec.strip()}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fingerprint a dataset and generate one PyRadiomics parameter YAML "
            "per modality."
        )
    )
    parser.add_argument("--manifest", required=True, help="Dataset manifest CSV path.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Directory used to resolve relative paths (defaults to the manifest parent).",
    )
    parser.add_argument(
        "--modality",
        action="append",
        default=[],
        help=(
            "Modality spec: 'name:image_column' (repeat for several). "
            "If omitted, all non-mask image columns are used as modalities "
            "with name equal to the column name."
        ),
    )
    parser.add_argument(
        "--mask-column",
        default=None,
        help="Optional mask column; when provided, intensity stats are taken inside the ROI.",
    )
    parser.add_argument(
        "--output-dir",
        default="configs",
        help="Where to write the generated YAML files.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=20,
        help="Max images to sample per modality. Pass 0 or a negative value to use every image.",
    )
    parser.add_argument(
        "--no-save-fingerprint",
        action="store_true",
        help="Do not write the full fingerprint as a JSON sidecar.",
    )
    parser.add_argument("--target-bin-count", type=int, default=32)
    parser.add_argument("--label", type=int, default=1)
    parser.add_argument(
        "--filename-template",
        default="pyradiomics_{name}.yaml",
        help="Filename template for each generated file.",
    )
    return parser.parse_args(argv)


def _infer_modalities_from_manifest(
    manifest_path: Path,
    mask_column: str | None,
) -> list[dict[str, str]]:
    """Fallback when the user does not pass --modality explicitly."""

    from radiomics_framework import generate_config as _generate_config

    columns, rows = _generate_config.read_manifest_preview(manifest_path)
    metadata_columns: set[str] = set()
    if mask_column:
        metadata_columns.add(mask_column)
    image_columns, inferred_masks = _generate_config.infer_path_columns(
        columns,
        rows,
        metadata_columns=metadata_columns,
    )
    if not image_columns:
        raise ValueError(
            "Could not infer image columns from manifest; "
            "pass --modality name:column explicitly."
        )
    if not mask_column and inferred_masks:
        # Drop inferred masks from image columns defensively.
        image_columns = [column for column in image_columns if column not in inferred_masks]
    return [
        {"name": _generate_config.strip_path_suffix(column), "image_column": column}
        for column in image_columns
    ]


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args(argv)
    manifest_path = Path(args.manifest).resolve()
    project_root = (
        Path(args.project_root).resolve() if args.project_root else manifest_path.parent
    )
    if args.modality:
        modalities = [_parse_modality_spec(spec) for spec in args.modality]
    else:
        modalities = _infer_modalities_from_manifest(manifest_path, args.mask_column)

    written = generate_params_for_modalities(
        manifest_path=manifest_path,
        project_root=project_root,
        modalities=modalities,
        output_dir=Path(args.output_dir),
        mask_column=args.mask_column,
        max_samples=args.max_samples,
        target_bin_count=args.target_bin_count,
        label=args.label,
        filename_template=args.filename_template,
        save_fingerprint=not args.no_save_fingerprint,
    )
    for name, result in written.items():
        print(f"{name}: {result.yaml_path}")


if __name__ == "__main__":
    main()
