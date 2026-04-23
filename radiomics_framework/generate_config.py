from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from typing import Any


IMAGE_EXTENSIONS = (
    ".nii",
    ".nii.gz",
    ".nrrd",
    ".mha",
    ".mhd",
    ".dcm",
    ".dicom",
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
)
IMAGE_KEYWORDS = (
    "image",
    "img",
    "scan",
    "volume",
    "ct",
    "mr",
    "mri",
    "pet",
    "adc",
    "dwi",
    "t1",
    "t2",
    "flair",
    "path",
)
MASK_KEYWORDS = (
    "mask",
    "seg",
    "segmentation",
    "labelmap",
    "roi",
    "contour",
)
LABEL_CANDIDATES = (
    "label",
    "target",
    "class",
    "outcome",
    "y",
    "diagnosis",
)
GROUP_CANDIDATES = (
    "patient_id",
    "subject_id",
    "case_id",
    "group_id",
    "participant_id",
    "study_id",
)
SAMPLE_CANDIDATES = (
    "sample_id",
    "study_id",
    "image_id",
    "scan_id",
    "lesion_id",
    "id",
)


def normalize_name(value: str) -> str:
    """Convert a column or project name into a stable config identifier."""

    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value.strip().lower()).strip("_")
    return normalized or "item"


def strip_path_suffix(column_name: str) -> str:
    """Remove common path suffixes from a column name."""

    value = normalize_name(column_name)
    for suffix in ("_path", "_file", "_filename", "_image", "_img"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
    return value or normalize_name(column_name)


def infer_project_name(manifest: Path) -> str:
    """Infer a readable project name from the manifest path."""

    parent_name = normalize_name(manifest.parent.name)
    if parent_name and parent_name not in {".", "data", "datasets", "examples"}:
        return parent_name
    return normalize_name(manifest.stem).replace("manifest", "radiomics_project").strip("_")


def read_manifest_preview(manifest: Path, max_rows: int = 25) -> tuple[list[str], list[dict[str, str]]]:
    """Read manifest headers and a small preview using the Python standard library."""

    with manifest.open("r", encoding="utf-8-sig", newline="") as file_handle:
        reader = csv.DictReader(file_handle)
        if not reader.fieldnames:
            raise ValueError(f"Manifest has no header row: {manifest}")
        rows = []
        for index, row in enumerate(reader):
            if index >= max_rows:
                break
            rows.append({key: (value or "") for key, value in row.items()})
    return list(reader.fieldnames), rows


def column_values(rows: list[dict[str, str]], column: str) -> list[str]:
    """Return non-empty preview values for one column."""

    return [row.get(column, "").strip() for row in rows if row.get(column, "").strip()]


def looks_like_path(values: list[str]) -> bool:
    """Return True if preview values look like file paths."""

    if not values:
        return False
    path_like = 0
    for value in values:
        lower_value = value.lower()
        has_separator = "/" in value or "\\" in value
        has_image_extension = lower_value.endswith(IMAGE_EXTENSIONS)
        if has_separator or has_image_extension:
            path_like += 1
    return path_like / max(1, len(values)) >= 0.5


def pick_first_existing(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    """Pick the first column matching common candidate names."""

    normalized_to_column = {normalize_name(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized_to_column:
            return normalized_to_column[candidate]
    return None


def infer_label_column(columns: list[str]) -> str | None:
    """Infer the target column from common names."""

    label_column = pick_first_existing(columns, LABEL_CANDIDATES)
    if label_column:
        return label_column
    return None


def infer_sample_column(columns: list[str]) -> str:
    """Infer a sample-id column or default to sample_id."""

    return pick_first_existing(columns, SAMPLE_CANDIDATES) or "sample_id"


def infer_group_column(columns: list[str]) -> str | None:
    """Infer a group-id column for leakage-safe splitting."""

    return pick_first_existing(columns, GROUP_CANDIDATES)


def infer_path_columns(
    columns: list[str],
    rows: list[dict[str, str]],
    *,
    metadata_columns: set[str],
) -> tuple[list[str], list[str]]:
    """Infer image and mask columns from names and preview values."""

    image_columns: list[str] = []
    mask_columns: list[str] = []
    for column in columns:
        if column in metadata_columns:
            continue
        normalized = normalize_name(column)
        values = column_values(rows, column)
        name_suggests_mask = any(keyword in normalized for keyword in MASK_KEYWORDS)
        name_suggests_image = any(keyword in normalized for keyword in IMAGE_KEYWORDS)
        value_suggests_path = looks_like_path(values)

        if name_suggests_mask:
            mask_columns.append(column)
        elif value_suggests_path or name_suggests_image:
            image_columns.append(column)

    return image_columns, mask_columns


def validate_columns(columns: list[str], requested_columns: list[str], kind: str) -> None:
    """Validate CLI-provided columns against manifest headers."""

    missing = [column for column in requested_columns if column not in columns]
    if missing:
        raise ValueError(f"Unknown {kind} column(s): {missing}. Available columns: {columns}")


def build_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Infer and build a project config payload from CLI arguments."""

    manifest = Path(args.manifest).resolve()
    columns, rows = read_manifest_preview(manifest)

    label_column = args.label_column or infer_label_column(columns)
    sample_column = args.sample_id_column or infer_sample_column(columns)
    group_column = args.group_id_column if args.group_id_column is not None else infer_group_column(columns)
    metadata_columns = {label_column}
    if sample_column in columns:
        metadata_columns.add(sample_column)
    if group_column:
        metadata_columns.add(group_column)

    if args.image_column:
        validate_columns(columns, args.image_column, "image")
        image_columns = args.image_column
    else:
        image_columns, inferred_masks = infer_path_columns(
            columns,
            rows,
            metadata_columns=metadata_columns,
        )
        if not args.mask_column:
            mask_columns = inferred_masks

    if args.mask_column:
        validate_columns(columns, args.mask_column, "mask")
        mask_columns = args.mask_column
    elif "mask_columns" not in locals():
        mask_columns = []

    if not image_columns:
        raise ValueError(
            "Could not infer image columns. Use --image-column one or more times."
        )

    project_root = Path(args.project_root).resolve() if args.project_root else Path.cwd().resolve()
    manifest_value = str(manifest)
    try:
        manifest_value = str(manifest.relative_to(project_root))
    except ValueError:
        pass

    params_path = args.params or "configs/pyradiomics_default.yaml"
    modalities = [
        {
            "name": strip_path_suffix(column),
            "image_column": column,
            "params": params_path,
            "enabled": True,
        }
        for column in image_columns
    ]

    rois = [
        {
            "name": strip_path_suffix(column),
            "mask_column": column,
            "label": args.mask_label,
            "enabled": True,
        }
        for column in mask_columns
    ]
    if args.include_full_roi or not rois:
        rois.append(
            {
                "name": "full",
                "mode": "full",
                "label": args.mask_label,
                "enabled": not bool(mask_columns),
            }
        )

    return {
        "project": {
            "name": args.project_name or infer_project_name(manifest),
            "root": str(project_root),
            "manifest": manifest_value,
            "output_dir": args.output_dir,
        },
        "columns": {
            "sample_id": sample_column,
            "label": label_column,
            "group_id": group_column,
        },
        "modalities": modalities,
        "rois": rois,
        "preprocessing": {
            "cast_float32": True,
            "n4_bias_correction": args.n4_bias_correction,
            "n4_shrink_factor": 4,
            "denoise": args.denoise,
            "denoise_time_step": 0.01875,
            "resample_mask_to_image": True,
        },
        "execution": {
            "n_jobs": args.n_jobs,
            "continue_on_error": True,
        },
    }


def yaml_scalar(value: Any) -> str:
    """Serialize a scalar to simple YAML."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value)
    if text == "" or text.lower() in {"true", "false", "null"} or text.startswith(("{", "[")):
        return "'" + text.replace("'", "''") + "'"
    if any(character in text for character in [":", "#", "\n", "'", "\\"]):
        return "'" + text.replace("'", "''") + "'"
    return text


def dump_yaml(payload: dict[str, Any]) -> str:
    """Write the config YAML without requiring PyYAML."""

    lines: list[str] = []
    for section_name, section_payload in payload.items():
        lines.append(f"{section_name}:")
        if isinstance(section_payload, dict):
            for key, value in section_payload.items():
                lines.extend(format_yaml_item(key, value, indent=2))
        elif isinstance(section_payload, list):
            lines.extend(format_yaml_sequence(section_payload, indent=2))
        else:
            lines.append(f"  {yaml_scalar(section_payload)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_yaml_sequence(values: list[Any], indent: int) -> list[str]:
    """Format a YAML sequence recursively."""

    lines: list[str] = []
    for item in values:
        if isinstance(item, dict):
            first_key = next(iter(item))
            first_value = item[first_key]
            lines.append(f"{' ' * indent}- {first_key}: {yaml_scalar(first_value)}")
            for child_key, child_value in list(item.items())[1:]:
                lines.extend(format_yaml_item(child_key, child_value, indent + 2))
        else:
            lines.append(f"{' ' * indent}- {yaml_scalar(item)}")
    return lines


def format_yaml_item(key: str, value: Any, indent: int) -> list[str]:
    """Format one mapping item recursively."""

    spaces = " " * indent
    if isinstance(value, dict):
        lines = [f"{spaces}{key}:"]
        for child_key, child_value in value.items():
            lines.extend(format_yaml_item(child_key, child_value, indent + 2))
        return lines
    if isinstance(value, list):
        lines = [f"{spaces}{key}:"]
        lines.extend(format_yaml_sequence(value, indent + 2))
        return lines
    return [f"{spaces}{key}: {yaml_scalar(value)}"]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Generate a radiomics project YAML from a dataset manifest."
    )
    parser.add_argument("--manifest", required=True, help="Input dataset manifest CSV.")
    parser.add_argument(
        "--output",
        default="configs/project.yaml",
        help="Output YAML path.",
    )
    parser.add_argument("--project-name", default=None, help="Optional project name.")
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root used to resolve relative paths. Defaults to the current working directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/radiomics",
        help="Extraction output directory written into the generated config.",
    )
    parser.add_argument("--sample-id-column", default=None, help="Override sample-id column.")
    parser.add_argument("--label-column", default=None, help="Override target label column.")
    parser.add_argument(
        "--group-id-column",
        default=None,
        help="Override group column. Use an empty string to disable grouping.",
    )
    parser.add_argument(
        "--image-column",
        action="append",
        help="Explicit image column. Can be passed multiple times.",
    )
    parser.add_argument(
        "--mask-column",
        action="append",
        help="Explicit mask/ROI column. Can be passed multiple times.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="PyRadiomics parameter YAML path to assign to each modality.",
    )
    parser.add_argument("--mask-label", type=int, default=1, help="Label value inside ROI masks.")
    parser.add_argument(
        "--include-full-roi",
        action="store_true",
        help="Also add a disabled whole-image ROI when masks exist, or enabled full ROI when no masks exist.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Extraction parallel jobs.")
    parser.add_argument("--n4-bias-correction", action="store_true")
    parser.add_argument("--denoise", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.group_id_column == "":
        args.group_id_column = None
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = build_config_payload(args)
    output_path.write_text(dump_yaml(payload), encoding="utf-8")
    print(f"Generated config: {output_path}")
    print(f"Modalities: {[item['name'] for item in payload['modalities']]}")
    print(f"ROIs: {[item['name'] for item in payload['rois']]}")


if __name__ == "__main__":
    main()
