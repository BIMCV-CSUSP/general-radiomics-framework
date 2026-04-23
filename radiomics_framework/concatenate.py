from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from radiomics_framework.config import ProjectConfig, load_project_config
from radiomics_framework.features import ensure_directory


def setup_logger(output_path: Path) -> logging.Logger:
    """Create a logger for feature-table concatenation."""

    logger = logging.getLogger("radiomics_framework.concatenate")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ensure_directory(output_path.parent)
    file_handler = logging.FileHandler(
        output_path.parent / f"{output_path.stem}.log",
        mode="w",
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def _metadata_columns(config: ProjectConfig) -> list[str]:
    metadata = ["sample_id"]
    if config.columns.label:
        metadata.append("label")
    if config.columns.group_id:
        metadata.append("group_id")
    return metadata


def output_path_for(config: ProjectConfig, modality_name: str, roi_name: str) -> Path:
    """Return the extracted feature CSV path for one modality and ROI."""

    safe_modality = modality_name.lower().replace(" ", "_")
    safe_roi = roi_name.lower().replace(" ", "_")
    return config.output_dir / f"features_{safe_modality}_{safe_roi}.csv"


def load_feature_table(
    path: Path,
    *,
    prefix: str,
    metadata_columns: list[str],
    keep_shape: bool,
) -> pd.DataFrame:
    """Load one extracted CSV and prefix radiomics columns."""

    if not path.exists():
        raise FileNotFoundError(f"Missing extracted feature table: {path}")
    df = pd.read_csv(path)
    missing_columns = [column for column in metadata_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing metadata columns in {path}: {missing_columns}")

    optional_metadata = [column for column in ["modality", "roi"] if column in df.columns]
    diagnostics = [column for column in df.columns if column.startswith("diagnostics_")]
    feature_columns = [
        column
        for column in df.columns
        if column not in metadata_columns + optional_metadata + diagnostics
    ]
    if not keep_shape:
        feature_columns = [column for column in feature_columns if "_shape_" not in column]

    renamed_columns = {column: f"{prefix}_{column}" for column in feature_columns}
    clean_df = df[metadata_columns + feature_columns].rename(columns=renamed_columns)
    return clean_df.drop_duplicates(subset=metadata_columns, keep="first")


def build_concatenated_table(
    config: ProjectConfig,
    *,
    roi_filter: str | None = None,
    shape_reference_modality: str | None = None,
) -> pd.DataFrame:
    """Merge extracted modality/ROI CSV files into one modeling table."""

    metadata_columns = _metadata_columns(config)
    enabled_rois = [
        roi for roi in config.enabled_rois if roi_filter is None or roi.name == roi_filter
    ]
    if not enabled_rois:
        raise ValueError(f"No enabled ROI matched roi_filter={roi_filter!r}.")

    shape_reference_modality = shape_reference_modality or config.enabled_modalities[0].name
    merged_df: pd.DataFrame | None = None

    for roi in enabled_rois:
        for modality in config.enabled_modalities:
            path = output_path_for(config, modality.name, roi.name)
            prefix = f"{roi.name}_{modality.name}".lower().replace(" ", "_")
            keep_shape = modality.name == shape_reference_modality
            table = load_feature_table(
                path,
                prefix=prefix,
                metadata_columns=metadata_columns,
                keep_shape=keep_shape,
            )
            if merged_df is None:
                merged_df = table
            else:
                merged_df = merged_df.merge(
                    table,
                    on=metadata_columns,
                    how="inner",
                    validate="one_to_one",
                )

    if merged_df is None:
        raise RuntimeError("No feature tables were concatenated.")

    feature_columns = [
        column for column in merged_df.columns if column not in set(metadata_columns)
    ]
    return merged_df[metadata_columns + feature_columns]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Concatenate extracted radiomics CSV files.")
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--roi", default=None, help="Optional ROI name to concatenate. Defaults to all ROIs.")
    parser.add_argument(
        "--shape_reference_modality",
        default=None,
        help="Modality from which shape features are kept when several modalities share an ROI.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <output_dir>/concatenated/features_all.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    output_path = (
        Path(args.output).resolve()
        if args.output
        else config.output_dir / "concatenated" / "features_all.csv"
    )
    logger = setup_logger(output_path)
    logger.info("Concatenating features for project=%s", config.name)
    concatenated_df = build_concatenated_table(
        config,
        roi_filter=args.roi,
        shape_reference_modality=args.shape_reference_modality,
    )
    ensure_directory(output_path.parent)
    concatenated_df.to_csv(output_path, index=False)
    metadata_count = len(_metadata_columns(config))
    logger.info("Saved concatenated table to %s", output_path)
    logger.info(
        "Rows=%d | feature columns=%d",
        len(concatenated_df),
        len(concatenated_df.columns) - metadata_count,
    )


if __name__ == "__main__":
    main()
