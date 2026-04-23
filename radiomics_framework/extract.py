from __future__ import annotations

import argparse
import logging
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

from radiomics_framework.config import (
    ModalityConfig,
    ProjectConfig,
    RoiConfig,
    load_project_config,
    resolve_project_path,
)
from radiomics_framework.features import ensure_directory, make_sample_id
from radiomics_framework.preprocessing import (
    create_full_image_mask,
    preprocess_image,
    resample_to_reference,
)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Create a console/file logger for extraction runs."""

    ensure_directory(output_dir)
    logger = logging.getLogger("radiomics_framework.extract")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "extraction.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_manifest(config: ProjectConfig) -> pd.DataFrame:
    """Load the configured manifest and add a sample-id column if needed."""

    if not config.manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {config.manifest}")
    manifest = pd.read_csv(config.manifest)
    manifest[config.columns.sample_id] = make_sample_id(manifest, config.columns.sample_id)
    required_columns = {config.columns.sample_id}
    if config.columns.label:
        required_columns.add(config.columns.label)
    if config.columns.group_id:
        required_columns.add(config.columns.group_id)
    for modality in config.enabled_modalities:
        required_columns.add(modality.image_column)
    for roi in config.enabled_rois:
        if roi.mode != "full" and roi.mask_column:
            required_columns.add(roi.mask_column)

    missing_columns = [column for column in sorted(required_columns) if column not in manifest.columns]
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")
    return manifest


def extractor_for_modality(config: ProjectConfig, modality: ModalityConfig):
    """Build a PyRadiomics extractor for a modality."""

    try:
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover - dependency error should be explicit.
        raise RuntimeError(
            "PyRadiomics is required for extraction. Install requirements.txt first."
        ) from exc

    if modality.params:
        params_path = resolve_project_path(config, modality.params)
        return featureextractor.RadiomicsFeatureExtractor(str(params_path))
    return featureextractor.RadiomicsFeatureExtractor()


def metadata_from_row(config: ProjectConfig, row: pd.Series) -> dict[str, Any]:
    """Collect common metadata stored next to extracted features."""

    metadata = {
        "sample_id": row[config.columns.sample_id],
    }
    if config.columns.label and config.columns.label in row.index:
        metadata["label"] = row[config.columns.label]
    if config.columns.group_id and config.columns.group_id in row.index:
        metadata["group_id"] = row[config.columns.group_id]
    return metadata


def load_roi_mask(config: ProjectConfig, row: pd.Series, image: sitk.Image, roi: RoiConfig) -> sitk.Image:
    """Load or generate the ROI mask for one image."""

    if roi.mode == "full":
        return create_full_image_mask(image, label=roi.label)
    if not roi.mask_column:
        raise ValueError(f"ROI '{roi.name}' requires either mode='full' or mask_column.")
    mask_path = resolve_project_path(config, row[roi.mask_column])
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask not found for ROI '{roi.name}': {mask_path}")
    mask = sitk.ReadImage(str(mask_path))
    if config.preprocessing.resample_mask_to_image:
        mask = resample_to_reference(mask, image, is_mask=True)
    return mask


def extract_one(
    config: ProjectConfig,
    row_payload: dict[str, Any],
    modality: ModalityConfig,
    roi: RoiConfig,
) -> dict[str, Any]:
    """Extract features for one manifest row, modality, and ROI."""

    row = pd.Series(row_payload)
    image_path = resolve_project_path(config, row[modality.image_column])
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found for modality '{modality.name}': {image_path}")

    image = sitk.ReadImage(str(image_path))
    image = preprocess_image(image, config.preprocessing)
    mask = load_roi_mask(config, row, image, roi)
    extractor = extractor_for_modality(config, modality)
    features = extractor.execute(image, mask, label=roi.label)

    output = metadata_from_row(config, row)
    output.update({"modality": modality.name, "roi": roi.name})
    output.update(features)
    return output


def output_path_for(config: ProjectConfig, modality: ModalityConfig, roi: RoiConfig) -> Path:
    """Return the feature CSV path for one modality and ROI."""

    safe_modality = modality.name.lower().replace(" ", "_")
    safe_roi = roi.name.lower().replace(" ", "_")
    return config.output_dir / f"features_{safe_modality}_{safe_roi}.csv"


def run_extraction(config: ProjectConfig) -> dict[tuple[str, str], Path]:
    """Extract radiomics features for all configured modalities and ROIs."""

    logger = setup_logger(config.output_dir)
    manifest = load_manifest(config)
    logger.info("Loaded manifest: %s (%d rows)", config.manifest, len(manifest))
    logger.info(
        "Enabled modalities=%s | enabled ROIs=%s",
        [item.name for item in config.enabled_modalities],
        [item.name for item in config.enabled_rois],
    )

    results: dict[tuple[str, str], list[dict[str, Any]]] = {
        (modality.name, roi.name): []
        for modality in config.enabled_modalities
        for roi in config.enabled_rois
    }
    failures: list[dict[str, Any]] = []
    jobs = [
        (row.to_dict(), modality, roi)
        for _, row in manifest.iterrows()
        for modality in config.enabled_modalities
        for roi in config.enabled_rois
    ]

    n_jobs = max(1, config.execution.n_jobs)
    if n_jobs > 1:
        n_jobs = min(n_jobs, multiprocessing.cpu_count())

    def handle_success(modality_name: str, roi_name: str, payload: dict[str, Any]) -> None:
        results[(modality_name, roi_name)].append(payload)

    if n_jobs == 1:
        iterator = tqdm(jobs, desc="Extracting radiomics")
        for row_payload, modality, roi in iterator:
            try:
                payload = extract_one(config, row_payload, modality, roi)
                handle_success(modality.name, roi.name, payload)
            except Exception as exc:
                failure = {
                    "sample_id": row_payload.get(config.columns.sample_id),
                    "modality": modality.name,
                    "roi": roi.name,
                    "error": str(exc),
                }
                failures.append(failure)
                logger.error("Extraction failed: %s", failure)
                if not config.execution.continue_on_error:
                    raise
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            future_to_job = {
                executor.submit(extract_one, config, row_payload, modality, roi): (
                    row_payload,
                    modality,
                    roi,
                )
                for row_payload, modality, roi in jobs
            }
            for future in tqdm(as_completed(future_to_job), total=len(future_to_job), desc="Extracting radiomics"):
                row_payload, modality, roi = future_to_job[future]
                try:
                    payload = future.result()
                    handle_success(modality.name, roi.name, payload)
                except Exception as exc:
                    failure = {
                        "sample_id": row_payload.get(config.columns.sample_id),
                        "modality": modality.name,
                        "roi": roi.name,
                        "error": str(exc),
                    }
                    failures.append(failure)
                    logger.error("Extraction failed: %s", failure)
                    if not config.execution.continue_on_error:
                        raise

    written_paths: dict[tuple[str, str], Path] = {}
    for modality in config.enabled_modalities:
        for roi in config.enabled_rois:
            output_path = output_path_for(config, modality, roi)
            ensure_directory(output_path.parent)
            pd.DataFrame(results[(modality.name, roi.name)]).to_csv(output_path, index=False)
            written_paths[(modality.name, roi.name)] = output_path
            logger.info(
                "Saved %d rows for modality=%s roi=%s to %s",
                len(results[(modality.name, roi.name)]),
                modality.name,
                roi.name,
                output_path,
            )

    if failures:
        failures_path = config.output_dir / "extraction_failures.csv"
        pd.DataFrame(failures).to_csv(failures_path, index=False)
        logger.warning("Saved %d extraction failures to %s", len(failures), failures_path)

    return written_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract generic PyRadiomics features from a manifest.")
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(args.config)
    run_extraction(config)


if __name__ == "__main__":
    main()
