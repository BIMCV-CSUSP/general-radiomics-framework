from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ColumnsConfig:
    """Column names used by the project manifest and modeling table."""

    sample_id: str = "sample_id"
    label: str | None = "label"
    group_id: str | None = "patient_id"


@dataclass(frozen=True)
class ModalityConfig:
    """One image input stream to process with PyRadiomics."""

    name: str
    image_column: str
    params: str | None = None
    enabled: bool = True


@dataclass(frozen=True)
class RoiConfig:
    """One region-of-interest definition.

    Use ``mask_column`` for ROI extraction or ``mode: full`` for whole-image
    extraction with a generated all-ones mask.
    """

    name: str
    mask_column: str | None = None
    mode: str = "mask"
    label: int = 1
    enabled: bool = True


@dataclass(frozen=True)
class PreprocessingConfig:
    """Image preprocessing options applied before feature extraction."""

    cast_float32: bool = True
    n4_bias_correction: bool = False
    n4_shrink_factor: int = 4
    denoise: bool = False
    denoise_time_step: float = 0.01875
    resample_mask_to_image: bool = True


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime controls for extraction and training."""

    n_jobs: int = 1
    continue_on_error: bool = True


@dataclass(frozen=True)
class ProjectConfig:
    """Validated project configuration loaded from YAML."""

    name: str
    root: Path
    manifest: Path
    output_dir: Path
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    modalities: tuple[ModalityConfig, ...] = field(default_factory=tuple)
    rois: tuple[RoiConfig, ...] = field(default_factory=tuple)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)

    @property
    def enabled_modalities(self) -> tuple[ModalityConfig, ...]:
        return tuple(modality for modality in self.modalities if modality.enabled)

    @property
    def enabled_rois(self) -> tuple[RoiConfig, ...]:
        return tuple(roi for roi in self.rois if roi.enabled)


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    """Resolve a config path relative to the config file location."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        payload = yaml.safe_load(file_handle) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file must contain a mapping: {path}")
    return payload


def load_project_config(config_path: str | Path) -> ProjectConfig:
    """Load and validate a project YAML configuration."""

    config_path = Path(config_path).resolve()
    raw = _load_yaml(config_path)
    base_dir = config_path.parent

    project_raw = raw.get("project", {})
    if not isinstance(project_raw, dict):
        raise ValueError("The 'project' section must be a mapping.")

    root = _resolve_path(project_raw.get("root", "."), base_dir)
    manifest = _resolve_path(project_raw.get("manifest", "data/manifest.csv"), root)
    output_dir = _resolve_path(project_raw.get("output_dir", "artifacts/radiomics"), root)

    columns_raw = raw.get("columns", {})
    columns = ColumnsConfig(
        sample_id=columns_raw.get("sample_id", "sample_id"),
        label=columns_raw.get("label", "label"),
        group_id=columns_raw.get("group_id", "patient_id"),
    )

    modalities = tuple(
        ModalityConfig(
            name=str(item["name"]).strip(),
            image_column=str(item["image_column"]).strip(),
            params=item.get("params"),
            enabled=bool(item.get("enabled", True)),
        )
        for item in raw.get("modalities", [])
    )
    if not modalities:
        raise ValueError("At least one modality must be defined in 'modalities'.")

    rois = tuple(
        RoiConfig(
            name=str(item["name"]).strip(),
            mask_column=item.get("mask_column"),
            mode=str(item.get("mode", "mask")).strip().lower(),
            label=int(item.get("label", 1)),
            enabled=bool(item.get("enabled", True)),
        )
        for item in raw.get("rois", [{"name": "full", "mode": "full"}])
    )
    if not rois:
        raise ValueError("At least one ROI must be defined in 'rois'.")

    preprocessing_raw = raw.get("preprocessing", {})
    preprocessing = PreprocessingConfig(
        cast_float32=bool(preprocessing_raw.get("cast_float32", True)),
        n4_bias_correction=bool(preprocessing_raw.get("n4_bias_correction", False)),
        n4_shrink_factor=int(preprocessing_raw.get("n4_shrink_factor", 4)),
        denoise=bool(preprocessing_raw.get("denoise", False)),
        denoise_time_step=float(preprocessing_raw.get("denoise_time_step", 0.01875)),
        resample_mask_to_image=bool(preprocessing_raw.get("resample_mask_to_image", True)),
    )

    execution_raw = raw.get("execution", {})
    execution = ExecutionConfig(
        n_jobs=int(execution_raw.get("n_jobs", 1)),
        continue_on_error=bool(execution_raw.get("continue_on_error", True)),
    )

    return ProjectConfig(
        name=str(project_raw.get("name", "radiomics_project")),
        root=root,
        manifest=manifest,
        output_dir=output_dir,
        columns=columns,
        modalities=modalities,
        rois=rois,
        preprocessing=preprocessing,
        execution=execution,
    )


def resolve_project_path(config: ProjectConfig, value: str | Path) -> Path:
    """Resolve user data paths relative to the configured project root."""

    path = Path(value)
    if path.is_absolute():
        return path
    return (config.root / path).resolve()
