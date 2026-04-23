# General Radiomics Framework

This repository provides a reusable radiomics framework for arbitrary medical
or biomedical imaging datasets. It covers the full classical radiomics path:

1. image and ROI manifest definition
2. PyRadiomics feature extraction
3. multi-modal and multi-ROI feature concatenation
4. leakage-safe feature selection inside cross-validation folds
5. classical ML model training and comparison
6. out-of-fold aggregation, bootstrap confidence intervals, and model export

The framework is configured through YAML files and CSV manifests, so new
projects can reuse the same extraction and modeling code without changing the
Python package.

## Repository structure

```text
├── configs/
│   ├── example_project.yaml          # Template project configuration
│   └── pyradiomics_default.yaml      # Generic PyRadiomics settings
├── docs/
│   ├── manifest_schema.md            # Input CSV schema
│   └── workflow.md                   # End-to-end usage guide
├── examples/
│   └── manifest_example.csv          # Minimal manifest example
├── radiomics_framework/
│   ├── extract.py                    # Generic feature extraction
│   ├── concatenate.py                # Feature-table merging
│   ├── train.py                      # Classical ML training/evaluation
│   ├── features.py                   # Feature selection utilities
│   ├── preprocessing.py              # SimpleITK preprocessing
│   └── config.py                     # YAML config loader
└── pyproject.toml                    # Package metadata and console script
```

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a project config automatically from your manifest:

```bash
python -m radiomics_framework.generate_config \
  --manifest examples/manifest_example.csv \
  --output configs/project.yaml \
  --label-column label
```

You can also use the installed CLI:

```bash
radiomics-framework init-config \
  --manifest examples/manifest_example.csv \
  --output configs/project.yaml \
  --label-column label
```

Review the generated YAML before running extraction. If the automatic inference
does not detect the right columns, pass them explicitly with `--image-column`,
`--mask-column`, `--sample-id-column`, `--group-id-column`, and
`--label-column`.

If your manifest does not have an outcome column yet, the generated config uses
`label: null`. Extraction and concatenation still work; training requires adding
a label column later.

Extract features:

```bash
python -m radiomics_framework.extract --config configs/project.yaml
```

Concatenate extracted tables:

```bash
python -m radiomics_framework.concatenate \
  --config configs/project.yaml \
  --output artifacts/radiomics/concatenated/features_all.csv
```

Train and evaluate models:

```bash
python -m radiomics_framework.train \
  --config configs/project.yaml \
  --features artifacts/radiomics/concatenated/features_all.csv \
  --output_dir results/radiomics_framework \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --export_best_model
```

## Manifest format

The manifest is a CSV with one row per sample. The default example expects:

```csv
sample_id,patient_id,label,image_path,mask_path
case_001,patient_001,0,data/images/case_001.nii.gz,data/masks/case_001_mask.nii.gz
case_002,patient_002,1,data/images/case_002.nii.gz,data/masks/case_002_mask.nii.gz
```

For multi-modal imaging, add more image columns and define them in YAML:

```yaml
modalities:
  - name: ct
    image_column: ct_path
  - name: pet
    image_column: pet_path
```

For multiple ROIs, add more mask columns and define them in YAML:

```yaml
rois:
  - name: lesion
    mask_column: lesion_mask_path
    label: 1
  - name: organ
    mask_column: organ_mask_path
    label: 1
```

Use `mode: full` for whole-image extraction without a mask.

## Automatic YAML generation

The command below reads the manifest header and a small preview of rows, then
infers image columns, mask columns, the label column, and the grouping column:

```bash
python -m radiomics_framework.generate_config \
  --manifest /path/to/manifest.csv \
  --output configs/project.yaml
```

Common names such as `sample_id`, `patient_id`, `label`, `image_path`,
`ct_path`, `t2_path`, `mask_path`, `segmentation_path`, or `roi_path` are
detected automatically. For ambiguous datasets, use explicit overrides:

```bash
python -m radiomics_framework.generate_config \
  --manifest /path/to/manifest.csv \
  --output configs/project.yaml \
  --project-name my_project \
  --sample-id-column sample_id \
  --group-id-column patient_id \
  --label-column outcome \
  --image-column ct_path \
  --image-column pet_path \
  --mask-column tumor_mask_path \
  --include-full-roi
```

For an unlabeled multi-modal manifest such as `patient_id,study_id,T1,T2,mask`,
run:

```bash
python -m radiomics_framework.generate_config \
  --manifest /path/to/manifest.csv \
  --output configs/pituitary.yaml \
  --project-name pituitary_radiomics
```

This will infer `study_id` as `sample_id`, `patient_id` as `group_id`, `T1` and
`T2` as modalities, `mask` as ROI, and `label: null`.

## Methodological safeguards

The training pipeline includes reliability-oriented defaults:

- Cross-validation can be grouped by patient, subject, acquisition, or any
  user-defined `group_id`.
- The fold plan is created once and reused across all classifiers.
- Feature selection is performed inside each training fold only.
- Discriminant feature selection combines univariate statistics, FDR correction,
  and correlation pruning.
- Repeated out-of-fold predictions are aggregated per sample.
- Confidence intervals are estimated by stratified bootstrap at group level.

## Training outputs

The main outputs are written to `results/radiomics_framework/` by default:

```text
fold_metrics.csv
oof_predictions_flat.csv
oof_predictions_aggregated.csv
summary_metrics.csv
bootstrap_group_level_ci.csv
feature_selection/selected_features_by_fold.csv
best_model.joblib
```

`best_model.joblib` contains the fitted sklearn pipeline, the selected feature
names, and model metadata.

## Documentation

Read `docs/workflow.md` for the complete step-by-step workflow and
`docs/manifest_schema.md` for manifest design. Use
`docs/standalone_repository.md` when exporting this framework into a clean
independent repository.
