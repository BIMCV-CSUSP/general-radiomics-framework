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
  --export_best_model \
  --explain_best_model
```

Create visual image QC panels before/after preprocessing:

```bash
python -m radiomics_framework.qc \
  --config configs/project.yaml \
  --max_cases 24
```

or with the installed CLI:

```bash
radiomics-framework qc-images \
  --config configs/project.yaml \
  --max_cases 24
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

## Modality-aware PyRadiomics params

The framework can build a PyRadiomics parameter YAML per modality from a
dataset *fingerprint* instead of reusing the generic
`configs/pyradiomics_default.yaml`. Each modality is sampled, its voxel
spacing, image size and intensity distribution are summarised, and the
following settings are derived automatically:

- `normalize` / `normalizeScale` / `voxelArrayShift` — enabled for MR-like
  modalities (T1, T2, FLAIR, DWI, ...); disabled for CT, PET and
  quantitative maps (ADC, T1/T2 maps).
- `resampledPixelSpacing` — median spacing across the sample. For
  anisotropic volumes the slice axis is set to `0` so PyRadiomics resamples
  only in-plane.
- `force2D` / `force2Ddimension` — enabled when the slice spacing is at
  least 2x larger than the in-plane spacing, using the axis with the
  largest spacing.
- `binWidth` — derived from the observed `p99 - p01` intensity range and
  the target number of gray-level bins (default 32). For normalized MR
  images the bin width is computed after applying the normalization gain
  so that the number of bins stays consistent across scanners.
- `resegmentRange` — set to `[-1000, 3000]` HU for CT.
- `imageType` — `Original`, `LoG` and `Wavelet` are always enabled;
  `Square`, `SquareRoot`, `Logarithm` and `Exponential` are added for
  MR-like modalities where monotonic intensity transforms are meaningful.

Generate the YAMLs standalone:

```bash
python -m radiomics_framework.pyradiomics_params \
  --manifest examples/dataset_igtpT1T2.csv \
  --modality t1:T1 \
  --modality t2:T2 \
  --mask-column mask \
  --output-dir configs
```

Or, more commonly, have `init-config` generate them alongside the project
YAML and wire them into each modality automatically:

```bash
python -m radiomics_framework.generate_config \
  --manifest examples/dataset_igtpT1T2.csv \
  --output configs/pituitary.yaml \
  --project-name pituitary_radiomics \
  --auto-params
```

The generated files are named `pyradiomics_<modality>.yaml` and carry a
header comment with the fingerprint used to derive their values
(sample size, median spacing, intensity percentiles, detected kind).

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
plots/evaluation/roc_curves.png
plots/evaluation/precision_recall_curves.png
plots/evaluation/calibration_curves.png
plots/evaluation/model_comparison_oof_metrics.png
plots/evaluation/fold_metric_distributions.png
plots/evaluation/confusion_matrices/*.png
threshold_metrics.csv
decision_curve.csv
calibration_summary.csv
plots/feature_distributions/*.png
feature_distribution_summary.csv
selected_feature_spearman_correlation.csv
plots/feature_correlation/selected_feature_correlation_heatmap.png
feature_importance/model_native_feature_importance.csv
feature_importance/model_native_feature_importance.png
feature_importance/model_native_group_importance.csv
feature_importance/permutation_importance_auc.csv
feature_importance/permutation_importance_auc.png
feature_importance/permutation_group_importance_auc.csv
feature_selection/selected_features_by_fold.csv
feature_selection/feature_selection_stability.csv
feature_selection/feature_selection_stability.png
feature_selection/feature_selection_group_stability.csv
best_model.joblib
```

`best_model.joblib` contains the fitted sklearn pipeline, the selected feature
names, and model metadata.

When `--explain_best_model` is enabled, the training command also writes:

```text
interpretability/shap_values_class1.csv
interpretability/shap_explained_feature_values.csv
interpretability/shap_base_values.csv
interpretability/shap_feature_importance.csv
interpretability/shap_bar_class1.png
interpretability/shap_beeswarm_class1.png
interpretability/lime_local_explanations_class1.csv
interpretability/lime_aggregate_importance_class1.csv
interpretability/lime_aggregate_importance_class1.png
```

The SHAP and LIME values explain the final exported best model's class-1
probability using the selected feature subset. Use `--shap_max_samples`,
`--shap_background_samples`, `--shap_max_display`, `--lime_max_samples`, and
`--lime_num_features` to control runtime and plot size.

Feature-importance outputs include native model coefficients/importances when
the estimator exposes them, permutation importance measured as AUC decrease,
feature-selection stability across folds/repeats, grouped summaries by
ROI/modality/image type/feature family, and a selected-feature correlation
heatmap. Use `--importance_top_n`, `--permutation_repeats`, and
`--correlation_top_n` to tune these reports.

Image QC outputs are written under `artifacts/radiomics/qc/` by default:

```text
image_qc_stats.csv
image_qc_failures.csv
images/<sample>__<modality>__<roi>.png
```

Each PNG shows the raw image slice, the preprocessed slice, and the ROI mask
overlay. The statistics CSV records spacing, size, intensity percentiles, and
mask volume for auditability.

## Documentation

Read `docs/workflow.md` for the complete step-by-step workflow and
`docs/manifest_schema.md` for manifest design. Use
`docs/standalone_repository.md` when exporting this framework into a clean
independent repository.
