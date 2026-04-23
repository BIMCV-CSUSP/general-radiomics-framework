# General radiomics workflow

This workflow describes how to use the framework for any image domain supported
by SimpleITK and PyRadiomics.

## 1. Prepare a manifest

Create a CSV with one row per sample. At minimum it should contain:

- a sample identifier
- a target label
- an optional group identifier for leakage-safe splitting
- one or more image-path columns
- one or more ROI mask columns, unless using whole-image extraction

See `docs/manifest_schema.md` for examples.

## 2. Configure the project

Copy `configs/example_project.yaml` and edit:

- `project.root`
- `project.manifest`
- `project.output_dir`
- `columns`
- `modalities`
- `rois`
- `preprocessing`

The modality and ROI names are free text. They are used only for output names
and feature prefixes.

## 3. Extract radiomics features

```bash
python -m radiomics_framework.extract --config configs/example_project.yaml
```

This writes one CSV per modality and ROI:

```text
artifacts/radiomics/features_<modality>_<roi>.csv
```

## 4. Concatenate feature tables

```bash
python -m radiomics_framework.concatenate \
  --config configs/example_project.yaml \
  --output artifacts/radiomics/concatenated/features_all.csv
```

The concatenation step:

- merges all extracted modality/ROI tables by `sample_id`, `group_id`, and `label`
- removes PyRadiomics `diagnostics_*` columns
- prefixes features with `<roi>_<modality>_`
- keeps shape features from one reference modality to reduce duplication

## 5. Train and evaluate classical models

```bash
python -m radiomics_framework.train \
  --config configs/example_project.yaml \
  --features artifacts/radiomics/concatenated/features_all.csv \
  --output_dir results/radiomics_framework \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --export_best_model
```

The training step uses reliability-oriented defaults:

- grouped cross-validation by `group_id` when available
- the same fold plan reused across all classifiers
- feature selection performed inside each training fold only
- univariate feature ranking, FDR correction, and correlation pruning
- out-of-fold prediction aggregation across repeats
- group-level bootstrap confidence intervals
- optional final export of the best model fitted on all available data

## 6. Main outputs

```text
results/radiomics_framework/
├── fold_metrics.csv
├── oof_predictions_flat.csv
├── oof_predictions_aggregated.csv
├── summary_metrics.csv
├── bootstrap_group_level_ci.csv
├── best_model.joblib
└── feature_selection/
    └── selected_features_by_fold.csv
```

## Notes for strict final evaluation

Repeated cross-validation is useful for model comparison. If you need a final
unbiased performance estimate, reserve an external test set before extraction or
training, and do not choose feature thresholds, model families, or decision
thresholds on that final test set.
