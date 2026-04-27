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

Generate a first YAML automatically from your manifest:

```bash
python -m radiomics_framework.generate_config \
  --manifest examples/manifest_example.csv \
  --output configs/project.yaml \
  --label-column label
```

For ambiguous datasets, provide explicit column names:

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

Then review the generated YAML. The fields you may edit manually are:

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
python -m radiomics_framework.extract --config configs/project.yaml
```

This writes one CSV per modality and ROI:

```text
artifacts/radiomics/features_<modality>_<roi>.csv
```

## 4. Concatenate feature tables

```bash
python -m radiomics_framework.concatenate \
  --config configs/project.yaml \
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
  --config configs/project.yaml \
  --features artifacts/radiomics/concatenated/features_all.csv \
  --output_dir results/radiomics_framework \
  --feature_strategy most_discriminant \
  --fixed_feature_count 30 \
  --n_splits 5 \
  --n_repeats 10 \
  --calibration_method sigmoid \
  --calibration_cv_splits 3 \
  --bootstrap_iterations 1000 \
  --export_best_model \
  --explain_best_model
```

The training step uses reliability-oriented defaults:

- grouped cross-validation by `group_id` when available
- the same fold plan reused across all classifiers
- feature selection performed inside each training fold only
- univariate feature ranking, FDR correction, and correlation pruning
- out-of-fold prediction aggregation across repeats
- group-level bootstrap confidence intervals
- optional final export of the best model fitted on all available data
- optional SHAP and LIME interpretability for the exported best model
- automatic evaluation plots and selected-feature distribution plots
- feature importance via native estimator values, permutation importance,
  selection stability, group summaries, and selected-feature correlation

## 6. Visual QC of images and masks

```bash
python -m radiomics_framework.qc \
  --config configs/project.yaml \
  --max_cases 24
```

This writes raw/preprocessed/mask overlay panels plus `image_qc_stats.csv`
under `<output_dir>/qc`. Use it before trusting extracted features, especially
after changing preprocessing or ROI definitions.

## 7. Main outputs

```text
results/radiomics_framework/
├── fold_metrics.csv
├── oof_predictions_flat.csv
├── oof_predictions_aggregated.csv
├── summary_metrics.csv
├── bootstrap_group_level_ci.csv
├── threshold_metrics.csv
├── decision_curve.csv
├── calibration_summary.csv
├── best_model.joblib
├── best_model_oof_predictions_calibrated_flat.csv
├── best_model_oof_predictions_calibrated_aggregated.csv
├── plots/
│   ├── evaluation/
│   │   ├── all_models/
│   │   │   ├── roc_curves.png
│   │   │   ├── precision_recall_curves.png
│   │   │   ├── calibration_curves.png
│   │   │   ├── model_comparison_oof_metrics.png
│   │   │   ├── fold_metric_distributions.png
│   │   │   ├── threshold_sweep.png
│   │   │   ├── decision_curve.png
│   │   │   └── confusion_matrices/
│   │   └── best_model/
│   │       ├── roc_curve.png
│   │       ├── precision_recall_curve.png
│   │       ├── calibration_curve.png
│   │       ├── confusion_matrix.png
│   │       ├── threshold_sweep.png
│   │       ├── decision_curve.png
│   │       └── fold_metric_distributions.png
│   ├── feature_distributions/
│   └── feature_correlation/
├── feature_distribution_summary.csv
├── selected_feature_spearman_correlation.csv
├── feature_importance/
│   ├── model_native_feature_importance.csv
│   ├── model_native_feature_importance.png
│   ├── model_native_group_importance.csv
│   ├── permutation_importance_auc.csv
│   ├── permutation_importance_auc.png
│   └── permutation_group_importance_auc.csv
├── interpretability/
│   ├── shap_values_class1.csv
│   ├── shap_explained_feature_values.csv
│   ├── shap_base_values.csv
│   ├── shap_feature_importance.csv
│   ├── shap_bar_class1.png
│   ├── shap_beeswarm_class1.png
│   ├── lime_local_explanations_class1.csv
│   ├── lime_aggregate_importance_class1.csv
│   └── lime_aggregate_importance_class1.png
├── best_model_reports/
│   ├── classification_report_before_calibration.csv
│   ├── classification_report_after_calibration.csv
│   └── external_test/
└── feature_selection/
    ├── selected_features_by_fold.csv
    ├── feature_selection_stability.csv
    ├── feature_selection_stability.png
    └── feature_selection_group_stability.csv
```

## Notes for strict final evaluation

Repeated cross-validation is useful for model comparison. If you need a final
unbiased performance estimate, reserve an external test set before extraction or
training, and do not choose feature thresholds, model families, or decision
thresholds on that final test set.
