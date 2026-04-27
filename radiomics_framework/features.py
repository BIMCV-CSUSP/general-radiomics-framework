from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, shapiro, ttest_ind
from sklearn import metrics
from statsmodels.stats.multitest import multipletests


def ensure_directory(path: str | Path) -> Path:
    """Create and return a directory."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def make_sample_id(df: pd.DataFrame, sample_column: str) -> pd.Series:
    """Return a stable sample identifier series."""

    if sample_column in df.columns:
        return df[sample_column].astype(str)
    return pd.Series([f"sample_{index:05d}" for index in range(len(df))], index=df.index)


def prepare_numeric_feature_matrix(
    df: pd.DataFrame,
    *,
    metadata_columns: set[str] | None = None,
) -> pd.DataFrame:
    """Return numeric radiomics features with metadata and diagnostics removed."""

    metadata_columns = metadata_columns or set()
    default_metadata = {
        "sample_id",
        "group_id",
        "patient_id",
        "study_id",
        "label",
        "roi",
        "modality",
        "mask_type",
    }
    all_metadata = metadata_columns | default_metadata
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    drop_columns = [
        column
        for column in numeric_df.columns
        if column in all_metadata or column.startswith("diagnostics_")
    ]
    return numeric_df.drop(columns=drop_columns, errors="ignore")


def score_single_feature(feature_values: pd.Series, labels: np.ndarray) -> dict:
    """Compute univariate statistics for one feature using training data only."""

    clean = feature_values.replace([np.inf, -np.inf], np.nan)
    finite_mask = clean.notna().to_numpy()
    clean_feature = clean.dropna()
    clean_labels = labels[finite_mask]

    if clean_feature.nunique(dropna=True) <= 1 or len(clean_feature) < 4:
        return _empty_feature_score("not_computable")

    class_zero = clean_feature[clean_labels == 0]
    class_one = clean_feature[clean_labels == 1]
    if len(class_zero) < 2 or len(class_one) < 2:
        return _empty_feature_score("not_computable")

    try:
        _, normality_p = shapiro(clean_feature)
    except Exception:
        normality_p = 0.0

    if normality_p > 0.05:
        test_name = "t_test"
        _, p_value = ttest_ind(class_zero, class_one, equal_var=False, nan_policy="omit")
    else:
        test_name = "mann_whitney_u"
        _, p_value = mannwhitneyu(class_zero, class_one, alternative="two-sided")

    feature_array = clean_feature.to_numpy()
    fpr, tpr, thresholds = metrics.roc_curve(clean_labels, feature_array, pos_label=1)
    auc_value = metrics.auc(fpr, tpr)
    direction = "positive_higher"
    if auc_value < 0.5:
        fpr, tpr, thresholds = metrics.roc_curve(clean_labels, feature_array, pos_label=0)
        auc_value = metrics.auc(fpr, tpr)
        direction = "negative_higher"

    best_position = int(np.nanargmax(tpr - fpr))
    return {
        "auc": auc_value,
        "direction": direction,
        "threshold": thresholds[best_position],
        "sensitivity": tpr[best_position],
        "specificity": 1 - fpr[best_position],
        "test": test_name,
        "p_value": p_value,
    }


def _empty_feature_score(test_name: str) -> dict:
    return {
        "auc": np.nan,
        "direction": np.nan,
        "threshold": np.nan,
        "sensitivity": np.nan,
        "specificity": np.nan,
        "test": test_name,
        "p_value": np.nan,
    }


def _score_feature_record(
    feature_name: str,
    feature_values: pd.Series,
    labels: np.ndarray,
    repeat_index: int | None,
    fold_index: int | None,
) -> dict:
    return {
        "Repeat": repeat_index,
        "Fold": fold_index,
        "feature": feature_name,
        **score_single_feature(feature_values, labels),
    }


def infer_feature_limit(
    y_train: np.ndarray,
    n_samples: int,
    *,
    fixed_feature_count: int | None = None,
    min_features: int = 10,
    max_features_cap: int = 60,
    samples_per_feature: int = 25,
    minority_samples_per_feature: int = 8,
) -> int:
    """Infer a conservative feature cap from training-fold size."""

    if fixed_feature_count is not None:
        return max(1, int(fixed_feature_count))

    class_counts = np.bincount(np.asarray(y_train).astype(int))
    minority_count = int(class_counts.min()) if len(class_counts) > 1 else int(class_counts[0])
    sample_bound = max(1, n_samples // max(1, samples_per_feature))
    minority_bound = max(1, minority_count // max(1, minority_samples_per_feature))
    return max(min_features, min(max_features_cap, sample_bound, minority_bound))


def apply_correlation_pruning(
    X_data: pd.DataFrame,
    ranked_features: list[str],
    *,
    correlation_threshold: float,
) -> tuple[list[str], dict[str, list[str]]]:
    """Greedily prune highly correlated features while preserving rank order."""

    if not ranked_features:
        return [], {}

    candidate_df = X_data[ranked_features].replace([np.inf, -np.inf], np.nan).copy()
    candidate_df = candidate_df.fillna(candidate_df.median(numeric_only=True))
    correlation_matrix = candidate_df.corr().abs()

    kept_features: list[str] = []
    removed_by_feature: dict[str, list[str]] = {}
    for feature_name in ranked_features:
        correlated_with_kept = [
            kept_feature
            for kept_feature in kept_features
            if pd.notna(correlation_matrix.loc[feature_name, kept_feature])
            and correlation_matrix.loc[feature_name, kept_feature] >= correlation_threshold
        ]
        if correlated_with_kept:
            removed_by_feature[feature_name] = correlated_with_kept
        else:
            kept_features.append(feature_name)

    return kept_features, removed_by_feature


def select_radiomics_features(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    repeat_index: int | None = None,
    fold_index: int | None = None,
    fixed_feature_count: int | None = None,
    min_features: int = 10,
    max_features_cap: int = 60,
    samples_per_feature: int = 25,
    minority_samples_per_feature: int = 8,
    fdr_alpha: float = 0.05,
    correlation_threshold: float = 0.90,
    max_candidate_pool: int | None = None,
    n_jobs: int | None = None,
) -> tuple[list[str], pd.DataFrame, dict]:
    """Select a leakage-safe radiomics subset using only training data."""

    if n_jobs is None:
        n_jobs = max(1, min(8, os.cpu_count() or 1))

    if n_jobs == 1:
        selection_records = [
            _score_feature_record(feature, X_train[feature], y_train, repeat_index, fold_index)
            for feature in X_train.columns
        ]
    else:
        selection_records = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=32)(
            delayed(_score_feature_record)(
                feature,
                X_train[feature],
                y_train,
                repeat_index,
                fold_index,
            )
            for feature in X_train.columns
        )

    selection_df = pd.DataFrame(selection_records)
    selection_df["q_value"] = np.nan
    valid_mask = selection_df["p_value"].notna()
    if valid_mask.any():
        _, q_values, _, _ = multipletests(
            selection_df.loc[valid_mask, "p_value"],
            alpha=fdr_alpha,
            method="fdr_bh",
        )
        selection_df.loc[valid_mask, "q_value"] = q_values

    selection_df = selection_df.sort_values(
        by=["p_value", "auc", "feature"],
        ascending=[True, False, True],
        na_position="last",
    ).reset_index(drop=True)
    selection_df["selection_rank"] = np.arange(1, len(selection_df) + 1)
    selection_df["passes_fdr"] = selection_df["q_value"] <= fdr_alpha
    selection_df["removed_by_correlation_with"] = ""

    feature_limit = infer_feature_limit(
        y_train=y_train,
        n_samples=X_train.shape[0],
        fixed_feature_count=fixed_feature_count,
        min_features=min_features,
        max_features_cap=max_features_cap,
        samples_per_feature=samples_per_feature,
        minority_samples_per_feature=minority_samples_per_feature,
    )
    feature_limit = min(feature_limit, max(1, X_train.shape[1]))

    ranked_valid_features = selection_df.loc[valid_mask, "feature"].tolist()
    fdr_features = selection_df.loc[selection_df["passes_fdr"], "feature"].tolist()
    candidate_features = fdr_features or ranked_valid_features or X_train.columns.tolist()
    candidate_pool_limit = max_candidate_pool or max(250, feature_limit * 12)
    candidate_features = candidate_features[: min(len(candidate_features), candidate_pool_limit)]

    pruned_features, removed_by_feature = apply_correlation_pruning(
        X_data=X_train,
        ranked_features=candidate_features,
        correlation_threshold=correlation_threshold,
    )
    selected_features = (pruned_features or candidate_features)[:feature_limit]
    selected_set = set(selected_features)
    for feature_name in ranked_valid_features or X_train.columns.tolist():
        if len(selected_features) >= feature_limit:
            break
        if feature_name not in selected_set:
            selected_features.append(feature_name)
            selected_set.add(feature_name)

    for feature_name, correlated_features in removed_by_feature.items():
        selection_df.loc[
            selection_df["feature"] == feature_name,
            "removed_by_correlation_with",
        ] = " | ".join(correlated_features)

    selection_df["is_selected"] = selection_df["feature"].isin(selected_features)
    selection_df["feature_limit"] = feature_limit
    metadata = {
        "feature_limit": feature_limit,
        "n_valid_features": int(valid_mask.sum()),
        "n_fdr_features": int(selection_df["passes_fdr"].sum()),
        "n_candidate_features": len(candidate_features),
        "n_pruned_features": len(pruned_features),
        "correlation_threshold": correlation_threshold,
        "fdr_alpha": fdr_alpha,
        "selection_n_jobs": n_jobs,
    }
    return selected_features, selection_df, metadata
