from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedGroupKFold, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from lightgbm import LGBMClassifier

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from radiomics_framework.config import load_project_config
from radiomics_framework.features import (
    ensure_directory,
    prepare_numeric_feature_matrix,
    select_radiomics_features,
)


def setup_logger(output_dir: Path) -> logging.Logger:
    """Create a logger for model training."""

    ensure_directory(output_dir)
    logger = logging.getLogger("radiomics_framework.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "training.log", mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def encode_binary_labels(labels: np.ndarray, positive_label: str | None = None) -> tuple[np.ndarray, dict]:
    """Encode labels as 0/1 and return the mapping used."""

    labels_as_string = pd.Series(labels).astype(str)
    if positive_label is not None:
        y = (labels_as_string == str(positive_label)).astype(int).to_numpy()
        if len(np.unique(y)) != 2:
            raise ValueError(
                f"positive_label={positive_label!r} did not create both classes. "
                "Check the label values in the feature table."
            )
        return y, {"negative": "not_" + str(positive_label), "positive": str(positive_label)}

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    if len(encoder.classes_) != 2:
        raise ValueError(
            "The current training CLI expects binary labels. "
            f"Found classes: {encoder.classes_.tolist()}"
        )
    mapping = {str(class_label): int(code) for code, class_label in enumerate(encoder.classes_)}
    return y, mapping


def get_models(random_state: int = 42) -> dict[str, object]:
    """Build classical model pipelines for radiomics benchmarking."""

    base_steps = [SimpleImputer(strategy="median"), StandardScaler(), VarianceThreshold()]
    models = {
        "SVM": make_pipeline(
            *base_steps,
            SVC(random_state=random_state, class_weight="balanced", probability=True),
        ),
        "LogisticRegression": make_pipeline(
            *base_steps,
            LogisticRegression(
                penalty="elasticnet",
                l1_ratio=0.5,
                class_weight="balanced",
                random_state=random_state,
                solver="saga",
                max_iter=10000,
            ),
        ),
        "RandomForest": make_pipeline(
            *base_steps,
            RandomForestClassifier(
                n_jobs=-1,
                class_weight="balanced_subsample",
                random_state=random_state,
            ),
        ),
        "ExtraTrees": make_pipeline(
            *base_steps,
            ExtraTreesClassifier(n_jobs=-1, class_weight="balanced", random_state=random_state),
        ),
        "DecisionTree": make_pipeline(
            *base_steps,
            DecisionTreeClassifier(class_weight="balanced", random_state=random_state),
        ),
        "NaiveBayes": make_pipeline(*base_steps, GaussianNB()),
        "KNN": make_pipeline(*base_steps, KNeighborsClassifier(n_jobs=-1)),
        "GradientBoosting": make_pipeline(
            *base_steps,
            GradientBoostingClassifier(random_state=random_state),
        ),
        "AdaBoost": make_pipeline(*base_steps, AdaBoostClassifier(random_state=random_state)),
    }
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = make_pipeline(
            *base_steps,
            LGBMClassifier(
                random_state=random_state,
                class_weight="balanced",
                n_jobs=-1,
                verbose=-1,
            ),
        )
    return models


def get_param_distributions() -> dict[str, dict]:
    """Return compact search spaces for optional nested tuning."""

    distributions = {
        "SVM": {
            "svc__C": [0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", 0.001, 0.01, 0.1],
            "svc__kernel": ["rbf"],
        },
        "LogisticRegression": {
            "logisticregression__C": [0.01, 0.1, 1.0, 10.0, 100.0],
            "logisticregression__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
        },
        "RandomForest": {
            "randomforestclassifier__n_estimators": [200, 500],
            "randomforestclassifier__max_depth": [None, 5, 10, 20],
            "randomforestclassifier__min_samples_leaf": [1, 2, 5],
        },
        "ExtraTrees": {
            "extratreesclassifier__n_estimators": [200, 500],
            "extratreesclassifier__max_depth": [None, 5, 10, 20],
            "extratreesclassifier__min_samples_leaf": [1, 2, 5],
        },
        "KNN": {
            "kneighborsclassifier__n_neighbors": [3, 5, 7, 11, 15],
            "kneighborsclassifier__weights": ["uniform", "distance"],
            "kneighborsclassifier__p": [1, 2],
        },
        "GradientBoosting": {
            "gradientboostingclassifier__n_estimators": [100, 200],
            "gradientboostingclassifier__learning_rate": [0.01, 0.05, 0.1],
            "gradientboostingclassifier__max_depth": [2, 3, 5],
        },
        "AdaBoost": {
            "adaboostclassifier__n_estimators": [50, 100, 200],
            "adaboostclassifier__learning_rate": [0.5, 1.0],
        },
    }
    if LIGHTGBM_AVAILABLE:
        distributions["LightGBM"] = {
            "lgbmclassifier__n_estimators": [200, 500],
            "lgbmclassifier__learning_rate": [0.01, 0.05, 0.1],
            "lgbmclassifier__num_leaves": [15, 31, 63],
        }
    return distributions


def predict_probability(model, X: pd.DataFrame) -> np.ndarray:
    """Return class-1 probabilities or scaled decision scores."""

    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    return model.predict(X).astype(float)


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, *, threshold: float) -> dict:
    """Compute threshold-free and threshold-based binary metrics."""

    y_pred = (y_prob >= threshold).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ppv": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "npv": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_fold_plan(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    n_splits: int,
    n_repeats: int,
    random_state: int,
    feature_strategy: str,
    min_features: int,
    max_features_cap: int,
    samples_per_feature: int,
    minority_samples_per_feature: int,
    fdr_alpha: float,
    correlation_threshold: float,
    selection_n_jobs: int,
    logger: logging.Logger,
) -> tuple[list[dict], list[dict]]:
    """Precompute folds and feature subsets once, then reuse them for all models."""

    fold_plan: list[dict] = []
    selection_records: list[dict] = []
    fold_index = 0

    for repeat in range(1, n_repeats + 1):
        repeat_seed = random_state + repeat - 1
        if groups is not None:
            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            splits = splitter.split(X, y, groups=groups)
        else:
            splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat_seed)
            splits = splitter.split(X, y)

        for train_idx, val_idx in splits:
            fold_index += 1
            if feature_strategy == "most_discriminant":
                selected_features, selection_df, selection_metadata = select_radiomics_features(
                    X_train=X.iloc[train_idx],
                    y_train=y[train_idx],
                    repeat_index=repeat,
                    fold_index=fold_index,
                    min_features=min_features,
                    max_features_cap=max_features_cap,
                    samples_per_feature=samples_per_feature,
                    minority_samples_per_feature=minority_samples_per_feature,
                    fdr_alpha=fdr_alpha,
                    correlation_threshold=correlation_threshold,
                    n_jobs=selection_n_jobs,
                )
                selection_records.extend(
                    {**record, **selection_metadata}
                    for record in selection_df.to_dict(orient="records")
                )
            else:
                selected_features = X.columns.tolist()

            fold_plan.append(
                {
                    "Fold": fold_index,
                    "Repeat": repeat,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "selected_features": selected_features,
                }
            )
            logger.info(
                "Prepared fold %d | repeat=%d | train=%d | val=%d | selected_features=%d",
                fold_index,
                repeat,
                len(train_idx),
                len(val_idx),
                len(selected_features),
            )

    return fold_plan, selection_records


def maybe_tune_model(
    model,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    groups_train: np.ndarray | None,
    *,
    tune: bool,
    tune_n_iter: int,
    tune_inner_splits: int,
    random_state: int,
    n_jobs: int,
) -> object:
    """Optionally tune a model inside the training fold."""

    distributions = get_param_distributions().get(model_name)
    if not tune or not distributions:
        return clone(model).fit(X_train, y_train)

    if groups_train is not None and len(np.unique(groups_train)) >= tune_inner_splits:
        cv = GroupKFold(n_splits=tune_inner_splits)
        fit_kwargs = {"groups": groups_train}
    else:
        cv = StratifiedKFold(n_splits=tune_inner_splits, shuffle=True, random_state=random_state)
        fit_kwargs = {}

    search = RandomizedSearchCV(
        estimator=clone(model),
        param_distributions=distributions,
        n_iter=tune_n_iter,
        scoring="roc_auc",
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    search.fit(X_train, y_train, **fit_kwargs)
    return search.best_estimator_


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    groups: np.ndarray | None,
    fold_plan: list[dict],
    *,
    model_names: list[str],
    threshold: float,
    tune: bool,
    tune_n_iter: int,
    tune_inner_splits: int,
    random_state: int,
    n_jobs: int,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate all requested models on the shared fold plan."""

    all_models = get_models(random_state=random_state)
    missing = [name for name in model_names if name not in all_models]
    if missing:
        raise ValueError(f"Unknown models {missing}. Available models: {sorted(all_models)}")

    metrics_rows: list[dict] = []
    prediction_rows: list[dict] = []
    for model_name in model_names:
        logger.info("Starting model: %s", model_name)
        base_model = all_models[model_name]
        for fold in fold_plan:
            selected_features = fold["selected_features"]
            train_idx = fold["train_idx"]
            val_idx = fold["val_idx"]
            X_train = X.iloc[train_idx][selected_features]
            X_val = X.iloc[val_idx][selected_features]
            groups_train = groups[train_idx] if groups is not None else None
            fitted = maybe_tune_model(
                base_model,
                model_name,
                X_train,
                y[train_idx],
                groups_train,
                tune=tune,
                tune_n_iter=tune_n_iter,
                tune_inner_splits=tune_inner_splits,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            train_prob = predict_probability(fitted, X_train)
            val_prob = predict_probability(fitted, X_val)
            train_metrics = compute_binary_metrics(y[train_idx], train_prob, threshold=threshold)
            val_metrics = compute_binary_metrics(y[val_idx], val_prob, threshold=threshold)

            metrics_rows.append(
                {
                    "Classifier": model_name,
                    "Fold": fold["Fold"],
                    "Repeat": fold["Repeat"],
                    "num_selected_features": len(selected_features),
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }
            )
            for local_index, sample_id in enumerate(sample_ids[val_idx]):
                prediction_rows.append(
                    {
                        "Classifier": model_name,
                        "Fold": fold["Fold"],
                        "Repeat": fold["Repeat"],
                        "sample_id": sample_id,
                        "group_id": groups[val_idx][local_index] if groups is not None else sample_id,
                        "true_label": int(y[val_idx][local_index]),
                        "prob_class_1": float(val_prob[local_index]),
                        "predicted_label": int(val_prob[local_index] >= threshold),
                        "selected_features": json.dumps(selected_features),
                    }
                )
            logger.info(
                "%s fold=%d | val_auc=%.4f | val_balanced_accuracy=%.4f",
                model_name,
                fold["Fold"],
                val_metrics["auc"],
                val_metrics["balanced_accuracy"],
            )

    return pd.DataFrame(metrics_rows), pd.DataFrame(prediction_rows)


def aggregate_oof_predictions(predictions_df: pd.DataFrame, *, threshold: float) -> pd.DataFrame:
    """Average repeated out-of-fold probabilities at sample level."""

    aggregated = (
        predictions_df.groupby(["Classifier", "sample_id", "group_id", "true_label"], as_index=False)
        .agg(
            prob_class_1=("prob_class_1", "mean"),
            n_validation_predictions=("prob_class_1", "size"),
        )
        .sort_values(["Classifier", "sample_id"])
    )
    aggregated["predicted_label"] = (aggregated["prob_class_1"] >= threshold).astype(int)
    return aggregated


def bootstrap_group_level_ci(
    aggregated_df: pd.DataFrame,
    *,
    threshold: float,
    n_bootstrap: int,
    ci_level: float,
    random_state: int,
) -> pd.DataFrame:
    """Estimate group-level metric confidence intervals by stratified bootstrap."""

    rng = np.random.default_rng(random_state)
    rows = []
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        point = compute_binary_metrics(
            classifier_df["true_label"].to_numpy(),
            classifier_df["prob_class_1"].to_numpy(),
            threshold=threshold,
        )
        group_rows = {
            group_id: group_df.copy()
            for group_id, group_df in classifier_df.groupby("group_id")
        }
        group_labels = classifier_df.groupby("group_id")["true_label"].agg(lambda values: int(values.iloc[0]))
        strata = {
            label: group_labels[group_labels == label].index.to_numpy()
            for label in sorted(group_labels.unique())
        }
        samples = {metric_name: [] for metric_name in point}
        for _ in range(n_bootstrap):
            sampled_frames = []
            for group_ids in strata.values():
                sampled_group_ids = rng.choice(group_ids, size=len(group_ids), replace=True)
                sampled_frames.extend(group_rows[group_id] for group_id in sampled_group_ids)
            sampled_df = pd.concat(sampled_frames, ignore_index=True)
            if sampled_df["true_label"].nunique() < 2:
                continue
            boot_metrics = compute_binary_metrics(
                sampled_df["true_label"].to_numpy(),
                sampled_df["prob_class_1"].to_numpy(),
                threshold=threshold,
            )
            for metric_name, value in boot_metrics.items():
                samples[metric_name].append(value)

        alpha = 1.0 - ci_level
        for metric_name, point_value in point.items():
            values = samples[metric_name]
            rows.append(
                {
                    "Classifier": classifier_name,
                    "metric": metric_name,
                    "point_estimate": point_value,
                    "ci_low": float(np.nanpercentile(values, 100 * alpha / 2)) if values else np.nan,
                    "ci_high": float(np.nanpercentile(values, 100 * (1 - alpha / 2))) if values else np.nan,
                    "n_bootstrap_success": len(values),
                }
            )
    return pd.DataFrame(rows)


def summarize_performance(metrics_df: pd.DataFrame, aggregated_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Create one row per classifier with fold and OOF summaries."""

    rows = []
    for classifier_name, fold_df in metrics_df.groupby("Classifier"):
        classifier_oof = aggregated_df[aggregated_df["Classifier"] == classifier_name]
        oof_metrics = compute_binary_metrics(
            classifier_oof["true_label"].to_numpy(),
            classifier_oof["prob_class_1"].to_numpy(),
            threshold=threshold,
        )
        rows.append(
            {
                "Classifier": classifier_name,
                "val_auc_mean": fold_df["val_auc"].mean(),
                "val_auc_median": fold_df["val_auc"].median(),
                "val_balanced_accuracy_mean": fold_df["val_balanced_accuracy"].mean(),
                "oof_auc": oof_metrics["auc"],
                "oof_balanced_accuracy": oof_metrics["balanced_accuracy"],
                "oof_f1": oof_metrics["f1"],
                "oof_mcc": oof_metrics["mcc"],
                "n_samples": len(classifier_oof),
                "n_groups": classifier_oof["group_id"].nunique(),
            }
        )
    return pd.DataFrame(rows).sort_values(["oof_auc", "val_auc_median"], ascending=False)


def fit_export_model(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None,
    *,
    model_name: str,
    feature_strategy: str,
    output_dir: Path,
    selection_args: dict,
    random_state: int,
) -> None:
    """Fit the selected best model on all data and save it with feature metadata."""

    if feature_strategy == "most_discriminant":
        selected_features, selection_df, _ = select_radiomics_features(
            X_train=X,
            y_train=y,
            repeat_index=None,
            fold_index=None,
            **selection_args,
        )
        selection_df.to_csv(output_dir / "final_model_feature_selection.csv", index=False)
    else:
        selected_features = X.columns.tolist()

    model = get_models(random_state=random_state)[model_name]
    fitted = clone(model).fit(X[selected_features], y)
    payload = {
        "model": fitted,
        "model_name": model_name,
        "selected_features": selected_features,
        "uses_grouping": groups is not None,
    }
    joblib.dump(payload, output_dir / "best_model.joblib", compress=3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train classical ML models from radiomics features.")
    parser.add_argument("--config", required=True, help="Project YAML configuration.")
    parser.add_argument("--features", required=True, help="Concatenated radiomics feature CSV.")
    parser.add_argument("--output_dir", default="results/radiomics_framework", help="Directory for results.")
    parser.add_argument("--models", nargs="+", default=["SVM", "LogisticRegression", "RandomForest", "KNN"])
    parser.add_argument(
        "--positive_label",
        default=None,
        help="Optional raw label value to force-map to class 1. Defaults to LabelEncoder ordering.",
    )
    parser.add_argument("--feature_strategy", choices=["all", "most_discriminant"], default="most_discriminant")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--n_repeats", type=int, default=10)
    parser.add_argument("--classification_threshold", type=float, default=0.5)
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--ci_level", type=float, default=0.95)
    parser.add_argument("--min_features", type=int, default=10)
    parser.add_argument("--max_features_cap", type=int, default=60)
    parser.add_argument("--samples_per_feature", type=int, default=25)
    parser.add_argument("--minority_samples_per_feature", type=int, default=8)
    parser.add_argument("--fdr_alpha", type=float, default=0.05)
    parser.add_argument("--correlation_threshold", type=float, default=0.90)
    parser.add_argument("--selection_n_jobs", type=int, default=1)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--tune", action="store_true", help="Enable nested RandomizedSearchCV tuning.")
    parser.add_argument("--tune_n_iter", type=int, default=20)
    parser.add_argument("--tune_inner_splits", type=int, default=3)
    parser.add_argument("--search_n_jobs", type=int, default=1)
    parser.add_argument("--export_best_model", action="store_true")
    return parser.parse_args()


def run_training(args: argparse.Namespace) -> None:
    """Run the full training workflow from parsed arguments."""

    config = load_project_config(args.config)
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = config.root / output_dir
    output_dir = output_dir.resolve()
    logger = setup_logger(output_dir)

    feature_path = Path(args.features)
    if not feature_path.is_absolute():
        feature_path = (config.root / feature_path).resolve()
    df = pd.read_csv(feature_path)
    required = {"sample_id", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Feature table is missing required columns: {sorted(missing)}")

    sample_ids = df["sample_id"].astype(str).to_numpy()
    groups = df["group_id"].astype(str).to_numpy() if "group_id" in df.columns else None
    y, label_mapping = encode_binary_labels(df["label"].to_numpy(), args.positive_label)
    metadata_columns = {"sample_id", "label", "group_id"}
    X = prepare_numeric_feature_matrix(df, metadata_columns=metadata_columns)
    if X.empty:
        raise ValueError("No numeric radiomics feature columns were found.")

    logger.info("Loaded features: rows=%d | numeric_features=%d", len(X), X.shape[1])
    logger.info("Label mapping: %s", label_mapping)
    fold_plan, selection_records = build_fold_plan(
        X,
        y,
        groups,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
        feature_strategy=args.feature_strategy,
        min_features=args.min_features,
        max_features_cap=args.max_features_cap,
        samples_per_feature=args.samples_per_feature,
        minority_samples_per_feature=args.minority_samples_per_feature,
        fdr_alpha=args.fdr_alpha,
        correlation_threshold=args.correlation_threshold,
        selection_n_jobs=args.selection_n_jobs,
        logger=logger,
    )

    metrics_df, predictions_df = evaluate_models(
        X,
        y,
        sample_ids,
        groups,
        fold_plan,
        model_names=args.models,
        threshold=args.classification_threshold,
        tune=args.tune,
        tune_n_iter=args.tune_n_iter,
        tune_inner_splits=args.tune_inner_splits,
        random_state=args.random_state,
        n_jobs=args.search_n_jobs,
        logger=logger,
    )
    aggregated_df = aggregate_oof_predictions(
        predictions_df,
        threshold=args.classification_threshold,
    )
    summary_df = summarize_performance(metrics_df, aggregated_df, args.classification_threshold)
    ci_df = bootstrap_group_level_ci(
        aggregated_df,
        threshold=args.classification_threshold,
        n_bootstrap=args.bootstrap_iterations,
        ci_level=args.ci_level,
        random_state=args.random_state,
    )

    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    predictions_df.to_csv(output_dir / "oof_predictions_flat.csv", index=False)
    aggregated_df.to_csv(output_dir / "oof_predictions_aggregated.csv", index=False)
    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    ci_df.to_csv(output_dir / "bootstrap_group_level_ci.csv", index=False)
    if selection_records:
        selection_dir = ensure_directory(output_dir / "feature_selection")
        pd.DataFrame(selection_records).to_csv(selection_dir / "selected_features_by_fold.csv", index=False)
    with (output_dir / "label_mapping.json").open("w", encoding="utf-8") as file_handle:
        json.dump(label_mapping, file_handle, indent=2)

    best_model = summary_df.iloc[0]["Classifier"]
    logger.info("Best model by aggregated OOF AUC: %s", best_model)
    if args.export_best_model:
        fit_export_model(
            X,
            y,
            groups,
            model_name=best_model,
            feature_strategy=args.feature_strategy,
            output_dir=output_dir,
            selection_args={
                "min_features": args.min_features,
                "max_features_cap": args.max_features_cap,
                "samples_per_feature": args.samples_per_feature,
                "minority_samples_per_feature": args.minority_samples_per_feature,
                "fdr_alpha": args.fdr_alpha,
                "correlation_threshold": args.correlation_threshold,
                "n_jobs": args.selection_n_jobs,
            },
            random_state=args.random_state,
        )
        logger.info("Saved final fitted model to %s", output_dir / "best_model.joblib")


def main() -> None:
    run_training(parse_args())


if __name__ == "__main__":
    main()
