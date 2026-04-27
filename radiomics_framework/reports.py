from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from radiomics_framework.features import ensure_directory


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


def _safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(value))


def _predict_probability(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return (scores - scores.min()) / max(scores.max() - scores.min(), 1e-12)
    return model.predict(X).astype(float)


def _binary_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    npv = tn / max(tn + fn, 1)
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    balanced_accuracy = (sensitivity + specificity) / 2
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": precision,
        "npv": npv,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _feature_parts(feature_name: str) -> dict:
    parts = feature_name.split("_")
    image_type_tokens = {
        "original",
        "wavelet",
        "log",
        "square",
        "squareroot",
        "logarithm",
        "exponential",
        "gradient",
        "lbp",
    }
    family_tokens = {
        "firstorder",
        "shape",
        "shape2d",
        "glcm",
        "glrlm",
        "glszm",
        "gldm",
        "ngtdm",
    }
    image_type_index = next(
        (index for index, token in enumerate(parts) if token.lower() in image_type_tokens),
        None,
    )
    family_index = next(
        (index for index, token in enumerate(parts) if token.lower() in family_tokens),
        None,
    )
    roi = parts[0] if parts else ""
    modality = parts[1] if len(parts) > 1 else ""
    if image_type_index is not None:
        prefix_tokens = parts[:image_type_index]
        if len(prefix_tokens) >= 2:
            roi = prefix_tokens[0]
            modality = "_".join(prefix_tokens[1:])
        image_type = parts[image_type_index]
    else:
        image_type = ""
    family = parts[family_index] if family_index is not None else ""
    return {
        "feature": feature_name,
        "roi": roi,
        "modality": modality,
        "image_type": image_type,
        "feature_family": family,
    }


def export_evaluation_plots(
    metrics_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_dir: Path,
    *,
    threshold: float,
) -> None:
    """Write ROC, PR, calibration, confusion, and model-comparison plots."""

    plt = _setup_matplotlib()
    plots_dir = ensure_directory(output_dir / "plots" / "evaluation")

    fig, ax = plt.subplots(figsize=(7, 6))
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        y_true = classifier_df["true_label"].to_numpy()
        y_prob = classifier_df["prob_class_1"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_value = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{classifier_name} (AUC={auc_value:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC curves")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "roc_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        y_true = classifier_df["true_label"].to_numpy()
        y_prob = classifier_df["prob_class_1"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap_value = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, label=f"{classifier_name} (AP={ap_value:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Out-of-fold precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "precision_recall_curves.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        y_true = classifier_df["true_label"].to_numpy()
        y_prob = classifier_df["prob_class_1"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=8, strategy="quantile")
        ax.plot(prob_pred, prob_true, marker="o", label=classifier_name)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.5", linewidth=1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed class-1 fraction")
    ax.set_title("Calibration curves")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "calibration_curves.png", dpi=200)
    plt.close(fig)

    metric_columns = [
        column
        for column in ["oof_auc", "oof_balanced_accuracy", "oof_f1", "oof_mcc"]
        if column in summary_df.columns
    ]
    if metric_columns:
        plot_df = summary_df.set_index("Classifier")[metric_columns]
        fig, ax = plt.subplots(figsize=(max(7, len(plot_df) * 0.8), 5))
        plot_df.plot(kind="bar", ax=ax)
        ax.set_ylim(-1 if "oof_mcc" in metric_columns else 0, 1)
        ax.set_ylabel("Metric value")
        ax.set_title("Out-of-fold model comparison")
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        fig.savefig(plots_dir / "model_comparison_oof_metrics.png", dpi=200)
        plt.close(fig)

    if {"Classifier", "val_auc", "val_balanced_accuracy"}.issubset(metrics_df.columns):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        metrics_df.boxplot(column="val_auc", by="Classifier", ax=axes[0], rot=45)
        axes[0].set_title("Validation AUC by fold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("AUC")
        metrics_df.boxplot(column="val_balanced_accuracy", by="Classifier", ax=axes[1], rot=45)
        axes[1].set_title("Validation balanced accuracy by fold")
        axes[1].set_xlabel("")
        axes[1].set_ylabel("Balanced accuracy")
        fig.suptitle("")
        fig.tight_layout()
        fig.savefig(plots_dir / "fold_metric_distributions.png", dpi=200)
        plt.close(fig)

    confusion_dir = ensure_directory(plots_dir / "confusion_matrices")
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        y_true = classifier_df["true_label"].to_numpy()
        y_pred = (classifier_df["prob_class_1"].to_numpy() >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(matrix, cmap="Blues")
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center")
        ax.set_xticks([0, 1], labels=["Pred 0", "Pred 1"])
        ax.set_yticks([0, 1], labels=["True 0", "True 1"])
        ax.set_title(f"{classifier_name} confusion matrix")
        fig.tight_layout()
        fig.savefig(confusion_dir / f"{_safe_name(classifier_name)}.png", dpi=200)
        plt.close(fig)

    threshold_rows = []
    decision_rows = []
    thresholds = np.linspace(0.01, 0.99, 99)
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        y_true = classifier_df["true_label"].to_numpy()
        y_prob = classifier_df["prob_class_1"].to_numpy()
        prevalence = float(np.mean(y_true))
        for value in thresholds:
            threshold_rows.append(
                {
                    "Classifier": classifier_name,
                    **_binary_metrics_at_threshold(y_true, y_prob, float(value)),
                }
            )
            y_pred = y_prob >= value
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            n = max(len(y_true), 1)
            net_benefit = (tp / n) - (fp / n) * (value / (1 - value))
            treat_all = prevalence - (1 - prevalence) * (value / (1 - value))
            decision_rows.append(
                {
                    "Classifier": classifier_name,
                    "threshold": float(value),
                    "net_benefit": net_benefit,
                    "treat_all_net_benefit": treat_all,
                    "treat_none_net_benefit": 0.0,
                }
            )
    threshold_df = pd.DataFrame(threshold_rows)
    decision_df = pd.DataFrame(decision_rows)
    threshold_df.to_csv(output_dir / "threshold_metrics.csv", index=False)
    decision_df.to_csv(output_dir / "decision_curve.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for classifier_name, classifier_df in threshold_df.groupby("Classifier"):
        axes[0].plot(classifier_df["threshold"], classifier_df["sensitivity"], label=f"{classifier_name} sensitivity")
        axes[0].plot(classifier_df["threshold"], classifier_df["specificity"], linestyle="--", label=f"{classifier_name} specificity")
        axes[1].plot(classifier_df["threshold"], classifier_df["balanced_accuracy"], label=classifier_name)
    axes[0].set_xlabel("Threshold")
    axes[0].set_ylabel("Metric value")
    axes[0].set_title("Sensitivity and specificity")
    axes[0].legend(fontsize=7)
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("Balanced accuracy")
    axes[1].set_title("Threshold sweep")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "threshold_sweep.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 6))
    for classifier_name, classifier_df in decision_df.groupby("Classifier"):
        ax.plot(classifier_df["threshold"], classifier_df["net_benefit"], label=classifier_name)
    first_curve = decision_df.groupby("threshold", as_index=False).first()
    ax.plot(first_curve["threshold"], first_curve["treat_all_net_benefit"], color="0.4", linestyle="--", label="Treat all")
    ax.plot(first_curve["threshold"], first_curve["treat_none_net_benefit"], color="0.2", linestyle=":", label="Treat none")
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title("Decision curve analysis")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(plots_dir / "decision_curve.png", dpi=200)
    plt.close(fig)

    calibration_rows = []
    for classifier_name, classifier_df in aggregated_df.groupby("Classifier"):
        y_true = classifier_df["true_label"].to_numpy()
        y_prob = classifier_df["prob_class_1"].to_numpy()
        calibration_rows.append(
            {
                "Classifier": classifier_name,
                "brier_score": brier_score_loss(y_true, y_prob),
                "mean_predicted_probability": float(np.mean(y_prob)),
                "observed_prevalence": float(np.mean(y_true)),
            }
        )
    pd.DataFrame(calibration_rows).to_csv(output_dir / "calibration_summary.csv", index=False)


def export_feature_distribution_plots(
    X: pd.DataFrame,
    y: np.ndarray,
    selected_features: list[str],
    output_dir: Path,
    *,
    max_features: int,
) -> None:
    """Write distribution plots for the selected radiomics features."""

    plt = _setup_matplotlib()
    distributions_dir = ensure_directory(output_dir / "plots" / "feature_distributions")
    selected = [feature for feature in selected_features if feature in X.columns][:max_features]
    if not selected:
        return

    summary_rows = []
    for feature in selected:
        values = X[feature].replace([np.inf, -np.inf], np.nan)
        for label in sorted(np.unique(y)):
            class_values = values[np.asarray(y) == label].dropna()
            summary_rows.append(
                {
                    "feature": feature,
                    "label": int(label),
                    "n": int(len(class_values)),
                    "mean": class_values.mean(),
                    "std": class_values.std(),
                    "median": class_values.median(),
                    "iqr": class_values.quantile(0.75) - class_values.quantile(0.25),
                }
            )
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "feature_distribution_summary.csv",
        index=False,
    )

    for feature in selected:
        feature_df = pd.DataFrame({"value": X[feature], "label": y}).replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna()
        if feature_df.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        labels = sorted(feature_df["label"].unique())
        axes[0].boxplot(
            [feature_df.loc[feature_df["label"] == label, "value"] for label in labels],
            tick_labels=[str(label) for label in labels],
        )
        axes[0].set_xlabel("Label")
        axes[0].set_ylabel(feature)
        axes[0].set_title("Class distribution")
        for label in labels:
            axes[1].hist(
                feature_df.loc[feature_df["label"] == label, "value"],
                bins=20,
                alpha=0.5,
                label=f"Label {label}",
            )
        axes[1].set_title("Histogram")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(distributions_dir / f"{_safe_name(feature)}.png", dpi=180)
        plt.close(fig)


def export_feature_selection_stability(
    selection_df: pd.DataFrame,
    output_dir: Path,
    *,
    top_n: int,
) -> None:
    """Summarize how often each feature is selected across folds/repeats."""

    if selection_df.empty or "feature" not in selection_df.columns or "is_selected" not in selection_df.columns:
        return

    feature_selection_dir = ensure_directory(output_dir / "feature_selection")
    stability_df = (
        selection_df.groupby("feature", as_index=False)
        .agg(
            selection_count=("is_selected", "sum"),
            evaluated_count=("is_selected", "size"),
            median_rank=("selection_rank", "median"),
            mean_auc=("auc", "mean"),
            min_q_value=("q_value", "min"),
        )
        .assign(selection_frequency=lambda frame: frame["selection_count"] / frame["evaluated_count"].clip(lower=1))
        .sort_values(["selection_frequency", "selection_count", "median_rank"], ascending=[False, False, True])
    )
    parts_df = pd.DataFrame([_feature_parts(feature) for feature in stability_df["feature"]])
    stability_df = stability_df.merge(parts_df, on="feature", how="left")
    stability_df.to_csv(feature_selection_dir / "feature_selection_stability.csv", index=False)

    plt = _setup_matplotlib()
    top_df = stability_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_df))))
    ax.barh(top_df["feature"], top_df["selection_frequency"])
    ax.set_xlabel("Selection frequency across folds/repeats")
    ax.set_title("Most stable selected features")
    fig.tight_layout()
    fig.savefig(feature_selection_dir / "feature_selection_stability.png", dpi=200)
    plt.close(fig)

    group_rows = []
    for group_name in ["roi", "modality", "image_type", "feature_family"]:
        if group_name in stability_df.columns:
            grouped = (
                stability_df.groupby(group_name, dropna=False)
                .agg(
                    mean_selection_frequency=("selection_frequency", "mean"),
                    selected_features=("selection_count", lambda values: int((values > 0).sum())),
                    total_features=("feature", "size"),
                )
                .reset_index()
                .rename(columns={group_name: "group_value"})
            )
            grouped.insert(0, "group_type", group_name)
            group_rows.append(grouped)
    if group_rows:
        pd.concat(group_rows, ignore_index=True).to_csv(
            feature_selection_dir / "feature_selection_group_stability.csv",
            index=False,
        )


def export_selected_feature_correlation(
    X: pd.DataFrame,
    selected_features: list[str],
    output_dir: Path,
    *,
    max_features: int,
) -> None:
    """Export a correlation matrix and heatmap for selected features."""

    features = [feature for feature in selected_features if feature in X.columns][:max_features]
    if len(features) < 2:
        return
    correlation_dir = ensure_directory(output_dir / "plots" / "feature_correlation")
    correlation_df = X[features].replace([np.inf, -np.inf], np.nan).corr(method="spearman")
    correlation_df.to_csv(output_dir / "selected_feature_spearman_correlation.csv")

    plt = _setup_matplotlib()
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(features)), max(6, 0.35 * len(features))))
    image = ax.imshow(correlation_df.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(features)), labels=features, rotation=90, fontsize=7)
    ax.set_yticks(range(len(features)), labels=features, fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Spearman correlation among selected features")
    fig.tight_layout()
    fig.savefig(correlation_dir / "selected_feature_correlation_heatmap.png", dpi=200)
    plt.close(fig)


def export_model_feature_importance(
    fitted_model,
    X_selected: pd.DataFrame,
    y: np.ndarray,
    output_dir: Path,
    *,
    permutation_repeats: int,
    random_state: int,
    n_jobs: int,
    top_n: int,
    logger: logging.Logger,
) -> None:
    """Export native and permutation importances for the final fitted model."""

    importance_dir = ensure_directory(output_dir / "feature_importance")
    feature_names = np.asarray(X_selected.columns.tolist())
    final_estimator = fitted_model
    transformed_feature_names = feature_names
    if hasattr(fitted_model, "steps"):
        final_estimator = fitted_model.steps[-1][1]
        for step_name, step in fitted_model.steps[:-1]:
            if step_name == "variancethreshold" and hasattr(step, "get_support"):
                transformed_feature_names = transformed_feature_names[step.get_support()]

    native_values = None
    native_kind = None
    if hasattr(final_estimator, "feature_importances_"):
        native_values = np.asarray(final_estimator.feature_importances_)
        native_kind = "feature_importances_"
    elif hasattr(final_estimator, "coef_"):
        native_values = np.asarray(final_estimator.coef_)
        if native_values.ndim > 1:
            native_values = native_values[0]
        native_kind = "coef_"

    if native_values is not None and len(native_values) == len(transformed_feature_names):
        native_df = pd.DataFrame(
            {
                "feature": transformed_feature_names,
                "importance": native_values,
                "abs_importance": np.abs(native_values),
                "importance_type": native_kind,
            }
        ).sort_values("abs_importance", ascending=False)
        parts_df = pd.DataFrame([_feature_parts(feature) for feature in native_df["feature"]])
        native_df = native_df.merge(parts_df, on="feature", how="left")
        native_df.to_csv(importance_dir / "model_native_feature_importance.csv", index=False)
        _plot_importance_bar(
            native_df,
            importance_dir / "model_native_feature_importance.png",
            value_column="abs_importance",
            title="Native model feature importance",
            top_n=top_n,
        )
        _export_grouped_importance(native_df, importance_dir / "model_native_group_importance.csv")
    else:
        logger.info("Native feature importance is not available for this final estimator.")

    try:
        permutation_result = permutation_importance(
            fitted_model,
            X_selected,
            y,
            scoring="roc_auc",
            n_repeats=permutation_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    except Exception as exc:
        logger.warning("Permutation importance failed: %s", exc)
        return

    permutation_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": permutation_result.importances_mean,
            "importance_std": permutation_result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    parts_df = pd.DataFrame([_feature_parts(feature) for feature in permutation_df["feature"]])
    permutation_df = permutation_df.merge(parts_df, on="feature", how="left")
    permutation_df.to_csv(importance_dir / "permutation_importance_auc.csv", index=False)
    _plot_importance_bar(
        permutation_df,
        importance_dir / "permutation_importance_auc.png",
        value_column="importance_mean",
        title="Permutation importance by AUC decrease",
        top_n=top_n,
    )
    _export_grouped_importance(
        permutation_df.rename(columns={"importance_mean": "abs_importance"}),
        importance_dir / "permutation_group_importance_auc.csv",
    )
    logger.info("Saved feature-importance outputs to %s", importance_dir)


def _plot_importance_bar(
    importance_df: pd.DataFrame,
    output_path: Path,
    *,
    value_column: str,
    title: str,
    top_n: int,
) -> None:
    plt = _setup_matplotlib()
    top_df = importance_df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_df))))
    ax.barh(top_df["feature"], top_df[value_column])
    ax.set_xlabel(value_column)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _export_grouped_importance(importance_df: pd.DataFrame, output_path: Path) -> None:
    rows = []
    for group_name in ["roi", "modality", "image_type", "feature_family"]:
        if group_name not in importance_df.columns:
            continue
        grouped = (
            importance_df.groupby(group_name, dropna=False)
            .agg(
                total_abs_importance=("abs_importance", "sum"),
                mean_abs_importance=("abs_importance", "mean"),
                n_features=("feature", "size"),
            )
            .reset_index()
            .rename(columns={group_name: "group_value"})
        )
        grouped.insert(0, "group_type", group_name)
        rows.append(grouped)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(output_path, index=False)


def export_lime_interpretability(
    fitted_model,
    X_selected: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    output_dir: Path,
    *,
    max_samples: int,
    num_features: int,
    random_state: int,
    logger: logging.Logger,
) -> None:
    """Export local LIME explanations and aggregate absolute weights."""

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise RuntimeError(
            "LIME interpretability requires lime. Install the project requirements before using --explain_best_model."
        ) from exc

    interpretability_dir = ensure_directory(output_dir / "interpretability")
    rng = np.random.default_rng(random_state)
    if len(X_selected) <= max_samples:
        chosen_positions = np.arange(len(X_selected))
    else:
        chosen_positions = np.sort(rng.choice(len(X_selected), size=max_samples, replace=False))

    feature_names = X_selected.columns.tolist()
    explainer = LimeTabularExplainer(
        training_data=X_selected.to_numpy(),
        feature_names=feature_names,
        class_names=["class_0", "class_1"],
        discretize_continuous=True,
        random_state=random_state,
        mode="classification",
    )

    def predict_proba(data: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(data, columns=feature_names)
        probabilities = _predict_probability(fitted_model, frame)
        return np.column_stack([1.0 - probabilities, probabilities])

    rows = []
    for position in chosen_positions:
        explanation = explainer.explain_instance(
            X_selected.iloc[position].to_numpy(),
            predict_proba,
            num_features=min(num_features, len(feature_names)),
            labels=(1,),
        )
        for rank, (condition, weight) in enumerate(explanation.as_list(label=1), start=1):
            rows.append(
                {
                    "sample_id": str(sample_ids[position]),
                    "sample_position": int(position),
                    "true_label": int(y[position]),
                    "rank": rank,
                    "condition": condition,
                    "weight_class1": float(weight),
                }
            )

    lime_df = pd.DataFrame(rows)
    lime_df.to_csv(interpretability_dir / "lime_local_explanations_class1.csv", index=False)
    if lime_df.empty:
        return

    aggregate_df = (
        lime_df.assign(abs_weight=lime_df["weight_class1"].abs())
        .groupby("condition", as_index=False)
        .agg(
            mean_abs_weight=("abs_weight", "mean"),
            mean_weight=("weight_class1", "mean"),
            n=("weight_class1", "size"),
        )
        .sort_values("mean_abs_weight", ascending=False)
    )
    aggregate_df.to_csv(interpretability_dir / "lime_aggregate_importance_class1.csv", index=False)

    plt = _setup_matplotlib()
    top_df = aggregate_df.head(num_features).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(top_df))))
    ax.barh(top_df["condition"], top_df["mean_abs_weight"])
    ax.set_xlabel("Mean absolute LIME weight")
    ax.set_title("LIME aggregate importance for class 1")
    fig.tight_layout()
    fig.savefig(interpretability_dir / "lime_aggregate_importance_class1.png", dpi=200)
    plt.close(fig)
    logger.info("Saved LIME interpretability outputs to %s", interpretability_dir)


def write_run_manifest(output_dir: Path, payload: dict) -> None:
    """Persist a small JSON manifest describing generated report artifacts."""

    with (output_dir / "report_manifest.json").open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)
