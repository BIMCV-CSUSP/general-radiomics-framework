from __future__ import annotations

import argparse
from pathlib import Path

from radiomics_framework import generate_config


def main() -> None:
    parser = argparse.ArgumentParser(description="General radiomics framework CLI.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract_parser = subparsers.add_parser("extract", help="Extract PyRadiomics features.")
    extract_parser.add_argument("--config", required=True)

    init_parser = subparsers.add_parser("init-config", help="Generate a project YAML from a manifest CSV.")
    init_parser.add_argument("--manifest", required=True)
    init_parser.add_argument("--output", default="configs/project.yaml")
    init_parser.add_argument("--project-name", default=None)
    init_parser.add_argument("--project-root", default=None)
    init_parser.add_argument("--output-dir", default="artifacts/radiomics")
    init_parser.add_argument("--sample-id-column", default=None)
    init_parser.add_argument("--label-column", default=None)
    init_parser.add_argument("--group-id-column", default=None)
    init_parser.add_argument("--image-column", action="append")
    init_parser.add_argument("--mask-column", action="append")
    init_parser.add_argument("--params", default=None)
    init_parser.add_argument("--mask-label", type=int, default=1)
    init_parser.add_argument("--include-full-roi", action="store_true")
    init_parser.add_argument("--n-jobs", type=int, default=1)
    init_parser.add_argument("--n4-bias-correction", action="store_true")
    init_parser.add_argument("--denoise", action="store_true")

    concat_parser = subparsers.add_parser("concatenate", help="Concatenate extracted features.")
    concat_parser.add_argument("--config", required=True)
    concat_parser.add_argument("--roi", default=None)
    concat_parser.add_argument("--shape_reference_modality", default=None)
    concat_parser.add_argument("--output", default=None)

    train_parser = subparsers.add_parser("train", help="Train radiomics models.")
    train_parser.add_argument("--config", required=True)
    train_parser.add_argument("--features", required=True)
    train_parser.add_argument("--output_dir", default="results/radiomics_framework")
    train_parser.add_argument("--models", nargs="+", default=["SVM", "LogisticRegression", "RandomForest", "KNN"])
    train_parser.add_argument("--positive_label", default=None)
    train_parser.add_argument("--feature_strategy", choices=["all", "most_discriminant"], default="most_discriminant")
    train_parser.add_argument("--n_splits", type=int, default=5)
    train_parser.add_argument("--n_repeats", type=int, default=10)
    train_parser.add_argument("--classification_threshold", type=float, default=0.5)
    train_parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    train_parser.add_argument("--ci_level", type=float, default=0.95)
    train_parser.add_argument("--min_features", type=int, default=10)
    train_parser.add_argument("--max_features_cap", type=int, default=60)
    train_parser.add_argument("--samples_per_feature", type=int, default=25)
    train_parser.add_argument("--minority_samples_per_feature", type=int, default=8)
    train_parser.add_argument("--fdr_alpha", type=float, default=0.05)
    train_parser.add_argument("--correlation_threshold", type=float, default=0.90)
    train_parser.add_argument("--selection_n_jobs", type=int, default=1)
    train_parser.add_argument("--random_state", type=int, default=42)
    train_parser.add_argument("--tune", action="store_true")
    train_parser.add_argument("--tune_n_iter", type=int, default=20)
    train_parser.add_argument("--tune_inner_splits", type=int, default=3)
    train_parser.add_argument("--search_n_jobs", type=int, default=1)
    train_parser.add_argument("--export_best_model", action="store_true")

    args = parser.parse_args()
    if args.command == "init-config":
        if args.group_id_column == "":
            args.group_id_column = None
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = generate_config.build_config_payload(args)
        output.write_text(generate_config.dump_yaml(payload), encoding="utf-8")
        print(f"Generated config: {output}")
        print(f"Modalities: {[item['name'] for item in payload['modalities']]}")
        print(f"ROIs: {[item['name'] for item in payload['rois']]}")
    elif args.command == "extract":
        from radiomics_framework import extract

        config = extract.load_project_config(args.config)
        extract.run_extraction(config)
    elif args.command == "concatenate":
        from radiomics_framework import concatenate

        config = concatenate.load_project_config(args.config)
        output = Path(args.output).resolve() if args.output else None
        if output is None:
            output = config.output_dir / "concatenated" / "features_all.csv"
        logger = concatenate.setup_logger(output)
        logger.info("Concatenating features for project=%s", config.name)
        df = concatenate.build_concatenated_table(
            config,
            roi_filter=args.roi,
            shape_reference_modality=args.shape_reference_modality,
        )
        concatenate.ensure_directory(output.parent)
        df.to_csv(output, index=False)
        logger.info("Saved concatenated table to %s", output)
    elif args.command == "train":
        from radiomics_framework import train

        train.run_training(args)


if __name__ == "__main__":
    main()
