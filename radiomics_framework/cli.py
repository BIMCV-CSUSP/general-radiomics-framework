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
    init_parser.add_argument("--auto-params", action="store_true")
    init_parser.add_argument("--auto-params-dir", default=None)
    init_parser.add_argument("--auto-params-samples", type=int, default=20)
    init_parser.add_argument("--auto-params-target-bins", type=int, default=32)

    params_parser = subparsers.add_parser(
        "init-pyradiomics-params",
        help="Fingerprint a manifest and write a PyRadiomics YAML per modality.",
    )
    params_parser.add_argument("--manifest", required=True)
    params_parser.add_argument("--project-root", default=None)
    params_parser.add_argument("--modality", action="append", default=[])
    params_parser.add_argument("--mask-column", default=None)
    params_parser.add_argument("--output-dir", default="configs")
    params_parser.add_argument("--max-samples", type=int, default=20)
    params_parser.add_argument("--target-bin-count", type=int, default=32)
    params_parser.add_argument("--label", type=int, default=1)
    params_parser.add_argument(
        "--filename-template", default="pyradiomics_{name}.yaml"
    )
    params_parser.add_argument("--no-save-fingerprint", action="store_true")

    concat_parser = subparsers.add_parser("concatenate", help="Concatenate extracted features.")
    concat_parser.add_argument("--config", required=True)
    concat_parser.add_argument("--roi", default=None)
    concat_parser.add_argument("--shape_reference_modality", default=None)
    concat_parser.add_argument("--output", default=None)

    qc_parser = subparsers.add_parser("qc-images", help="Export raw/preprocessed/mask QC images.")
    qc_parser.add_argument("--config", required=True)
    qc_parser.add_argument("--output_dir", default=None)
    qc_parser.add_argument("--max_cases", type=int, default=12)
    qc_parser.add_argument("--random_state", type=int, default=42)

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
    train_parser.add_argument("--fixed_feature_count", type=int, default=None)
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
    train_parser.add_argument("--test_features", default=None)
    train_parser.add_argument("--calibration_method", choices=["sigmoid", "isotonic"], default="sigmoid")
    train_parser.add_argument("--calibration_cv_splits", type=int, default=3)
    train_parser.add_argument("--export_best_model", action="store_true")
    train_parser.add_argument("--explain_best_model", action="store_true")
    train_parser.add_argument("--shap_max_samples", type=int, default=100)
    train_parser.add_argument("--shap_background_samples", type=int, default=50)
    train_parser.add_argument("--shap_max_display", type=int, default=30)
    train_parser.add_argument("--lime_max_samples", type=int, default=25)
    train_parser.add_argument("--lime_num_features", type=int, default=15)
    train_parser.add_argument("--feature_distribution_top_n", type=int, default=30)
    train_parser.add_argument("--importance_top_n", type=int, default=30)
    train_parser.add_argument("--permutation_repeats", type=int, default=20)
    train_parser.add_argument("--correlation_top_n", type=int, default=50)
    train_parser.add_argument("--skip_report_plots", action="store_true")
    train_parser.add_argument("--skip_lime", action="store_true")
    train_parser.add_argument("--skip_permutation_importance", action="store_true")

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
    elif args.command == "init-pyradiomics-params":
        from radiomics_framework import pyradiomics_params

        manifest_path = Path(args.manifest).resolve()
        project_root = (
            Path(args.project_root).resolve()
            if args.project_root
            else manifest_path.parent
        )
        if args.modality:
            modalities = [
                pyradiomics_params._parse_modality_spec(spec) for spec in args.modality
            ]
        else:
            modalities = pyradiomics_params._infer_modalities_from_manifest(
                manifest_path, args.mask_column
            )
        written = pyradiomics_params.generate_params_for_modalities(
            manifest_path=manifest_path,
            project_root=project_root,
            modalities=modalities,
            output_dir=Path(args.output_dir),
            mask_column=args.mask_column,
            max_samples=args.max_samples,
            target_bin_count=args.target_bin_count,
            label=args.label,
            filename_template=args.filename_template,
            save_fingerprint=not args.no_save_fingerprint,
        )
        for name, result in written.items():
            print(f"{name}: {result.yaml_path}")
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
    elif args.command == "qc-images":
        from radiomics_framework import qc

        config = qc.load_project_config(args.config)
        output_dir = Path(args.output_dir).resolve() if args.output_dir else None
        qc.export_image_qc(
            config,
            output_dir=output_dir,
            max_cases=args.max_cases,
            random_state=args.random_state,
        )
    elif args.command == "train":
        from radiomics_framework import train

        train.run_training(args)


if __name__ == "__main__":
    main()
