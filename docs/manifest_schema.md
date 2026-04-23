# Manifest schema

The framework is driven by a CSV manifest. Each row is one modeling sample, for
example one patient, lesion, study, slice stack, or scan.

## Required modeling columns

- `sample_id`: unique sample identifier. If omitted, extraction can create a stable row-based identifier, but explicit IDs are recommended.
- `label`: target variable. Binary labels are expected by the current training CLI.
- `group_id`: optional grouping identifier used to avoid leakage across folds. Typical values are patient IDs, subject IDs, acquisition IDs, or site IDs.

## Required image columns

Image columns are defined in `configs/example_project.yaml` under `modalities`.
The names are arbitrary:

```yaml
modalities:
  - name: ct
    image_column: ct_path
  - name: pet
    image_column: pet_path
```

With this configuration, the manifest must contain `ct_path` and `pet_path`.
Paths can be absolute or relative to `project.root`.

## Required mask columns

Mask columns are defined under `rois`.

```yaml
rois:
  - name: tumor
    mask_column: tumor_mask_path
    label: 1
  - name: organ
    mask_column: organ_mask_path
    label: 1
```

For whole-image extraction, use `mode: full` and no mask column is required.

```yaml
rois:
  - name: full
    mode: full
```

## Multi-modal example

```csv
sample_id,patient_id,label,ct_path,pet_path,tumor_mask_path
case_001,p001,0,data/ct/case_001.nii.gz,data/pet/case_001.nii.gz,data/masks/case_001.nii.gz
case_002,p002,1,data/ct/case_002.nii.gz,data/pet/case_002.nii.gz,data/masks/case_002.nii.gz
```

## Image format

Any format readable by SimpleITK can be used, including NIfTI, NRRD, MHA/MHD,
DICOM series converted to a volume, and many standard medical-image formats.
