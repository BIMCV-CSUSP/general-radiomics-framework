# Manifest schema

The framework is driven by a CSV manifest. Each row is one modeling sample, for
example one patient, lesion, study, slice stack, or scan.

## Required modeling columns

- `sample_id`: unique sample identifier. If omitted, extraction can create a stable row-based identifier, but explicit IDs are recommended.
- `label`: optional target variable. Binary labels are expected by the current training CLI, but extraction and concatenation can run with `label: null`.
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

## Generate the YAML automatically

Once the manifest exists, generate a first configuration with:

```bash
python -m radiomics_framework.generate_config \
  --manifest /path/to/manifest.csv \
  --output configs/project.yaml
```

The generator detects common column names and path-like values. If a dataset has
ambiguous names, pass columns explicitly:

```bash
python -m radiomics_framework.generate_config \
  --manifest /path/to/manifest.csv \
  --output configs/project.yaml \
  --label-column outcome \
  --group-id-column patient_id \
  --image-column ct_path \
  --mask-column lesion_mask_path
```

For a manifest without labels, such as:

```csv
patient_id,study_id,T1,T2,mask
PIT_001,S001,/path/to/PIT_001_0000.nii.gz,/path/to/PIT_001_0001.nii.gz,/path/to/PIT_001_mask.nii.gz
```

the generator will write `label: null`. You can extract and concatenate
features, then add a label column later before running training.
