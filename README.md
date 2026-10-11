# cineQMAE: Raw Data Preparation and Pretraining

This package contains only raw private/public CMR preparation and pretraining.
It does not need the old QCD-MAE project, V5 output folder, downstream training
code, or experiment-signature checks. Start with `Pretraining.ipynb` in this folder.

## Environment

Use Python 3.8 or newer and a working PyTorch/torchvision combination matching
your CUDA installation. Install missing dependencies from `requirements.txt`;
this file does not reinstall PyTorch. No dependency/model download occurs during
data preparation or training.

The bundled `vendor/mae_st` source and license are retained unchanged. TrackMAE
also requires an offline CoTracker3 source checkout and `scaled_offline.pth`.
Set `motion.cotracker_repo` and `motion.cotracker_checkpoint` to your existing
files, or obtain them from the [official repository](https://github.com/facebookresearch/co-tracker)
and [official offline weights](https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth).
Keep your already-used teacher version for numerical comparability. These large
third-party weights are not bundled. Other methods do not require CoTracker.

## Run

1. Edit paths and `SOURCE` in the notebook's first code cell.
2. Run raw preparation. By default only the selected source is prepared; change
   `PREPARE_SOURCES` to `["private", "public"]` to prepare both independently.
3. Run motion-cache preparation and pretraining.

```python
settings["pretrain_source"] = "private"  # or "public"
settings["methods"] = ["videomae", "heartmae", "trackmae", "cineqmae"]
```

The CLI is equivalent:

```bash
python run_pretrain.py --settings paths.json --source public --methods cineqmae
python run_pretrain.py --settings paths.json --source private --prepare-only
python run_pretrain.py --settings paths.json --source private --resume
```

`paths.json` can contain partial nested overrides, for example:

```json
{
  "private_root": "/public/home/chenweilin/CMRDataSet",
  "public_raw_roots": {
    "acdc": "/data/SAX/ACDC",
    "mnm1": "/data/SAX/MnM1",
    "mnm2": "/data/SAX/MnM2"
  },
  "public_cache_root": "/public/home/chenweilin/CMRDataSet/public_cmr_cache",
  "output_root": "/public/home/chenweilin/CMRDataSet/training_outputs",
  "run_name": "cineq_paper_public",
  "motion": {"cache_root": "/public/home/chenweilin/CMRDataSet/motion_cache"}
}
```

## Public Raw Data

Set the three `public_raw_roots` explicitly, or let `public_raw_root` discover
the extracted ACDC/MnM1/MnM2 directories. Expected layouts:

```text
ACDC/{training,testing}/patientXXX/
  Info.cfg
  patientXXX_4d.nii.gz
MnM1/
  *diagnosis*opendataset.csv
  Training/{Labeled,Unlabeled}/PATIENT/
  Validation/PATIENT/
  Testing/PATIENT/
    PATIENT_sa.nii.gz              # original 4D release
    # alternatively PATIENT_frameXX.nii.gz (the historical frame-series layout)
MnM2/
  dataset_information.csv
  dataset/PATIENT/
    PATIENT_SA_CINE.nii.gz
    PATIENT_SA_ED.nii.gz
    PATIENT_SA_ES.nii.gz
```

Pretraining does not require segmentation labels or the extra `MMs_226.csv`
classification list. MnM1 ED/ES indices come from metadata; ACDC uses Info.cfg;
MnM2 matches supplied ED/ES volumes to cine frames as in the historical code.

Public cine processing retains the previous V5 recipe: canonical orientation,
1.5 mm XY spacing, centered crop/pad to 224, center slices + `[-2,0,2]`, ED-aligned
16-frame periodic interpolation, percentile 1/99 normalization. FP16 disk
caches are converted to FP32 by the unchanged training dataset. No segmentation
or EF targets are computed in this pretraining-only package.

`SOURCE="public"` uses **every discovered patient from all three datasets**:
train/validation/test, NOR, classification-ineligible and unlabeled cases.
The actual count comes from raw files, not a hardcoded target number. Official
splits are saved only as provenance. No classification subset determines this pool.

**Paper protocol:** public-full pretraining includes unlabeled downstream test
images and is therefore transductive, not an unseen-test/inductive evaluation.
Report this exposure explicitly. Public snapshot/monitor images also belong to
the training pool; their reconstruction metrics are not held-out validation.
