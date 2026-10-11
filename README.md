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

## Private Raw Data

`private_train_csv` and `private_test_csv` require `sax_path,disease`; optional
columns are `patient_id` and `ed_index` (explicitly zero-based). Relative paths
resolve under `private_root`. NIfTI storage must be native `X,Y,Z,T`.

The original private preprocessing is unchanged: select original Z center +
`[-2,0,2]`, preserve native XY and full field of view, resize/pad using physical
aspect ratio, normalize selected slices at percentiles 1/99, and periodically
interpolate to 16 frames. Cache storage is FP32 `[3,16,224,224]`.

Without a previous manifest, the original class-stratified 20% validation rule
and `split_seed=0` are used. Set `private_previous_manifest` to the old manifest
only when you need identical historical patient membership; it is not required
for a new experiment. Existing `patients.csv` preserves the chosen split. To
change it deliberately, run `prepare_data(..., overwrite=True)`; valid image
caches remain reusable. Failed cases are listed in `native_prepare_report.csv`.

Private pretraining uses **train only**. Private validation/test patients do
not enter training. NOR remains in pretraining, as in the supplied notebook.

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

## Unchanged Numerical Recipe

- Pretraining: 400 epochs, batch 32, accumulation 1, LR `1e-4`, warmup 10,
  cosine schedule, minimum LR `1e-6`, AdamW decay `0.05`, betas `(0.9,0.95)`,
  gradient clipping `1.0`; FP32, AMP/TF32 off, no per-epoch validation.
- Encoder: 768 dimensions / 12 layers / 12 heads; 16 frames, tubelet 4,
  patch 16, mask ratio `0.75`, raw-intensity reconstruction.
- Shared decoder: 512 dimensions / 8 layers / 16 heads. Query fraction `0.10`,
  top share `0.70`, chunk size 2048; latest 34-query sparse-path implementation.
- cineQ losses: reconstruction + DIS + DPCC; DIS weight `1.0`, Jacobian weight
  `0.1` inside DIS, DPCC weight `0.2`. TGWR uses `M` without baseline, alpha `1.0`.
- Input augmentation: horizontal flip `0.5`, intensity jitter `0.1`.
- Snapshots: every 50 epochs, up to 25 fixed patients, gain 2.5, maps 896 pixels.
  Reconstruction and motion snapshots remain enabled. To intentionally disable
  their separate registration/visualization cost set `snapshots.enabled=False`.

Model, loss, optimizer, motion-provider and visualization kernels are reused
from the latest historical implementation. VideoMAE/HeartMAE/TrackMAE are the
existing controlled CMR adaptations on the common MAE-ST backbone, **not full
reproductions of each original paper's training recipe**.

## Outputs and Resume

```text
OUTPUT_ROOT/RUN_NAME/
  settings_latest.json
  data_protocol.json
  pretrain_patients.csv
  patient_counts.csv
  pretrain_results.csv
  pretrain/METHOD/seed0/
    last.pt                       # full model + optimizer/scaler/RNG/history
    encoder_last.pt
    history.csv
    summary.json
    epoch0050.pt                  # periodic full model
    snapshots/epoch0050/...
```

`last.pt` retains the usual `settings,model,method,seed,epoch` keys for later
downstream loading. `resume=True` loads the actual checkpoint directly, with
ordinary strict tensor-shape loading but no settings/cache/code-signature gate.
Keep source, architecture, cohort and recipe unchanged for a scientifically
meaningful continuation. `resume=False` will not overwrite an existing checkpoint.
Use different run names for private/public experiments. The old outputs are not
edited by this package.

## Checks

`python -m unittest discover -s tests -v` covers configuration retention,
source/patient selection, scanners, cache generation and orchestration using
small fixtures/mocks. It does not replace a real CUDA/data training run.
