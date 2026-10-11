"""Public raw-CMR scanners and unchanged V5 cine geometry (no old cache needed)."""


from __future__ import annotations


import csv


import json


import re


from pathlib import Path


from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union


import numpy as np


import pandas as pd


import nibabel as nib


from nibabel.processing import resample_from_to, resample_to_output


from tqdm.auto import tqdm


CLASS_NAMES = {'acdc': ('NOR', 'DCM', 'HCM', 'MINF', 'RV'), 'mnm1': ('NOR', 'DCM', 'HCM'), 'mnm2': ('NOR', 'LV', 'HCM', 'ARR', 'FALL', 'CIA')}


PATIENT_COLUMNS = ('patient_uid', 'dataset', 'patient_id', 'official_split', 'cine_format', 'cine_path', 'disease', 'vendor', 'ed_index', 'es_index', 'ed_image_path', 'es_image_path', 'ed_label_path', 'es_label_path', 'has_labels', 'classification_eligible', 'pretrain_eligible')


def _find_one(root: Path, pattern: str) -> Path:
    matches = sorted((path for path in root.rglob(pattern) if path.is_file()))
    if not matches:
        raise FileNotFoundError(f'Could not find {pattern!r} under {root}')
    if len(matches) > 1:
        exact = [path for path in matches if path.name == pattern]
        if len(exact) == 1:
            return exact[0]
    return matches[0]


def _read_cfg(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        if ':' in line:
            key, value = line.split(':', 1)
            result[key.strip()] = value.strip()
    return result


def _uid(dataset: str, patient_id: str) -> str:
    return f'{dataset}:{patient_id}'


def scan_acdc(root: Union[str, Path]) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    info_files = sorted(root.rglob('patient*/Info.cfg'))
    if not info_files:
        entries = sorted((path.name for path in root.iterdir()))[:10] if root.is_dir() else []
        raise FileNotFoundError(f'No ACDC patient Info.cfg files found under {root}; root_exists={root.exists()}, first_entries={entries}')
    log(f'ACDC scanner found {len(info_files)} Info.cfg files under {root}')
    rows: List[Dict[str, Any]] = []
    for info in progress(info_files, total=len(info_files), desc='scan ACDC', unit='patient'):
        patient_dir = info.parent
        patient_id = patient_dir.name
        split_name = patient_dir.parent.name.lower()
        if split_name not in {'training', 'testing'}:
            continue
        metadata = _read_cfg(info)
        ed = int(metadata['ED'])
        es = int(metadata['ES'])
        cine = patient_dir / f'{patient_id}_4d.nii.gz'
        ed_image = patient_dir / f'{patient_id}_frame{ed:02d}.nii.gz'
        es_image = patient_dir / f'{patient_id}_frame{es:02d}.nii.gz'
        ed_label = patient_dir / f'{patient_id}_frame{ed:02d}_gt.nii.gz'
        es_label = patient_dir / f'{patient_id}_frame{es:02d}_gt.nii.gz'
        required = (cine,)
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Incomplete ACDC case {patient_id}: {missing}')
        rows.append({'patient_uid': _uid('acdc', patient_id), 'dataset': 'acdc', 'patient_id': patient_id, 'official_split': 'train' if split_name == 'training' else 'test', 'cine_format': 'nifti4d', 'cine_path': str(cine), 'disease': metadata.get('Group', '').upper(), 'vendor': '', 'ed_index': ed - 1, 'es_index': es - 1, 'ed_image_path': str(ed_image), 'es_image_path': str(es_image), 'ed_label_path': str(ed_label), 'es_label_path': str(es_label), 'has_labels': ed_label.is_file() and es_label.is_file(), 'classification_eligible': metadata.get('Group', '').upper() in CLASS_NAMES['acdc'], 'pretrain_eligible': split_name == 'training'})
    return pd.DataFrame(rows, columns=PATIENT_COLUMNS)


def _read_mnm1_classification_list(path: Path) -> Dict[str, str]:
    frame = pd.read_csv(path)
    result: Dict[str, str] = {}
    for disease in ('NOR', 'DCM', 'HCM'):
        if disease not in frame.columns:
            raise KeyError(f'{path} is missing classification column {disease}')
        for value in frame[disease].dropna().astype(str):
            value = value.strip()
            if value:
                result[value] = disease
    return result


def scan_mnm1(root: Union[str, Path]) -> pd.DataFrame:
    root = Path(root).expanduser().resolve()
    diagnosis_path = _find_one(root, '*diagnosis*opendataset.csv')
    metadata = pd.read_csv(diagnosis_path).dropna(subset=['External code'])
    metadata['External code'] = metadata['External code'].astype(str)
    lookup = metadata.set_index('External code').to_dict('index')
    class_files = sorted(root.rglob('MMs_226.csv'))
    class_lookup = _read_mnm1_classification_list(class_files[0]) if class_files else {}
    split_dirs: List[Tuple[str, Path]] = []
    candidates = {'train_labeled': list(root.rglob('Training/Labeled')), 'train_unlabeled': list(root.rglob('Training/Unlabeled')), 'val': list(root.rglob('Validation')), 'test': list(root.rglob('Testing'))}
    for split, paths in candidates.items():
        if not paths:
            raise FileNotFoundError(f'Missing M&M1 {split} directory under {root}')
        split_dirs.append((split, sorted(paths)[0]))
    patient_entries = [(split, patient_dir) for split, split_dir in split_dirs for patient_dir in sorted((path for path in split_dir.iterdir() if path.is_dir()))]
    log(f'M&M1 scanner found {len(patient_entries)} patient directories under {root}')
    rows: List[Dict[str, Any]] = []
    for split, patient_dir in progress(patient_entries, total=len(patient_entries), desc='scan M&M1', unit='patient'):
        patient_id = patient_dir.name
        if patient_id not in lookup:
            raise KeyError(f'M&M1 metadata is missing patient {patient_id}')
        item = lookup[patient_id]
        images = sorted((path for path in patient_dir.glob(f'{patient_id}_frame*.nii.gz') if not path.name.endswith('_gt.nii.gz')))
        ed = int(item['ED'])
        es = int(item['ES'])
        if not images:
            cine = patient_dir / (patient_id + '_sa.nii.gz')
            if not cine.is_file():
                raise FileNotFoundError(f'No 4D cine or frame series for M&M1 patient {patient_id}')
            labels = patient_dir / (patient_id + '_sa_gt.nii.gz')
            disease = class_lookup.get(patient_id, str(item['Pathology']).upper())
            rows.append(dict(patient_uid=_uid('mnm1', patient_id), dataset='mnm1', patient_id=patient_id,
                             official_split=split, cine_format='nifti4d', cine_path=str(cine),
                             disease=disease, vendor=str(item['Vendor']), ed_index=ed, es_index=es,
                             ed_image_path='', es_image_path='', ed_label_path=str(labels),
                             es_label_path=str(labels), has_labels=labels.is_file(),
                             classification_eligible=disease in CLASS_NAMES['mnm1'],
                             pretrain_eligible=split in {'train_labeled', 'train_unlabeled'}))
            continue
        frame_numbers = sorted(_frame_number(p) for p in images)
        if frame_numbers != list(range(frame_numbers[0], frame_numbers[-1] + 1)):
            raise ValueError(f'Missing or duplicate M&M1 time frames for {patient_id}: {frame_numbers}')
        ed_image = patient_dir / f'{patient_id}_frame{ed:02d}.nii.gz'
        es_image = patient_dir / f'{patient_id}_frame{es:02d}.nii.gz'
        ed_label = patient_dir / f'{patient_id}_frame{ed:02d}_gt.nii.gz'
        es_label = patient_dir / f'{patient_id}_frame{es:02d}_gt.nii.gz'
        has_labels = ed_label.is_file() and es_label.is_file()
        missing_images = [str(path) for path in (ed_image, es_image) if not path.is_file()]
        if missing_images:
            raise FileNotFoundError(f'Incomplete M&M1 phase images for {patient_id}: {missing_images}')
        rows.append({'patient_uid': _uid('mnm1', patient_id), 'dataset': 'mnm1', 'patient_id': patient_id, 'official_split': split, 'cine_format': 'frame_series', 'cine_path': str(patient_dir), 'disease': str(item['Pathology']).upper(), 'vendor': str(item['Vendor']), 'ed_index': frame_numbers.index(ed), 'es_index': frame_numbers.index(es), 'ed_image_path': str(ed_image), 'es_image_path': str(es_image), 'ed_label_path': str(ed_label) if has_labels else '', 'es_label_path': str(es_label) if has_labels else '', 'has_labels': has_labels, 'classification_eligible': patient_id in class_lookup, 'pretrain_eligible': split in {'train_labeled', 'train_unlabeled'}})
    frame = pd.DataFrame(rows, columns=PATIENT_COLUMNS)
    for patient_id, disease in class_lookup.items():
        frame.loc[frame['patient_id'] == patient_id, 'disease'] = disease
    return frame


def _phase_match(cine_path: Path, phase_path: Path) -> Tuple[int, float]:
    import nibabel as nib
    cine = np.asarray(nib.load(str(cine_path)).dataobj, dtype=np.float32)
    phase = np.asarray(nib.load(str(phase_path)).dataobj, dtype=np.float32)
    if cine.ndim != 4 or phase.ndim != 3 or cine.shape[:3] != phase.shape:
        raise ValueError(f'Cannot match {phase_path.name} to {cine_path.name}: {cine.shape}, {phase.shape}')
    stride = tuple((max(1, size // 64) for size in phase.shape))
    target = phase[::stride[0], ::stride[1], ::stride[2]].reshape(-1)
    target = target - target.mean()
    target_norm = float(np.linalg.norm(target))
    scores = []
    for index in range(cine.shape[3]):
        candidate = cine[::stride[0], ::stride[1], ::stride[2], index].reshape(-1)
        candidate = candidate - candidate.mean()
        denominator = float(np.linalg.norm(candidate)) * target_norm
        scores.append(float(np.dot(candidate, target) / denominator) if denominator > 0 else -1.0)
    best = int(np.argmax(scores))
    return (best, float(scores[best]))


def _require_nibabel():
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise ImportError('Install nibabel to load Cine-CMR NIfTI files') from exc
    return (nib, resample_from_to)


def _frame_number(path: Path) -> int:
    match = re.search('_frame(\\d+)\\.nii\\.gz$', path.name)
    if not match:
        raise ValueError(f'Cannot parse frame number from {path}')
    return int(match.group(1))


def load_raw_cine(row: Mapping[str, Any]):
    nib, _ = _require_nibabel()
    cine_format = str(row['cine_format'])
    if cine_format == 'nifti4d':
        image = nib.load(str(row['cine_path']))
        if len(image.shape) != 4:
            raise ValueError(f"Expected 4D Cine for {row['patient_uid']}, got {image.shape}")
        return (image, str(row['cine_path']))
    if cine_format != 'frame_series':
        raise ValueError(f'Unsupported cine_format {cine_format!r}')
    patient_dir = Path(str(row['cine_path']))
    frame_paths = sorted((path for path in patient_dir.glob(f"{row['patient_id']}_frame*.nii.gz") if not path.name.endswith('_gt.nii.gz')), key=_frame_number)
    if not frame_paths:
        raise FileNotFoundError(f'No M&M1 frames found under {patient_dir}')
    first = nib.as_closest_canonical(nib.load(str(frame_paths[0])))
    arrays = [np.asarray(first.dataobj, dtype=np.float32)]
    for path in frame_paths[1:]:
        frame = nib.as_closest_canonical(nib.load(str(path)))
        if frame.shape != first.shape or not np.allclose(frame.affine, first.affine, atol=0.0001):
            raise ValueError(f'Inconsistent frame geometry in {patient_dir}: {path.name}')
        arrays.append(np.asarray(frame.dataobj, dtype=np.float32))
    image = nib.Nifti1Image(np.stack(arrays, axis=3), first.affine, first.header)
    return (image, str(frame_paths[0]))


def _target_geometry(image, target_spacing_xy: float, size: int) -> Tuple[Tuple[int, int, int], np.ndarray]:
    nib, _ = _require_nibabel()
    canonical = nib.as_closest_canonical(image)
    spacing = nib.affines.voxel_sizes(canonical.affine)
    direction = canonical.affine[:3, :3] / spacing[None, :]
    resampled_xy = np.maximum(1, np.rint(np.asarray(canonical.shape[:2]) * spacing[:2] / float(target_spacing_xy)).astype(int))
    start_x = int((resampled_xy[0] - size) // 2)
    start_y = int((resampled_xy[1] - size) // 2)
    affine = canonical.affine.copy()
    affine[:3, 0] = direction[:, 0] * target_spacing_xy
    affine[:3, 1] = direction[:, 1] * target_spacing_xy
    origin = affine @ np.asarray([start_x, start_y, 0.0, 1.0])
    affine[:3, 3] = origin[:3]
    return ((size, size, int(canonical.shape[2])), affine)


def _resample_spatial(image, shape: Tuple[int, int, int], affine: np.ndarray, order: int) -> np.ndarray:
    nib, resample_from_to = _require_nibabel()
    canonical = nib.as_closest_canonical(image)
    if len(canonical.shape) == 3:
        result = resample_from_to(canonical, (shape, affine), order=order, cval=0.0)
        return np.asarray(result.dataobj, dtype=np.float32)
    frames = []
    for index in range(canonical.shape[3]):
        frame = nib.Nifti1Image(np.asarray(canonical.dataobj[..., index]), canonical.affine)
        result = resample_from_to(frame, (shape, affine), order=order, cval=0.0)
        frames.append(np.asarray(result.dataobj, dtype=np.float32))
    return np.stack(frames, axis=3)


def _periodic_resample(video: np.ndarray, ed_index: int, es_index: int, target_frames: int) -> Tuple[np.ndarray, float]:
    frames = video.shape[3]
    if not 0 <= ed_index < frames or not 0 <= es_index < frames:
        raise IndexError(f'ED/ES indices {(ed_index, es_index)} outside Cine with {frames} frames')
    order = (ed_index + np.arange(frames)) % frames
    shifted = video[..., order]
    positions = np.arange(target_frames, dtype=np.float32) * frames / float(target_frames)
    lower = np.floor(positions).astype(int) % frames
    upper = (lower + 1) % frames
    weight = positions - np.floor(positions)
    output = shifted[..., lower] * (1.0 - weight)[None, None, None, :]
    output += shifted[..., upper] * weight[None, None, None, :]
    es_shifted = (es_index - ed_index) % frames
    es_position = es_shifted * target_frames / float(frames)
    return (output.astype(np.float32), float(es_position))


def _normalize(video: np.ndarray) -> np.ndarray:
    if not np.isfinite(video).all():
        raise ValueError('Non-finite Cine intensities; check the source image')
    finite = video[np.isfinite(video)]
    if finite.size == 0:
        return np.zeros_like(video, dtype=np.float32)
    low, high = np.percentile(finite, (1.0, 99.0))
    if high <= low:
        return np.zeros_like(video, dtype=np.float32)
    result = np.clip((video - low) / (high - low), 0.0, 1.0)
    return result.astype(np.float32)


def discover(config, dataset):
    explicit = config['data'].get(dataset + '_root')
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_dir():
            raise FileNotFoundError(f'Missing extracted {dataset} directory: {path}')
        return path.resolve()
    root = Path(config['data']['root']).expanduser()
    names = {'acdc': {'acdc'}, 'mnm1': {'mnm1', 'm&m', 'm&m1', 'm&ms', 'mnms', 'mms'}, 'mnm2': {'mnm2', 'm&m2', 'm&ms2'}, 'brats2021': {'brats2021', 'brats_2021', 'brats'}}
    candidates = [p for p in [*root.glob('*'), *root.glob('*/*')] if p.is_dir() and p.name.lower() in names[dataset]]
    if len(candidates) != 1:
        raise FileNotFoundError(f'Expected one extracted {dataset} under {root}; found {candidates}. Set data.{dataset}_root explicitly.')
    return candidates[0].resolve()


def save_array(path, array):
    temp = path.with_suffix('.tmp.npy')
    np.save(temp, array, allow_pickle=False)
    temp.replace(path)


def log(message, level="INFO"):
    print(message, flush=True)


def progress(iterable=None, **kwargs):
    return tqdm(iterable, **kwargs)


def _read_mnm2_metadata(path):
    rows = []
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("SUBJECT_CODE", "")).strip().isdigit():
                rows.append(row)
    return pd.DataFrame(rows)


def scan_mnm2(root):
    root = Path(root).expanduser().resolve()
    metadata = _read_mnm2_metadata(_find_one(root, "dataset_information.csv"))
    lookup = {int(row["SUBJECT_CODE"]): row for _, row in metadata.iterrows()}
    candidates = [root] if root.name == "dataset" else sorted(p for p in root.rglob("dataset") if p.is_dir())
    if not candidates:
        raise FileNotFoundError("No MnM2 dataset directory under " + str(root))
    dataset_dir = candidates[0]
    rows = []
    for folder in progress(sorted(p for p in dataset_dir.iterdir() if p.is_dir()),
                           desc="scan MnM2 + ED/ES", unit="patient"):
        pid = folder.name
        code = int(pid)
        item = lookup[code]
        cine = folder / (pid + "_SA_CINE.nii.gz")
        ed_image = folder / (pid + "_SA_ED.nii.gz")
        es_image = folder / (pid + "_SA_ES.nii.gz")
        ed, _ = _phase_match(cine, ed_image)
        es, _ = _phase_match(cine, es_image)
        split = "train" if code <= 160 else "val" if code <= 200 else "test"
        disease = str(item["DISEASE"]).upper()
        rows.append(dict(
            patient_uid="mnm2:" + pid, patient_id=pid, dataset="mnm2",
            official_split=split, cine_format="nifti4d", cine_path=str(cine),
            disease=disease, vendor=str(item["VENDOR"]), ed_index=ed, es_index=es,
            ed_image_path=str(ed_image), es_image_path=str(es_image),
            ed_label_path=str(folder / (pid + "_SA_ED_gt.nii.gz")),
            es_label_path=str(folder / (pid + "_SA_ES_gt.nii.gz")),
            has_labels=(folder / (pid + "_SA_ED_gt.nii.gz")).is_file() and
                       (folder / (pid + "_SA_ES_gt.nii.gz")).is_file(),
            classification_eligible=disease in CLASS_NAMES["mnm2"],
            pretrain_eligible=split == "train"))
    return pd.DataFrame(rows, columns=PATIENT_COLUMNS)


def prepare_public(settings, overwrite=False):
    """Generate cine-only caches for all three datasets, including unlabeled/test cases."""
    from ..config import PUBLIC_DATASETS
    from ..utils import write_json
    root = Path(settings['public_cache_root']).expanduser().resolve()
    manifest = root / 'patients.csv'
    config = dict(data=dict(root=settings['public_raw_root']))
    for name, path in settings['public_raw_roots'].items():
        if path:
            config['data'][name + '_root'] = path
    if manifest.is_file() and not overwrite:
        frame = pd.read_csv(manifest, dtype={'patient_uid': str, 'patient_id': str}).fillna('')
        for key in ('has_labels', 'classification_eligible', 'pretrain_eligible'):
            frame[key] = frame[key].astype(str).str.lower().isin(['true', '1', '1.0'])
        if set(frame.dataset) != set(PUBLIC_DATASETS):
            raise ValueError('Public manifest must contain ACDC, MnM1 and MnM2; regenerate with overwrite=True')
        if frame['cache_path'].map(lambda p: Path(p).is_file()).all():
            print('Reuse public patient manifest:', manifest)
            return frame
    else:
        scanners = dict(acdc=scan_acdc, mnm1=scan_mnm1, mnm2=scan_mnm2)
        frame = pd.concat([scanners[name](discover(config, name)) for name in PUBLIC_DATASETS], ignore_index=True)
    if frame.patient_uid.duplicated().any():
        raise ValueError('Duplicate public patient UID')
    cache_paths = []
    frames, size = settings['data']['num_frames'], settings['data']['input_size']
    options = settings['public_preprocess']
    for row in progress(frame.to_dict('records'), desc='Public cine caches', unit='patient'):
        directory = root / 'cine' / row['dataset'] / row['patient_id']
        path = directory / 'cine.npy'
        if overwrite or not path.is_file():
            directory.mkdir(parents=True, exist_ok=True)
            raw, _ = load_raw_cine(row)
            shape, affine = _target_geometry(raw, options['target_spacing_xy'], size)
            spatial = _resample_spatial(raw, shape, affine, 1)
            temporal, es = _periodic_resample(spatial, int(row['ed_index']), int(row['es_index']), frames)
            indices = np.clip(np.asarray(options['slice_offsets']) + shape[2] // 2, 0, shape[2] - 1)
            cine = _normalize(temporal).transpose(2, 3, 0, 1)[indices].astype(np.float16)
            write_json(directory / 'metadata.json', dict(
                patient_uid=row['patient_uid'], source=row['cine_path'], layout='STHW',
                shape=list(cine.shape), dtype=str(cine.dtype), slice_indices=indices.tolist(),
                es_position=es, processed_affine=affine.tolist(), intensity='p1p99_unit_v5'))
            save_array(path, cine)
        cache_paths.append(str(path))
    frame['cache_path'] = cache_paths
    frame['views'] = len(options['slice_offsets'])
    # Splits are provenance only: public pretraining does not filter by this column.
    frame['split'] = frame.official_split.replace({'train_labeled': 'train', 'train_unlabeled': 'train'})
    root.mkdir(parents=True, exist_ok=True)
    temporary = root / 'patients.csv.tmp'
    frame.to_csv(temporary, index=False)
    temporary.replace(manifest)
    print('Public patients:', frame.groupby('dataset').size().to_dict())
    return frame
