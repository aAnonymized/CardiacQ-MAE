"""Publish patient manifests once; training reads them without rescanning cine arrays."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..config import PUBLIC_DATASETS


def private_cache_root(settings):
    path = Path(settings['private_cache']).expanduser()
    return path.resolve() if path.is_absolute() else (Path(settings['private_root']).expanduser() / path).resolve()


def manifest_path(settings, source=None):
    source = source or settings['pretrain_source']
    if source == 'private':
        return private_cache_root(settings) / 'patients.csv'
    if source == 'public':
        return Path(settings['public_cache_root']).expanduser().resolve() / 'patients.csv'
    raise ValueError("source must be 'private' or 'public'")


def assign_private_splits(rows, settings):
    previous = settings.get('private_previous_manifest')
    if previous:
        previous = Path(previous).expanduser()
        if not previous.is_absolute():
            previous = Path(settings['private_root']).expanduser() / previous
        prior = {str(r['patient_id']): r['split'] for r in json.loads(previous.read_text(encoding='utf-8'))['records']
                 if r.get('dataset', 'private') == 'private'}
        for row in rows:
            split = prior.get(row['patient_id'])
            if split is None:
                raise ValueError('Patient absent from previous manifest: ' + row['patient_id'])
            if (row['split'] == 'test') != (split == 'test'):
                raise ValueError('Train/test membership differs from previous manifest: ' + row['patient_id'])
            row['split'] = split
        return rows
    rng = np.random.RandomState(settings['split_seed'])
    for disease in sorted({r['disease'] for r in rows if r['split'] == 'train'}):
        indices = [i for i, r in enumerate(rows) if r['split'] == 'train' and r['disease'] == disease]
        rng.shuffle(indices)
        if len(indices) < 2:
            raise ValueError('Need at least two private training patients in ' + disease)
        nval = min(len(indices) - 1, max(1, round(len(indices) * settings['val_fraction'])))
        for index in indices[:nval]:
            rows[index]['split'] = 'val'
    return rows


def prepare_private(settings, overwrite=False):
    from .native import prepare_private as native_prepare
    path = manifest_path(settings, 'private')
    if path.is_file() and not overwrite:
        print('Reuse private patient manifest:', path)
        return pd.read_csv(path, dtype={'patient_uid': str, 'patient_id': str}).fillna('')
    report = native_prepare(settings)
    rows = []
    for row in report[report.status != 'failed'].to_dict('records'):
        disease = settings['label_map'].get(row['disease'], row['disease'])
        rows.append(dict(patient_uid='private:' + row['patient_id'], patient_id=row['patient_id'],
                         dataset='private', disease=disease, split=row['csv_split'],
                         official_split=row['csv_split'], cache_path=row['cache_path'],
                         views=len(settings['native_preprocess']['slice_offsets']),
                         cine_path=row['sax_path']))
    frame = pd.DataFrame(assign_private_splits(rows, settings))
    if frame.empty:
        raise ValueError('No usable private cine cases; see native_prepare_report.csv')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.csv.tmp')
    frame.to_csv(temporary, index=False)
    temporary.replace(path)
    return frame


def prepare_data(settings, sources=None, overwrite=False):
    """By default prepare only the selected source; pass both to build both caches."""
    sources = [settings['pretrain_source']] if sources is None else sources
    if isinstance(sources, str):
        sources = [sources]
    result = {}
    for source in sources:
        if source not in ('private', 'public'):
            raise ValueError("sources must contain 'private' or 'public'")
        if source == 'private':
            prepare = prepare_private
        else:
            from .public import prepare_public
            prepare = prepare_public
        result[source] = prepare(settings, overwrite=overwrite)
    return result


def load_records(settings):
    path = manifest_path(settings)
    if not path.is_file():
        raise FileNotFoundError('Run prepare_data(settings) first: ' + str(path))
    frame = pd.read_csv(path, dtype={'patient_uid': str, 'patient_id': str}).fillna('')
    frame['views'] = frame['views'].astype(int)
    if frame.patient_uid.duplicated().any():
        raise ValueError('Repeated patient_uid in ' + str(path))
    return frame.to_dict('records')


def pretraining_cohorts(records, settings):
    source = settings['pretrain_source']
    if source == 'private':
        selected = [r for r in records if r['dataset'] == 'private']
        train = [r for r in selected if r['split'] == 'train']
        monitor = [r for r in selected if r['split'] == 'val']
    elif source == 'public':
        selected = [r for r in records if r['dataset'] in PUBLIC_DATASETS]
        if {r['dataset'] for r in selected} != set(PUBLIC_DATASETS):
            raise ValueError('Public pretraining requires ACDC, MnM1 and MnM2')
        # Deliberately retain all splits, NOR and classification-ineligible/unlabeled cases.
        train = selected
        monitor = [r for r in selected if r['split'] == 'val'] or selected
    else:
        raise ValueError("pretrain_source must be 'private' or 'public'")
    if not train:
        raise ValueError('Empty pretraining cohort: ' + source)
    return train, monitor
