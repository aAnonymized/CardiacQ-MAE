"""Lightweight cohort/scanner/orchestration tests; no CUDA or real patient data."""
import ast
import csv
from copy import deepcopy
from contextlib import contextmanager, redirect_stdout
import io
import json
from pathlib import Path
import pickle
import shutil
import sys
import uuid
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cmr_study import default_settings
from cmr_study.config import validate
from cmr_study.data.prepare import (assign_private_splits, load_records,
                                    manifest_path, pretraining_cohorts, prepare_data, prepare_private)
from cmr_study.utils import write_json


@contextmanager
def temporary_directory():
    workspace = Path.cwd().resolve()
    directory = workspace / ('_qa_paper_' + uuid.uuid4().hex[:12])
    directory.mkdir()
    try:
        with redirect_stdout(io.StringIO()):
            yield str(directory)
    finally:
        resolved = directory.resolve()
        if resolved.parent != workspace or not resolved.name.startswith('_qa_paper_'):
            raise RuntimeError('Refusing to remove a fixture outside its workspace')
        shutil.rmtree(resolved)


def definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and (names is None or n.name in names)]
    tree.body = nodes
    exec(compile(tree, str(path), 'exec'), namespace)
    return namespace


def public_namespace():
    env = dict(np=np, pd=pd, Path=Path, csv=csv, json=json,
               re=__import__('re'), __name__='cmr_study.data.public', __package__='cmr_study.data')
    # Future annotations keep the original scanner type annotations inert.
    source = (ROOT / 'cmr_study/data/public.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Assign)) or
                 isinstance(n, ast.ImportFrom) and n.module == '__future__']
    exec(compile(tree, '<public fixtures>', 'exec'), env)
    env['tqdm'] = lambda iterable=None, **kwargs: iterable
    return env


def patient(dataset, pid, split='train', disease='NOR', **extra):
    return dict(patient_uid=dataset + ':' + str(pid), patient_id=str(pid), dataset=dataset,
                split=split, disease=disease, views=3, cache_path='/unused/cine.npy', **extra)


class CohortTests(unittest.TestCase):
    def test_exact_recipe(self):
        s = default_settings()
        validate(s)
        self.assertEqual(s['pretrain'], dict(validate=False, epochs=400, batch_size=32,
            accum_steps=1, lr=1e-4, min_lr=1e-6, warmup_epochs=10, weight_decay=.05,
            grad_clip=1., max_consecutive_amp_skips=8))
        self.assertEqual(s['shared_decoder'], dict(dim=512, depth=8, heads=16, dpcc_loss_weight=.2))
        self.assertFalse(s['amp'])
        self.assertFalse(s['allow_tf32'])
        self.assertFalse(s['weights']['add_baseline'])
        self.assertEqual(s['query']['jacobian_loss_weight'], .1)
        self.assertEqual(s['query']['query_fraction'], .1)
        self.assertEqual(s['data'], dict(num_frames=16, input_size=224, intensity_jitter=.1, horizontal_flip=.5))
        default_settings()['query']['query_fraction'] = .9
        self.assertEqual(default_settings()['query']['query_fraction'], .1)

    def test_public_all_three_all_splits_unlabeled_and_nor(self):
        s = default_settings()
        s['pretrain_source'] = 'public'
        rows = [patient(name, i, split, classification_eligible=False, pretrain_eligible=False,
                        has_labels=False) for name in ('acdc', 'mnm1', 'mnm2')
                for i, split in enumerate(('train', 'val', 'test'))]
        rows.append(patient('private', 99))
        train, monitor = pretraining_cohorts(rows, s)
        self.assertEqual(len(train), 9)
        self.assertEqual({r['dataset'] for r in train}, {'acdc', 'mnm1', 'mnm2'})
        self.assertEqual(sum(r['split'] == 'test' for r in train), 3)
        self.assertTrue(all(r['disease'] == 'NOR' for r in train))
        self.assertEqual(len(monitor), 3)
        with self.assertRaises(ValueError):
            pretraining_cohorts(rows[:6], s)

    def test_private_never_trains_val_test(self):
        rows = [patient('private', i, split) for i, split in enumerate(('train', 'val', 'test'))]
        rows.append(patient('acdc', 0))
        train, monitor = pretraining_cohorts(rows, default_settings())
        self.assertEqual([r['split'] for r in train], ['train'])
        self.assertEqual([r['split'] for r in monitor], ['val'])

    def test_private_original_split_rule_deterministic(self):
        rows = [patient('private', str(i), 'train', 'HCM' if i < 10 else 'NOR') for i in range(20)]
        rows += [patient('private', 'test', 'test')]
        one = assign_private_splits(deepcopy(rows), default_settings())
        two = assign_private_splits(deepcopy(rows), default_settings())
        self.assertEqual(one, two)
        self.assertEqual(sum(r['split'] == 'val' for r in one), 4)
        self.assertEqual(one[-1]['split'], 'test')

    def test_exact_historical_split_is_optional(self):
        rows = [patient('private', 'a'), patient('private', 'b'), patient('private', 'c', 'test')]
        with temporary_directory() as folder:
            previous = Path(folder) / 'manifest.json'
            previous.write_text(json.dumps(dict(records=[dict(r, split='val' if r['patient_id'] == 'a'
                                                            else r['split']) for r in rows])), encoding='utf-8')
            s = default_settings()
            s['private_previous_manifest'] = str(previous)
            assigned = assign_private_splits(rows, s)
            self.assertEqual([r['split'] for r in assigned], ['val', 'train', 'test'])

    def test_training_reads_only_selected_manifest(self):
        with temporary_directory() as folder:
            s = default_settings()
            s.update(pretrain_source='public', public_cache_root=folder, private_root='/absent/private')
            rows = [patient(name, i) for i, name in enumerate(('acdc', 'mnm1', 'mnm2'))]
            pd.DataFrame(rows).to_csv(Path(folder) / 'patients.csv', index=False)
            self.assertEqual(load_records(s), rows)
            self.assertEqual(manifest_path(s), Path(folder).resolve() / 'patients.csv')

    def test_private_prepare_publishes_split_and_cache_paths(self):
        with temporary_directory() as folder:
            s = default_settings()
            s.update(private_root=folder, private_cache='native')
            report = pd.DataFrame([dict(patient_id=str(i), disease='NOR', csv_split='train' if i < 10 else 'test',
                cache_path='/cache/' + str(i) + '.npy', sax_path='/raw/' + str(i) + '.nii.gz',
                status='created', reason='') for i in range(12)])
            report.loc[len(report)] = dict(patient_id='bad', disease='NOR', csv_split='train',
                cache_path='', sax_path='/raw/bad.nii.gz', status='failed', reason='bad data')
            native = SimpleNamespace(prepare_private=Mock(return_value=report))
            with patch.dict(sys.modules, {'cmr_study.data.native': native}):
                frame = prepare_private(s)
                again = prepare_private(s)
            self.assertEqual(len(frame), 12)
            self.assertEqual(frame.groupby('split').size().to_dict(), {'test': 2, 'train': 8, 'val': 2})
            self.assertEqual(len(again), 12)
            self.assertEqual(native.prepare_private.call_count, 1)
            self.assertTrue(manifest_path(s).is_file())

    def test_private_only_does_not_import_public_preprocessor(self):
        s = default_settings()
        frame = pd.DataFrame([patient('private', 1)])
        with patch('cmr_study.data.prepare.prepare_private', return_value=frame) as private:
            with patch.dict(sys.modules, {'cmr_study.data.public': None}):
                result = prepare_data(s)
        self.assertEqual(list(result), ['private'])
        private.assert_called_once_with(s, overwrite=False)


class PublicPreprocessingTests(unittest.TestCase):
    def test_periodic_ed_alignment_and_normalization(self):
        env = public_namespace()
        raw = np.arange(8, dtype=np.float32).reshape(1, 1, 1, 8)
        video, es = env['_periodic_resample'](raw, 2, 6, 16)
        self.assertEqual(video.shape[-1], 16)
        self.assertEqual(video[0, 0, 0, 0], 2)
        self.assertEqual(es, 8)
        normal = env['_normalize'](video)
        self.assertGreaterEqual(normal.min(), 0)
        self.assertLessEqual(normal.max(), 1)

    def test_acdc_scans_test_nor_without_labels(self):
        env = public_namespace()
        with temporary_directory() as folder:
            root = Path(folder)
            for split, pid in [('training', 'patient001'), ('testing', 'patient101')]:
                case = root / split / pid
                case.mkdir(parents=True)
                (case / 'Info.cfg').write_text('ED: 1\nES: 3\nGroup: NOR\n', encoding='utf-8')
                (case / (pid + '_4d.nii.gz')).touch()
            frame = env['scan_acdc'](root)
            self.assertEqual(len(frame), 2)
            self.assertEqual(set(frame.official_split), {'train', 'test'})
            self.assertFalse(frame.has_labels.any())
            self.assertTrue((frame.ed_index == 0).all())

    def test_mnm1_original_4d_without_curated_class_list(self):
        env = public_namespace()
        with temporary_directory() as folder:
            root = Path(folder)
            entries = [('Training/Labeled', 'A'), ('Training/Unlabeled', 'B'), ('Validation', 'C'), ('Testing', 'D')]
            for split, pid in entries:
                case = root / split / pid
                case.mkdir(parents=True)
                (case / (pid + '_sa.nii.gz')).touch()
            pd.DataFrame([{'External code': pid, 'ED': 0, 'ES': 3, 'Pathology': 'NOR', 'Vendor': 'A'}
                          for _, pid in entries]).to_csv(root / 'information_diagnosis_opendataset.csv', index=False)
            frame = env['scan_mnm1'](root)
            self.assertEqual(len(frame), 4)
            self.assertEqual(set(frame.cine_format), {'nifti4d'})
            self.assertIn('train_unlabeled', set(frame.official_split))

    def test_mnm1_legacy_frame_series(self):
        env = public_namespace()
        with temporary_directory() as folder:
            root = Path(folder)
            for name in ('Training/Labeled', 'Training/Unlabeled', 'Validation', 'Testing'):
                (root / name).mkdir(parents=True)
            case = root / 'Training/Labeled/ABC'
            case.mkdir()
            for index in range(4):
                (case / ('ABC_frame%02d.nii.gz' % index)).touch()
            pd.DataFrame([{'External code': 'ABC', 'ED': 0, 'ES': 3, 'Pathology': 'HCM', 'Vendor': 'B'}]).to_csv(
                root / 'information_diagnosis_opendataset.csv', index=False)
            frame = env['scan_mnm1'](root)
            self.assertEqual(frame.iloc[0].cine_format, 'frame_series')
            self.assertEqual(frame.iloc[0].es_index, 3)

    def test_mnm2_keeps_official_splits_without_labels(self):
        env = public_namespace()
        env['_phase_match'] = lambda cine, phase: (0 if '_ED' in phase.name else 3, 1.)
        with temporary_directory() as folder:
            root = Path(folder)
            for pid in ('001', '161', '201'):
                case = root / 'dataset' / pid
                case.mkdir(parents=True)
            pd.DataFrame([dict(SUBJECT_CODE=int(pid), DISEASE='NOR', VENDOR='A')
                          for pid in ('001', '161', '201')]).to_csv(root / 'dataset_information.csv', index=False)
            frame = env['scan_mnm2'](root)
            self.assertEqual(list(frame.official_split), ['train', 'val', 'test'])
            self.assertEqual(list(frame.patient_id), ['001', '161', '201'])

    def test_raw_public_cache_writes_all_patients_and_reuses(self):
        env = public_namespace()
        with temporary_directory() as folder:
            s = default_settings()
            s['public_cache_root'] = folder
            s['data']['input_size'] = 4
            rows = [dict(patient(name, '001', 'test'), official_split='test', cine_format='nifti4d',
                         cine_path='/fake/raw.nii.gz', ed_index=0, es_index=2, has_labels=False,
                         classification_eligible=False, pretrain_eligible=False)
                    for name in ('acdc', 'mnm1', 'mnm2')]
            env['discover'] = lambda config, name: name
            for name in ('acdc', 'mnm1', 'mnm2'):
                env['scan_' + name] = lambda root, name=name: pd.DataFrame([r for r in rows if r['dataset'] == name])
            env['load_raw_cine'] = Mock(return_value=('raw', 'source'))
            env['_target_geometry'] = lambda raw, spacing, size: ((4, 4, 5), np.eye(4))
            env['_resample_spatial'] = lambda *args: np.arange(4*4*5*4, dtype=np.float32).reshape(4, 4, 5, 4)
            frame = env['prepare_public'](s)
            self.assertEqual(len(frame), 3)
            for path in frame.cache_path:
                cine = np.load(path)
                self.assertEqual(cine.shape, (3, 16, 4, 4))
                self.assertEqual(cine.dtype, np.float16)
            self.assertTrue((Path(folder) / 'patients.csv').is_file())
            again = env['prepare_public'](s)
            self.assertEqual(len(again), 3)
            self.assertEqual(env['load_raw_cine'].call_count, 3)


class Value:
    def __init__(self, value=0):
        self.value = value
    def detach(self):
        return self
    def clone(self):
        return Value(self.value)
    def cpu(self):
        return self
    def norm(self):
        return Value(abs(self.value))
    def __sub__(self, other):
        return Value(self.value - other.value)
    def __float__(self):
        return float(self.value)


class FakeModel:
    def __init__(self, settings, method):
        self.parameter = Value()
        self.backbone = SimpleNamespace(state_dict=lambda: {'weight': Value(), 'decoder.weight': Value(), 'mask_token': Value()})
    def to(self, device):
        return self
    def reconstruction_parameter(self):
        return self.parameter
    def state_dict(self):
        return {'value': self.parameter.value}
    def load_state_dict(self, state, strict):
        assert strict is True
        self.parameter.value = state['value']


class FakeState:
    def state_dict(self):
        return {}
    def load_state_dict(self, state):
        pass


class FakeDataset:
    def __init__(self, rows, settings, method, training=False):
        self.rows = rows
    def __len__(self):
        return sum(r['views'] for r in self.rows)


class RunnerTests(unittest.TestCase):
    def environment(self, settings, rows):
        epochs = []
        def train(model, batches, optimizer, scaler, options, device, epoch, forward, directory):
            epochs.append(epoch)
            model.parameter.value += 1
            return dict(masked_mse=.5, amp_skipped=0)
        def save(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('wb') as handle:
                pickle.dump(value, handle)
        def load(path):
            with path.open('rb') as handle:
                return pickle.load(handle)
        cuda = SimpleNamespace(is_available=lambda: False)
        env = dict(__name__='cmr_study.pretrain', __package__='cmr_study', deepcopy=deepcopy,
                   Path=Path, pd=pd, tqdm=lambda items, **kw: items, validate=validate,
                   torch=SimpleNamespace(cuda=cuda, backends=SimpleNamespace(
                       cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                       cudnn=SimpleNamespace(allow_tf32=False))),
                   engine=SimpleNamespace(device_for=lambda s: SimpleNamespace(type='cpu'),
                       optimizer_for=lambda *a: FakeState(), make_scaler=lambda *a: FakeState(),
                       loader=lambda dataset, *a: dataset, epoch_train=train, pretrain_forward=None),
                   load_records=lambda s: rows, pretraining_cohorts=pretraining_cohorts,
                   seed_all=Mock(), restore_rng=Mock(), rng_state=lambda: {},
                   write_json=write_json, save_checkpoint=save, load_checkpoint=load,
                   StudyModel=FakeModel, CineDataset=FakeDataset,
                   providers=SimpleNamespace(prepare=Mock(), _TRACKERS={}),
                   motion_kind=lambda m: {'cineqmae': 'dis', 'heartmae': 'heart', 'trackmae': 'cotracker'}.get(m))
        definitions(ROOT / 'cmr_study/pretrain.py', env)
        return env, epochs, load

    def test_last_checkpoint_every_epoch_and_direct_resume(self):
        with temporary_directory() as folder:
            s = default_settings()
            s.update(output_root=folder, methods=['videomae'])
            s['pretrain']['epochs'] = 2
            s['snapshots']['enabled'] = False
            env, epochs, load = self.environment(s, [patient('private', 1), patient('private', 2, 'val')])
            exp = env['PretrainingExperiment'](s)
            result = exp.pretrain()
            checkpoint = load(Path(result.iloc[0].checkpoint))
            self.assertEqual(checkpoint['epoch'], 2)
            self.assertEqual(checkpoint['model']['value'], 2)
            self.assertEqual(len(checkpoint['history']), 2)
            self.assertNotIn('signature', checkpoint)
            self.assertEqual(epochs, [0, 1])
            with self.assertRaises(FileExistsError):
                exp.pretrain()
            s['resume'] = True
            s['pretrain']['epochs'] = 3
            # A settings change is not blocked by a protocol signature.
            s['num_workers'] = 0
            result = env['PretrainingExperiment'](s).pretrain()
            self.assertEqual(epochs, [0, 1, 2])
            self.assertEqual(load(Path(result.iloc[0].checkpoint))['model']['value'], 3)
            self.assertEqual(env['restore_rng'].call_count, 1)
            encoder = load(Path(result.iloc[0].checkpoint).with_name('encoder_last.pt'))['encoder']
            self.assertEqual(set(encoder), {'weight'})

    def test_motion_only_selected_source_and_needed_teachers(self):
        with temporary_directory() as folder:
            s = default_settings()
            s.update(output_root=folder, pretrain_source='public', methods=['cineqmae', 'videomae', 'heartmae'])
            s['snapshots']['enabled'] = False
            rows = [patient(name, 0, 'test') for name in ('acdc', 'mnm1', 'mnm2')]
            rows.append(patient('private', 0))
            env, _, _ = self.environment(s, rows)
            exp = env['PretrainingExperiment'](s)
            exp.prepare_motion()
            calls = env['providers'].prepare.call_args_list
            self.assertEqual(len(calls), 18)
            self.assertEqual({call.args[2] for call in calls}, {'dis', 'heart'})
            self.assertTrue(all(call.args[0]['dataset'] != 'private' for call in calls))

    def test_snapshot_schedule_kept(self):
        with temporary_directory() as folder:
            s = default_settings()
            s.update(output_root=folder, methods=['videomae'])
            s['pretrain']['epochs'] = 51
            env, _, _ = self.environment(s, [patient('private', 0), patient('private', 1, 'val')])
            refs = SimpleNamespace(require_references=Mock())
            with patch.dict(sys.modules, {'cmr_study.visualization.references': refs}):
                exp = env['PretrainingExperiment'](s)
                exp._snapshot = Mock()
                exp.pretrain()
            self.assertEqual(exp._snapshot.call_count, 1)
            self.assertEqual(exp._snapshot.call_args.args[1]['epoch'], 50)

    def test_each_method_and_seed_has_its_own_last(self):
        with temporary_directory() as folder:
            s = default_settings()
            s.update(output_root=folder, methods=['videomae', 'heartmae', 'trackmae', 'cineqmae'], seeds=[0, 1])
            s['pretrain']['epochs'] = 1
            s['snapshots']['enabled'] = False
            env, epochs, load = self.environment(s, [patient('private', 0)])
            exp = env['PretrainingExperiment'](s)
            result = exp.pretrain()
            self.assertEqual(len(result), 8)
            self.assertEqual(len(set(result.checkpoint)), 8)
            for row in result.to_dict('records'):
                saved = load(Path(row['checkpoint']))
                self.assertEqual((saved['method'], saved['seed'], saved['epoch']), (row['method'], row['seed'], 1))


class ArtifactTests(unittest.TestCase):
    def test_python38_sources_and_notebook_cells_parse(self):
        for path in ROOT.rglob('*.py'):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path), feature_version=(3, 8))
        notebook = json.loads((ROOT / 'Pretraining.ipynb').read_text(encoding='utf-8'))
        self.assertEqual(notebook['nbformat'], 4)
        code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
        self.assertEqual(len(code), 4)
        for cell in code:
            ast.parse(''.join(cell['source']), feature_version=(3, 8))
            self.assertEqual(cell['outputs'], [])
        source = '\n'.join(''.join(c['source']) for c in code)
        self.assertNotIn('v5_root', source)
        self.assertNotIn('finetune', source)
        self.assertIn('experiment.pretrain()', source)


if __name__ == '__main__':
    unittest.main()
