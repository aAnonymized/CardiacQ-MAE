"""Source-selectable pretraining with ordinary checkpoint resume, without protocol gates."""
from copy import deepcopy
from pathlib import Path
import pandas as pd
import torch
from tqdm import tqdm
from .config import validate
from .data.prepare import load_records, pretraining_cohorts
from .data.dataset import CineDataset, motion_kind
from .models.study import StudyModel
from .motion import providers
from .training import engine
from .utils import seed_all, write_json, save_checkpoint, load_checkpoint, rng_state, restore_rng


def encoder_state(model):
    return {key: value.detach().cpu() for key, value in model.backbone.state_dict().items()
            if not key.startswith('decoder') and key != 'mask_token'}


class PretrainingExperiment:
    def __init__(self, settings):
        self.settings = deepcopy(settings)
        validate(self.settings)
        s = self.settings
        torch.backends.cuda.matmul.allow_tf32 = s['allow_tf32']
        torch.backends.cudnn.allow_tf32 = s['allow_tf32']
        self.device = engine.device_for(s)
        self.records = load_records(s)
        self.train_rows, self.monitor_rows = pretraining_cohorts(self.records, s)
        self.root = Path(s['output_root']).expanduser().resolve() / s['run_name']
        self.root.mkdir(parents=True, exist_ok=True)
        write_json(self.root / 'settings_latest.json', s)
        pd.DataFrame(self.train_rows).to_csv(self.root / 'pretrain_patients.csv', index=False)
        self.counts().to_csv(self.root / 'patient_counts.csv', index=False)
        self.protocol = 'public_full_transductive' if s['pretrain_source'] == 'public' else 'private_train_only'
        write_json(self.root / 'data_protocol.json', dict(
            pretrain_source=s['pretrain_source'], protocol=self.protocol,
            patients=len(self.train_rows), views=sum(r['views'] for r in self.train_rows),
            datasets=sorted({r['dataset'] for r in self.train_rows}),
            public_test_images_in_pretraining=s['pretrain_source'] == 'public',
            nor_retained=True, monitor_is_held_out=s['pretrain_source'] == 'private'))
        print('Output:', self.root, '| device:', self.device, '| source:', s['pretrain_source'])
        print('Pretraining patients:', len(self.train_rows), '| cine views:', sum(r['views'] for r in self.train_rows))
        if s['pretrain_source'] == 'public':
            print('Public FULL pool includes original validation/test images; monitor images are training diagnostics.')

    def counts(self):
        return (pd.DataFrame(self.train_rows).groupby(['dataset', 'split', 'disease'], dropna=False)
                .size().rename('patients').reset_index())

    def checkpoint_dir(self, method, seed):
        return self.root / 'pretrain' / method / ('seed%d' % seed)

    @property
    def snapshots_enabled(self):
        return self.settings['snapshots'].get('enabled', True) and bool(self.monitor_rows)

    def prepare_motion(self, methods=None):
        s = self.settings
        methods = list(methods if methods is not None else s['methods'])
        kinds = sorted({motion_kind(method) for method in methods if motion_kind(method)})
        if 'cotracker' in kinds:
            repo = Path(s['motion']['cotracker_repo']).expanduser()
            weight = Path(s['motion']['cotracker_checkpoint']).expanduser()
            if not (repo / 'cotracker/predictor.py').is_file() or not weight.is_file():
                raise FileNotFoundError('TrackMAE needs offline CoTracker source/weights; set motion.cotracker_repo and motion.cotracker_checkpoint')
        rows = {r['patient_uid']: r for r in self.train_rows}
        if s['pretrain'].get('validate', False):
            rows.update({r['patient_uid']: r for r in self.monitor_rows})
        for kind in kinds:
            for row in tqdm(rows.values(), desc='Motion cache: ' + kind):
                for view in range(row['views']):
                    providers.prepare(row, view, kind, s)
        providers._TRACKERS.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if self.snapshots_enabled:
            from .visualization.references import prepare_references
            timings = prepare_references(self.monitor_rows, s=s)
            pd.DataFrame(timings).to_csv(self.root / 'snapshot_reference_prepare_timings.csv', index=False)
        print('Motion caches ready; training loads cached teachers only.')

    def _snapshot(self, model, checkpoint, directory):
        from .visualization.export import export_cases
        epoch, seed = checkpoint['epoch'], checkpoint['seed']
        path = directory / ('epoch%04d.pt' % epoch)
        save_checkpoint(path, {k: checkpoint[k] for k in ('settings', 'method', 'seed', 'epoch', 'model')})
        root = directory / 'snapshots' / ('epoch%04d' % epoch)
        write_json(root / 'checkpoint.json', dict(checkpoint=str(path), epoch=epoch,
                   method=checkpoint['method'], seed=seed, pretrain_source=self.settings['pretrain_source'],
                   monitor_is_held_out=self.settings['pretrain_source'] == 'private'))
        export_cases(model, self.monitor_rows, self.settings, seed, root, self.device, epoch)

    def pretrain(self, methods=None, stop_after_epoch=None):
        s = self.settings
        methods = list(methods if methods is not None else s['methods'])
        chosen = dict(s, methods=methods)
        validate(chosen)
        if self.snapshots_enabled:
            from .visualization.references import require_references
            require_references(self.monitor_rows, s)
        validate_pretrain = s['pretrain'].get('validate', False)
        if validate_pretrain and not self.monitor_rows:
            raise ValueError('No reconstruction-monitor patients available')
        end = s['pretrain']['epochs']
        if stop_after_epoch is not None:
            end = min(end, int(stop_after_epoch))
        if end < 1:
            raise ValueError('stop_after_epoch must be at least 1')
        jobs = []
        for seed in s['seeds']:
            for method in methods:
                directory = self.checkpoint_dir(method, seed)
                directory.mkdir(parents=True, exist_ok=True)
                last = directory / 'last.pt'
                if last.is_file() and not s['resume']:
                    raise FileExistsError('Checkpoint already exists: ' + str(last) + '; set resume=True or use a new run_name')
                seed_all(seed)
                model = StudyModel(s, method).to(self.device)
                optimizer = engine.optimizer_for(model, s['pretrain'])
                scaler = engine.make_scaler(s['amp'] and self.device.type == 'cuda')
                start, history, best = 0, [], float('inf')
                if last.is_file():
                    checkpoint = load_checkpoint(last)
                    model.load_state_dict(checkpoint['model'], strict=True)
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scaler.load_state_dict(checkpoint['scaler'])
                    start, history = checkpoint['epoch'], checkpoint['history']
                    best = checkpoint.get('best', float('inf'))
                    restore_rng(checkpoint['rng'])
                    print('Resume:', last, '| completed epochs:', start)
                    saved_source = checkpoint.get('pretrain_source')
                    if saved_source is not None and saved_source != s['pretrain_source']:
                        print('WARNING: continuing a checkpoint from a different pretraining source:', saved_source)
                    if self.snapshots_enabled and start % s['snapshots']['every_epochs'] == 0:
                        self._snapshot(model, checkpoint, directory)
                    del checkpoint
                train = CineDataset(self.train_rows, s, method, training=True)
                val = CineDataset(self.monitor_rows, s, method) if validate_pretrain else None
                batches = engine.loader(train, s, 'pretrain')
                for epoch in range(start, end):
                    before = model.reconstruction_parameter().detach().clone()
                    stats = engine.epoch_train(model, batches, optimizer, scaler, s['pretrain'], self.device,
                                               epoch, engine.pretrain_forward, directory)
                    stats['decoder_update_l2'] = float((model.reconstruction_parameter() - before).norm().detach())
                    del before
                    val_stats, cases = {}, None
                    if validate_pretrain:
                        val_stats, cases = engine.evaluate_reconstruction(model, val, seed, self.device, epoch)
                    history.append(dict(epoch=epoch + 1,
                                        **{'train_' + k: v for k, v in stats.items()},
                                        **{'val_' + k: v for k, v in val_stats.items()}))
                    improved = validate_pretrain and val_stats['masked_mse'] < best
                    if improved:
                        best = val_stats['masked_mse']
                    checkpoint = dict(settings=s, method=method, seed=seed, epoch=epoch + 1,
                                      pretrain_source=s['pretrain_source'], data_protocol=self.protocol,
                                      model=model.state_dict(), optimizer=optimizer.state_dict(),
                                      scaler=scaler.state_dict(), rng=rng_state(), history=history, best=best)
                    save_checkpoint(last, checkpoint)
                    save_checkpoint(directory / 'encoder_last.pt', dict(
                        encoder=encoder_state(model), settings=s, method=method, seed=seed,
                        epoch=epoch + 1, source_checkpoint='last.pt'))
                    if improved:
                        save_checkpoint(directory / 'best.pt', {k: checkpoint[k] for k in (
                            'settings', 'method', 'seed', 'epoch', 'model', 'pretrain_source')})
                        cases.to_csv(directory / 'best_validation_patients_views.csv', index=False)
                    pd.DataFrame(history).to_csv(directory / 'history.csv', index=False)
                    print('%s seed %d epoch %d/%d | train masked MSE %.6f | decoder update %.6f' % (
                        method, seed, epoch + 1, s['pretrain']['epochs'], stats['masked_mse'], stats['decoder_update_l2']))
                    if self.snapshots_enabled and (epoch + 1) % s['snapshots']['every_epochs'] == 0:
                        self._snapshot(model, checkpoint, directory)
                    del checkpoint
                result = dict(method=method, seed=seed, epoch=history[-1]['epoch'] if history else start,
                              pretrain_source=s['pretrain_source'], patients=len(self.train_rows),
                              cine_views=len(train), checkpoint=str(last))
                write_json(directory / 'summary.json', result)
                jobs.append(result)
                del model, optimizer, scaler, batches, train, val
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        frame = pd.DataFrame(jobs)
        frame.to_csv(self.root / 'pretrain_results.csv', index=False)
        return frame
