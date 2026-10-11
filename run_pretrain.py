"""Command-line equivalent of the short notebook."""
import argparse
import json
from pathlib import Path
from cmr_study import default_settings, prepare_data


def merge(target, changes):
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            merge(target[key], value)
        else:
            target[key] = value


def main():
    parser = argparse.ArgumentParser(description='Raw CMR preparation + MAE pretraining')
    parser.add_argument('--settings', type=Path, help='Optional JSON path/config overrides')
    parser.add_argument('--source', choices=['private', 'public'])
    parser.add_argument('--methods', nargs='+', choices=['videomae', 'heartmae', 'trackmae', 'cineqmae'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    settings = default_settings()
    if args.settings:
        merge(settings, json.loads(args.settings.read_text(encoding='utf-8')))
    if args.source:
        settings['pretrain_source'] = args.source
        if settings['run_name'] == 'cineq_paper_private' and args.source == 'public':
            settings['run_name'] = 'cineq_paper_' + args.source
    if args.methods:
        settings['methods'] = args.methods
    if args.resume:
        settings['resume'] = True
    from cmr_study.config import validate
    validate(settings)
    prepare_data(settings)
    if not args.prepare_only:
        from cmr_study import PretrainingExperiment
        experiment = PretrainingExperiment(settings)
        experiment.prepare_motion()
        experiment.pretrain()


if __name__ == '__main__':
    main()
