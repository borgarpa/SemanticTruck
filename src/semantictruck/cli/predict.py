"""Run segmentation inference over the truckloads, recording what each one cost and scored.

    semantictruck-predict --campaign results/segmentation
    semantictruck-predict --campaign results/segmentation --sets val --save-pred results/preds
    semantictruck-predict --campaign results/segmentation --grid-mode test --repeats 5
    semantictruck-predict --campaign results/segmentation --grid-mode test --check

Reads a finished `semantictruck-train` campaign and re-runs inference at each fold's checkpoint,
writing **one row per truckload** -- its inference time, loss, point-wise accuracy and mIoU. A
training campaign records only whole-set aggregates, so this is where a per-load number comes from.

Needs Pointcept and a CUDA device, in the training environment. See `docs/reproducing.md`.

Three things it is for:

    predictions   `--save-pred DIR` writes each held-out load's per-point labels in the layout and
                  the 0 = wood convention `semantictruck-pipeline --pred-dir` reads
    timing        `--repeats N` passes over each set N times, and `--timing-csv` writes the
                  per-load cost in the shape `semantictruck-timing --external-timings` joins on
    verification  `--check` compares this run against what the campaign recorded

**`--sets val` is usually what you want.** Across all folds it is exactly the out-of-fold
evaluations -- each truckload segmented once, by the model that held it out -- which is what a
deployment costs. Adding `train` times every load in every fold, which is several times more
inference than the method needs.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from os.path import join

from semantictruck import config
from semantictruck.training import backend

DEFAULT_CAMPAIGN = 'results/segmentation'


def check(args, summary: list, rows: list, log) -> bool:
    """Compare against whichever reference the active grid mode is supposed to reproduce.

    Repeats are a timing device, so only the first pass is checked: the later ones carry the same
    metrics and no stored-prediction comparison at all.
    """
    from semantictruck.training.inference import read_csv

    log('')
    log('consistency check ({} mode)'.format(args.grid_mode))
    log('-' * 78)

    if args.repeats > 1:
        log('  {} passes were run; the first is what is checked.'.format(args.repeats))
        summary = [e for e in summary if e.get('rep') == 0]
        rows = [r for r in rows if r['rep'] == 0]
    if args.limit is not None:
        log('  --limit was given, so the pooled figures cover part of each set only; the')
        log('  comparisons below are indicative.')

    ok = True
    if args.grid_mode == 'train':
        per_epoch = read_csv(join(args.campaign, 'metrics_per_epoch.csv'))
        last = {}
        for row in per_epoch:
            fold = int(row['fold'])
            if fold not in last or int(row['epoch']) > int(last[fold]['epoch']):
                last[fold] = row
        if not last:
            log('  no metrics_per_epoch.csv to compare against')
            return True

        log('  {:<5} {:<6} {:<8} {:>10} {:>10} {:>10}'.format(
            'fold', 'set', 'metric', 'here', 'campaign', 'delta'))
        for entry in summary:
            reference = last.get(entry['fold'])
            if entry['fold'] == 'all' or reference is None:
                continue
            prefix = 'eval_{}_'.format('val' if entry['set'] == 'val' else 'train')
            for label, mine, key in (('miou', entry['pooled_miou'], prefix + 'miou'),
                                     ('acc', entry['pooled_acc'], prefix + 'acc'),
                                     ('loss', entry['mean_loss'], prefix + 'loss')):
                if key not in reference:
                    continue
                published = float(reference[key])
                delta = mine - published
                within = abs(delta) <= args.tol
                ok = ok and (within or args.limit is not None)
                log('  {:<5} {:<6} {:<8} {:>10.5f} {:>10.5f} {:>+10.5f}{}'.format(
                    entry['fold'], entry['set'], label, mine, published, delta,
                    '' if within else '   <-- exceeds --tol'))
        log('')
        log('  Residuals around 1e-3 are voxel-representative jitter: the reference numbers came')
        log('  from one particular random draw and that state is not recoverable. Larger is a')
        log('  finding.')
    else:
        scored = [r for r in rows if r['set'] == 'val' and r['saved_miou'] != '']
        if not scored:
            log('  no stored predictions to compare against')
            return True

        log('  {:<5} {:<16} {:>10} {:>10} {:>10} {:>8}'.format(
            'fold', 'load', 'd_mIoU', 'd_acc', 'differ %', 'verdict'))
        worst = 0.0
        for row in scored:
            d_miou = row['miou'] - row['saved_miou']
            d_acc = row['acc'] - row['saved_acc']
            worst = max(worst, abs(d_miou), abs(d_acc))
            within = max(abs(d_miou), abs(d_acc)) <= args.tol
            ok = ok and within
            log('  {:<5} {:<16} {:>+10.5f} {:>+10.5f} {:>10.3f} {:>8}'.format(
                row['fold'], row['load_id'], d_miou, d_acc, row['saved_disagree_pct'],
                'ok' if within else 'EXCEEDS'))
        log('')
        log('  largest |delta| {:.6f} against --tol {}'.format(worst, args.tol))
        log('  Tenths of a percent of points differing is expected -- the forward is not')
        log('  bit-reproducible here. The metrics are what must agree.')

    log('-' * 78)
    return ok


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='semantictruck-predict', description=__doc__.split('\n\n')[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--campaign', default=DEFAULT_CAMPAIGN,
                        help='a finished semantictruck-train campaign directory')
    parser.add_argument('--out', default=None, help='where the CSVs go (default: --campaign)')
    parser.add_argument('--pointcept-root', default=None,
                        help='the Pointcept checkout (default: ${})'.format(backend.ENV_VAR))
    parser.add_argument('--folds-file', default=None,
                        help='the fold manifest (default: the one the campaign recorded)')
    parser.add_argument('--folds', nargs='+', type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument('--sets', nargs='+', default=['train', 'val'], choices=['train', 'val'],
                        help='which loads to run: val is the out-of-fold set, and is what a '
                             'deployment costs')
    parser.add_argument('--grid-mode', default='train', choices=['train', 'test'],
                        help='train replicates the per-epoch evaluator, test the path that wrote '
                             'the stored predictions -- see the module docstring; do not pool them')
    parser.add_argument('--checkpoint', default='model_last.pth',
                        help='never model_best: the validation fold is the held-out fold')

    parser.add_argument('--save-pred', nargs='?', const='auto', default=None, metavar='DIR',
                        help='write each held-out load\'s per-point labels; with no directory, a '
                             'temporary one is made and printed')
    parser.add_argument('--timing-csv', nargs='?', const='auto', default=None, metavar='PATH',
                        help='write the per-load cost for semantictruck-timing '
                             '--external-timings')
    parser.add_argument('--repeats', type=int, default=1,
                        help='passes over each set, for a timing study')
    parser.add_argument('--warmup', type=int, default=1,
                        help='discarded forwards before timing, so the first load does not carry '
                             'the kernel-selection cost')
    parser.add_argument('--limit', type=int, default=None, help='first N loads of each set only')
    parser.add_argument('--seed', type=int, default=None,
                        help='override the seed the campaign recorded')

    parser.add_argument('--check', action='store_true',
                        help='compare against what the campaign recorded, and exit non-zero on a '
                             'disagreement beyond --tol')
    parser.add_argument('--tol', type=float, default=2e-3)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # Pointcept first: importing the inference module needs it, and so does the dataset registry
    args.pointcept_root = backend.pointcept_root(args.pointcept_root)
    backend.ensure_importable(args.pointcept_root)
    import semantictruck.training.dataset                          # noqa: F401  registers it
    from pointcept.utils.config import Config
    from semantictruck.training import inference

    if args.repeats < 1:
        raise SystemExit('--repeats must be at least 1')
    if not os.path.isdir(args.campaign):
        raise SystemExit('no campaign at {} -- run semantictruck-train first'.format(args.campaign))

    args.out = args.out or args.campaign
    suffix = '_test' if args.grid_mode == 'test' else ''
    if args.save_pred == 'auto':
        args.save_pred = tempfile.mkdtemp(prefix='semantictruck_pred_')
    if args.save_pred and 'val' not in args.sets:
        raise SystemExit('--save-pred writes the held-out set only; add --sets val')
    if args.timing_csv == 'auto':
        args.timing_csv = join(args.out, 'inference_timing{}.csv'.format(suffix))

    import torch
    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device; this times GPU inference and will not run on CPU')

    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    first_config = join(args.campaign, 'fold_{}'.format(args.folds[0]), 'config.py')
    folds_file = args.folds_file or Config.fromfile(first_config).data.test.fold_file
    folds = inference.load_folds(folds_file)

    log('campaign    : {}'.format(args.campaign))
    log('folds file  : {}'.format(folds_file))
    log('grid mode   : {}'.format(args.grid_mode))
    log('checkpoint  : {}'.format(args.checkpoint))
    if args.save_pred:
        log('predictions : {}'.format(args.save_pred))
    if args.repeats > 1:
        log('passes      : {}'.format(args.repeats))
        log('')
        log('Timing study: nothing else should be using the GPU while this runs.')
    log('')

    started = time.perf_counter()
    rows = []
    for fold in args.folds:
        rows += inference.run_fold(args, fold, folds, log)
    log('\n{} row(s) in {:.1f} min'.format(len(rows), (time.perf_counter() - started) / 60))

    with_rep = args.repeats > 1
    summary = inference.summarise(rows, args.sets, args.grid_mode, with_rep)

    per_sample = join(args.out, 'inference_per_sample{}.csv'.format(suffix))
    inference.write_csv(per_sample, rows, inference.per_sample_columns(with_rep))
    summary_path = join(args.out, 'inference_summary{}.csv'.format(suffix))
    inference.write_csv(summary_path, summary, inference.summary_columns(with_rep))
    log('\n{}\n{}'.format(per_sample, summary_path))

    if args.timing_csv:
        inference.write_timing_csv(args.timing_csv, rows)
        log('{}  (semantictruck-timing --external-timings)'.format(args.timing_csv))

    ok = check(args, summary, rows, log) if args.check else True

    with open(join(args.out, 'inference{}.log'.format(suffix)), 'w') as handle:
        handle.write('\n'.join(lines) + '\n')

    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
