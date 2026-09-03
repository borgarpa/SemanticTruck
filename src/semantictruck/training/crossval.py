"""Five-fold cross-validation of the segmentation model over the truckloads.

The campaign the manuscript's §2.3 describes: five folds, each training on 24 truckloads and
predicting the 6 it never saw, pooling one out-of-fold prediction for every load. Those pooled
predictions are what the geometry stage then consumes as its `predicted` arm, which is the only
way the two halves of the paper connect.

Three decisions in here are not incidental:

* **The fold assignment is read, never regenerated.** It comes from a manifest, so the same
  truckloads are held out on every re-run and by any other model trained for comparison. A campaign
  that reshuffled its folds could not be compared with anything, including itself.
* **Inference uses the last checkpoint, never the best.** The validation fold *is* the held-out
  fold, so choosing a checkpoint by its score would be selecting on the test set.
* **A finished fold is skipped rather than retrained.** Training all five takes many hours on one
  card; without this an interrupted campaign could not be resumed. `--force` overrides it.

Each fold runs as a fresh subprocess so it gets a clean CUDA context, which on a small card decides
whether the next fold fits at all.
"""

from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import time
from os.path import basename, exists, join

import numpy as np

from semantictruck.training import CLASS_NAMES, backend

METRICS_PER_EPOCH_CSV = 'metrics_per_epoch.csv'
METRICS_SUMMARY_CSV = 'metrics_summary.csv'
PREDICTIONS_DIR = 'predictions'


# ----------------------------------------------------------------------------------- the folds

def load_folds(fold_file: str, n_folds: int) -> tuple[dict, dict]:
    """The fold manifest, checked: every fold present, and no truckload in two of them."""
    if not exists(fold_file):
        raise SystemExit('no folds manifest at {}. It maps each fold to the truckloads held out '
                         'in it; see docs/reproducing.md'.format(fold_file))

    with open(fold_file) as handle:
        manifest = json.load(handle)
    folds = manifest.get('folds', manifest)

    keys = [str(i) for i in range(n_folds)]
    missing = [k for k in keys if k not in folds]
    if missing:
        raise SystemExit('{} has no fold(s) {}'.format(fold_file, missing))

    names = [n for k in keys for n in folds[k]]
    if len(names) != len(set(names)):
        raise SystemExit('{} does not partition the truckloads: one appears in two folds, which '
                         'would leak it into the model that predicts it'.format(fold_file))

    return {k: sorted(folds[k]) for k in keys}, manifest


def describe_folds(folds: dict, manifest: dict) -> str:
    """The fold table, printed up front so a split can be checked before hours are spent."""
    campaign_of = manifest.get('campaign_of', {})
    lines = []
    for key in sorted(folds, key=int):
        per_campaign: dict = {}
        for name in folds[key]:
            label = campaign_of.get(name, '?')
            per_campaign[label] = per_campaign.get(label, 0) + 1
        detail = ', '.join('{} x {}'.format(v, c) for c, v in sorted(per_campaign.items()))
        lines.append('  fold {}: {:2d} loads ({})'.format(key, len(folds[key]), detail))
        lines.append('          ' + ' '.join(folds[key]))
    return '\n'.join(lines)


# ------------------------------------------------------------------------------------- one fold

def fold_is_done(log_dir: str, n_held_out: int) -> bool:
    """A fold counts as finished only if both the last checkpoint and every prediction exist."""
    if not exists(join(log_dir, 'model', 'model_last.pth')):
        return False
    return len(glob.glob(join(log_dir, PREDICTIONS_DIR, '*.npy'))) >= n_held_out


def train_fold(fold: int, folds: dict, log_dir: str, args) -> None:
    """Train on every fold but this one, holding this one out for validation and test."""
    train_splits = tuple('fold_{}'.format(k) for k in sorted(folds, key=int) if int(k) != fold)
    held_out = 'fold_{}'.format(fold)

    # `train_eval` holds the same loads as `train` -- it is the frozen pass that produces the
    # training curve -- so it moves with it from fold to fold.
    options = ['save_path={}'.format(log_dir)]
    for section, split in (('train', train_splits), ('train_eval', train_splits),
                           ('val', held_out), ('test', held_out)):
        options += ['data.{}.split={}'.format(section, repr(split)),
                    'data.{}.data_root={}'.format(section, repr(args.dataset)),
                    'data.{}.fold_file={}'.format(section, repr(args.folds_file))]

    if args.epoch is not None:
        options.append('epoch={}'.format(args.epoch))
    if args.eval_epoch is not None:
        options.append('eval_epoch={}'.format(args.eval_epoch))
    options += args.options

    backend.run_tool('train', options, args.pointcept_root, args.num_gpus, args.config)


def predict_fold(log_dir: str, n_held_out: int, args) -> list:
    """Collect the held-out predictions, running inference only if they are not already there.

    Training ends with a pass over the held-out fold on the final weights, so a fold that trained
    to completion has already written its predictions. Re-running inference is only needed when
    training was interrupted after the last epoch, or when an older fold folder is revisited -- and
    a full pass over six multi-hundred-thousand-point clouds is not cheap, so it is skipped when it
    can be.
    """
    existing = sorted(glob.glob(join(log_dir, 'result', '*_pred.npy')))
    if len(existing) < n_held_out:
        weight = join(log_dir, 'model', 'model_last.pth')
        if not exists(weight):
            raise FileNotFoundError('no last checkpoint at {}'.format(weight))
        print('  {}/{} predictions present, running inference on the last checkpoint'.format(
            len(existing), n_held_out))
        # The config saved beside the checkpoint, so inference matches what was trained
        backend.run_tool('test', ['save_path={}'.format(log_dir), 'weight={}'.format(weight)],
                         args.pointcept_root, args.num_gpus, join(log_dir, 'config.py'))
    else:
        print('  {} predictions already written at the end of training, skipping '
              'inference'.format(len(existing)))

    predictions = join(log_dir, PREDICTIONS_DIR)
    os.makedirs(predictions, exist_ok=True)
    written = sorted(glob.glob(join(log_dir, 'result', '*_pred.npy')))
    for source in written:
        shutil.copy(source, join(predictions, basename(source)))
    return [basename(p) for p in written]


# -------------------------------------------------------------------------------------- metrics

def read_tensorboard(log_dir: str) -> dict:
    """Per-epoch metrics, read back from the scalars the evaluators wrote.

    Both the training and the validation curves come from frozen, no-grad passes over whole clouds
    at original resolution, produced by the same code -- so they are directly comparable. The
    per-iteration training loss Pointcept also logs is not: it is measured on augmented crops.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print('  ! tensorboard is not importable, so per-epoch metrics cannot be read')
        return {}

    if not glob.glob(join(log_dir, 'events.out.tfevents.*')):
        return {}

    accumulator = EventAccumulator(log_dir, size_guidance={'scalars': 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get('scalars', []))

    wanted = {}
    for tag_prefix, column_prefix in (('val', 'eval_val'), ('train_eval', 'eval_train')):
        for tag, column in (('loss', 'loss'), ('allAcc', 'acc'), ('mIoU', 'miou'),
                            ('mAcc', 'macc'), ('macroF1', 'macrof1')):
            wanted['{}/{}'.format(tag_prefix, tag)] = '{}_{}'.format(column_prefix, column)
        for index, name in enumerate(CLASS_NAMES):
            wanted['{}/cls_{}-{}_IoU'.format(tag_prefix, index, name)] = \
                '{}_iou_{}'.format(column_prefix, name)
            wanted['{}/cls_{}-{}_F1'.format(tag_prefix, index, name)] = \
                '{}_f1_{}'.format(column_prefix, name)

    per_epoch: dict = {}
    for tag, column in wanted.items():
        if tag not in available:
            continue
        for event in accumulator.Scalars(tag):
            # Logged at epoch + 1; stored 0-based so the CSV counts from zero
            per_epoch.setdefault(event.step - 1, {})[column] = float(event.value)

    return per_epoch


def collect_fold_metrics(log_dir: str, fold: int) -> list:
    return [dict(fold=fold, epoch=epoch, **values)
            for epoch, values in sorted(read_tensorboard(log_dir).items())]


def metric_columns() -> list:
    columns = ['fold', 'epoch']
    for prefix in ('eval_train', 'eval_val'):
        columns += ['{}_loss'.format(prefix), '{}_acc'.format(prefix)]
        columns += ['{}_iou_{}'.format(prefix, n) for n in CLASS_NAMES]
        columns += ['{}_miou'.format(prefix), '{}_macc'.format(prefix),
                    '{}_macrof1'.format(prefix)]
        columns += ['{}_f1_{}'.format(prefix, n) for n in CLASS_NAMES]
    return columns


def write_metrics(out_dir: str, rows: list) -> None:
    """The per-epoch curve and the per-fold summary -- what `semantictruck-plot-curves` reads."""
    if not rows:
        print('  no metrics found')
        return

    per_epoch_path = join(out_dir, METRICS_PER_EPOCH_CSV)
    with open(per_epoch_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns(),
                                extrasaction='ignore', restval='')
        writer.writeheader()
        writer.writerows(rows)
    print('  {} ({} rows)'.format(per_epoch_path, len(rows)))

    key = 'eval_val_miou'
    summary_columns = ['fold', 'n_epochs', 'metric', 'final_val_miou', 'final_val_acc',
                       'final_train_miou', 'best_val_miou', 'best_epoch']
    summary = []
    for fold in sorted(set(r['fold'] for r in rows)):
        fold_rows = [r for r in rows if r['fold'] == fold and key in r]
        if not fold_rows:
            continue
        best = max(fold_rows, key=lambda r: r[key])
        summary.append({
            'fold': fold, 'n_epochs': len(fold_rows), 'metric': key,
            # The final epoch, not the best: the validation fold is the held-out fold
            'final_val_miou': fold_rows[-1][key],
            'final_val_acc': fold_rows[-1].get('eval_val_acc', float('nan')),
            'final_train_miou': fold_rows[-1].get('eval_train_miou', float('nan')),
            'best_val_miou': best[key], 'best_epoch': best['epoch'],
        })

    if not summary:
        return

    mean_row = {'fold': 'mean', 'n_epochs': '', 'metric': key}
    for column in summary_columns[3:]:
        mean_row[column] = float(np.mean([s[column] for s in summary]))
    summary.append(mean_row)

    summary_path = join(out_dir, METRICS_SUMMARY_CSV)
    with open(summary_path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(summary)
    print('  {}'.format(summary_path))
    for row in summary:
        print('    fold {:>4}  final mIoU {:.4f}  best {:.4f} @ epoch {}'.format(
            str(row['fold']), row['final_val_miou'], row['best_val_miou'], row['best_epoch']))


# ---------------------------------------------------------------------------------- the campaign

def pool_predictions(out_dir: str, folds: dict) -> int:
    """Gather every fold's held-out predictions into one directory.

    This is the set to hand to the geometry stage: one prediction per truckload, each made by the
    model that never saw it.
    """
    pooled = join(out_dir, PREDICTIONS_DIR)
    os.makedirs(pooled, exist_ok=True)

    count = 0
    for fold in sorted(folds, key=int):
        for source in sorted(glob.glob(join(out_dir, 'fold_{}'.format(fold),
                                            PREDICTIONS_DIR, '*.npy'))):
            shutil.copy(source, join(pooled, basename(source)))
            count += 1

    if count:
        total = sum(len(v) for v in folds.values())
        print('\nout-of-fold predictions ({}/{} truckloads) -> {}'.format(count, total, pooled))
        print('  labels are 0 = wood (cargo), 1 = vehicle, so the geometry stage reads them with')
        print('  --pred-dir {} --pred-wood-label 0'.format(pooled))
    return count


def run_campaign(args) -> int:
    """Every fold, then the pooled predictions and the metric tables."""
    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)

    folds, manifest = load_folds(args.folds_file, args.n_folds)
    n_loads = sum(len(v) for v in folds.values())

    print('=' * 78)
    print('cross-validation setup')
    print('=' * 78)
    print('  pointcept : {}'.format(args.pointcept_root))
    print('  dataset   : {}'.format(args.dataset))
    print('  folds     : {} (from {})'.format(args.n_folds, args.folds_file))
    print('  config    : {}'.format(args.config))
    print('  out       : {}'.format(out_dir))
    print(describe_folds(folds, manifest))

    shutil.copy(args.folds_file, join(out_dir, 'folds.json'))

    rows = []
    for fold in (range(args.n_folds) if args.only_fold is None else [args.only_fold]):
        log_dir = join(out_dir, 'fold_{}'.format(fold))
        n_held_out = len(folds[str(fold)])

        if not args.metrics_only:
            if fold_is_done(log_dir, n_held_out) and not args.force:
                print('\nfold {} already finished ({} predictions) -- skipping. --force '
                      'retrains it.'.format(fold, n_held_out))
            else:
                print('\n' + '=' * 78)
                print('fold {} -- training on {} loads, holding out {}'.format(
                    fold, n_loads - n_held_out, n_held_out))
                print('=' * 78)
                started = time.time()
                train_fold(fold, folds, log_dir, args)
                print('\nfold {} trained in {:.1f} min'.format(fold, (time.time() - started) / 60))

                if not args.skip_inference:
                    written = predict_fold(log_dir, n_held_out, args)
                    print('fold {}: stored {} held-out predictions'.format(fold, len(written)))

        rows += collect_fold_metrics(log_dir, fold)

    pool_predictions(out_dir, folds)

    print('\n' + '=' * 78)
    print('metrics')
    print('=' * 78)
    write_metrics(out_dir, rows)
    return 0
