"""Per-truckload inference: predictions, metrics and timings from a trained checkpoint.

`semantictruck-train` already predicts each fold's held-out loads at the end of training and pools
them. This module re-runs that inference deliberately, and records **one row per truckload** rather
than the whole-set aggregates a training campaign writes -- neither the evaluator nor the tester
keeps a per-cloud record, both push each cloud's counts straight into a running sum.

That per-load record is what two things need: the computation-time study, which needs to know what
segmenting one truckload costs, and any re-prediction of the dataset with a checkpoint other than
the one training happened to end on.

**Two evaluation paths exist and they do not agree**; `grid_mode` picks between them:

    train (default)   one forward per cloud over the voxelized points, the voxel's label mapped
                      back to every point that fell in it. This is what produced a campaign's
                      `metrics_per_epoch.csv`, so it is the number to reproduce -- but the voxel
                      representative is drawn at random, so agreement is to ~1e-3, not bitwise.
    test              the path that wrote each fold's stored predictions: roughly 25-30 fragments
                      per cloud with the softmax accumulated per point. No RNG, but about twenty
                      times the GPU work.

**The forward is not bit-reproducible on every GPU.** A verbatim replay of the tester loop can still
disagree with a stored prediction on a fraction of a percent of points, and merely adding a warm-up
forward changes which points -- kernel selection is call-history dependent. Those points sit on
near-ties along the wood/vehicle boundary and move accuracy and mIoU by ~1e-4. Compare the metrics,
not the label arrays; `--check` does exactly that.

Accuracy and IoU are always at **full point resolution**, as both evaluators report them. The loss
is voxel-level in both modes, and in test mode is the mean over a cloud's fragments -- the tester
never computes a loss at all, so it is computed here from each fragment's logits. The `grid_mode`
column records which path a row came from; **do not pool the two**.

This module imports Pointcept and runs in the training environment, which has no pandas: everything
here is `csv` and NumPy.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import time
from collections import OrderedDict
from os.path import dirname, exists, join

import numpy as np

from semantictruck.campaigns import campaign_of
from semantictruck.training import CLASS_NAMES

try:
    import torch
    import torch.nn.functional as F
    from pointcept.datasets import build_dataset
    from pointcept.datasets.utils import collate_fn
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    from pointcept.utils.env import set_seed
    from pointcept.utils.misc import intersection_and_union, intersection_and_union_gpu
except ImportError as missing:                                  # pragma: no cover
    raise ImportError(
        '{}\n\nsemantictruck.training.inference needs Pointcept and torch importable. They are '
        'dependencies of the training stage only; see docs/reproducing.md.'.format(missing))

# What a bare `--save-pred` stands for until a temporary directory is made for it
TEMP_PRED = 'auto'

# What the stored-prediction comparison writes when it has nothing to compare against: a training
# row, a missing stored prediction, or a repeat pass -- which reuses the first pass's answer rather
# than paying to reload tens of megabytes for a number that cannot change.
BLANK_SAVED = dict(saved_disagree_pts='', saved_disagree_pct='', saved_acc='', saved_miou='')

# Summarised per fold and set. `t_transform_s` is deliberately absent: it is a CPU cost that says
# nothing about the model, and it is on every per-sample row for anyone who wants it.
METRICS = ('t_forward_s', 't_total_s', 'loss', 'acc', 'miou')

EPS = 1e-10


# -------------------------------------------------------------------------------------- metrics

def metrics_from_counts(intersection, union, target) -> dict:
    """Per-class and global metrics from one set of intersection/union/target counts.

    The formulas are the evaluator's, unchanged. The only difference is what is fed in: the
    evaluator pools the counts over a whole set before applying them, this applies them to one
    cloud at a time. Pooling them back over a set therefore reproduces the evaluator exactly --
    which is why the summary carries `pooled_` figures beside the per-load means, and why the two
    are not interchangeable.
    """
    intersection = np.asarray(intersection, dtype=np.float64)
    union = np.asarray(union, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    area_output = union + intersection - target

    iou = intersection / (union + EPS)
    recall = intersection / (target + EPS)
    precision = intersection / (area_output + EPS)
    f1 = 2 * (precision * recall) / (precision + recall + EPS)

    out = dict(acc=float(intersection.sum() / (target.sum() + EPS)),
               miou=float(np.mean(iou)), macc=float(np.mean(recall)),
               macrof1=float(np.mean(f1)))
    for index, name in enumerate(CLASS_NAMES):
        out['iou_' + name] = float(iou[index])
        out['acc_' + name] = float(recall[index])
        out['f1_' + name] = float(f1[index])
    return out


def mean_sd(values) -> tuple:
    """Mean and *sample* SD (ddof=1), the convention used across this project's tables."""
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if values.size == 0:
        return float('nan'), float('nan')
    if values.size == 1:
        return float(values[0]), float('nan')
    return float(values.mean()), float(values.std(ddof=1))


# ---------------------------------------------------------------------------------------- setup

def load_folds(fold_file: str) -> dict:
    with open(fold_file) as handle:
        manifest = json.load(handle)
    return manifest.get('folds', manifest)


def build_fold_model(cfg, weight: str):
    """Instantiate a fold's model and load a checkpoint.

    The `module.` prefix is stripped for single-GPU, and the load is `strict=True` so that a
    configuration and checkpoint that disagree fail loudly rather than silently evaluating a
    partly random network.
    """
    if not exists(weight):
        raise FileNotFoundError('no checkpoint at {}'.format(weight))

    model = build_model(cfg.model)
    checkpoint = torch.load(weight, map_location='cpu', weights_only=False)
    state_dict = OrderedDict()
    for key, value in checkpoint['state_dict'].items():
        state_dict[key[7:] if key.startswith('module.') else key] = value
    model.load_state_dict(state_dict, strict=True)
    return model.cuda().eval(), checkpoint.get('epoch')


def dataset_config(cfg, grid_mode: str, split) -> dict:
    """The dataset entry for one set, with only `split` overridden.

    `train` mode uses the frozen, unaugmented entry the training run evaluated with; `test` mode
    uses the entry whose `test_mode=True` yields a fragment list plus the full-resolution labels.
    """
    source = cfg.data.test if grid_mode == 'test' else cfg.data.val
    entry = copy.deepcopy(dict(source))
    entry['split'] = split
    return entry


def to_cuda(input_dict: dict) -> dict:
    for key in input_dict:
        if isinstance(input_dict[key], torch.Tensor):
            input_dict[key] = input_dict[key].cuda(non_blocking=True)
    return input_dict


# ------------------------------------------------------------------------------------ inference

def run_sample_train_mode(model, dataset, index, num_classes, ignore_index) -> dict:
    """One cloud through the evaluator path: a single forward over the voxelized cloud."""
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    data = dataset[index]
    input_dict = to_cuda(collate_fn([data]))
    torch.cuda.synchronize()
    t_transform = time.perf_counter() - started

    forward_started = time.perf_counter()
    with torch.no_grad():
        output_dict = model(input_dict)
    torch.cuda.synchronize()
    t_forward = time.perf_counter() - forward_started

    seg_logits = output_dict['seg_logits']
    loss = output_dict['loss'].item()
    pred = seg_logits.max(1)[1]
    segment = input_dict['segment']
    n_voxels = int(seg_logits.shape[0])

    # Full point resolution, as the evaluator does it: the voxel's label goes back to every point
    # that fell in it
    if 'inverse' in input_dict:
        pred = pred[input_dict['inverse']]
        segment = input_dict['origin_segment']
    torch.cuda.synchronize()
    t_total = time.perf_counter() - started

    intersection, union, target = intersection_and_union_gpu(
        pred, segment, num_classes, ignore_index)
    return dict(
        load_id=dataset.get_data_name(index), n_points=int(segment.numel()), n_voxels=n_voxels,
        n_fragments=1, t_transform_s=t_transform, t_forward_s=t_forward, t_total_s=t_total,
        gpu_peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20, loss=loss,
        counts=(intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()),
        pred=pred.cpu().numpy(), segment=segment.cpu().numpy())


def run_sample_test_mode(model, dataset, index, num_classes, ignore_index) -> dict:
    """One cloud through the tester path: every fragment, softmax accumulated per point.

    The test-mode grid emits as many parts as the most populated voxel holds, each carrying one
    point per voxel and cycling through them, so a point in a sparsely populated voxel is seen
    several times and its softmax accumulated several times. Reproducing that accumulation is what
    makes the prediction match the one stored at training time.
    """
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()

    data = dataset[index]
    fragments = data.pop('fragment_list')
    segment_np = data.pop('segment')
    load_id = data.pop('name')
    t_transform = time.perf_counter() - started

    segment = torch.as_tensor(np.asarray(segment_np)).long().cuda()
    scores = torch.zeros((segment.numel(), num_classes), device='cuda')

    t_forward = 0.0
    losses = []
    n_voxels = 0
    for fragment in fragments:
        input_dict = to_cuda(collate_fn([fragment]))
        part = input_dict['index'].long()

        torch.cuda.synchronize()
        forward_started = time.perf_counter()
        with torch.no_grad():
            seg_logits = model(input_dict)['seg_logits']
        torch.cuda.synchronize()
        t_forward += time.perf_counter() - forward_started

        n_voxels = int(seg_logits.shape[0])
        with torch.no_grad():
            # The fragment carries no labels -- they are removed before fragmenting -- but `index`
            # says which original points it holds, so they are recoverable.
            losses.append(model.criteria(seg_logits, segment[part]).item())
            scores[part] += F.softmax(seg_logits, -1)

    pred = scores.max(1)[1]
    torch.cuda.synchronize()
    t_total = time.perf_counter() - started

    intersection, union, target = intersection_and_union_gpu(
        pred, segment, num_classes, ignore_index)
    return dict(
        load_id=load_id, n_points=int(segment.numel()), n_voxels=n_voxels,
        n_fragments=len(fragments), t_transform_s=t_transform, t_forward_s=t_forward,
        t_total_s=t_total, gpu_peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
        loss=float(np.mean(losses)),
        counts=(intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()),
        pred=pred.cpu().numpy(), segment=segment.cpu().numpy())


def warmup(model, dataset, grid_mode: str, n: int) -> None:
    """Discarded forwards, so the first truckload does not carry the kernel-selection cost."""
    if n <= 0:
        return
    data = dataset[0]
    fragment = data['fragment_list'][0] if grid_mode == 'test' else data
    for _ in range(n):
        with torch.no_grad():
            model(to_cuda(collate_fn([fragment])))
    torch.cuda.synchronize()


def save_prediction(directory: str, load_id: str, pred: np.ndarray) -> None:
    """Write one load's per-point labels where the geometry stage can read them.

    Same name, layout and convention as the predictions stored at training time --
    `<load>_pred.npy`, one label per point, 0 = wood -- so a directory written here is a drop-in
    for `semantictruck-pipeline --pred-dir`.
    """
    os.makedirs(directory, exist_ok=True)
    np.save(join(directory, '{}_pred.npy'.format(load_id)), pred)


def score_saved(campaign_dir: str, fold: int, load_id: str, pred, segment,
                num_classes: int, ignore_index: int) -> dict:
    """Score the fold's stored prediction on the same labels, and count where the two disagree.

    The stored predictions came from the tester at the end of training, so `test` mode is
    reproducing them and this is the tightest reference available: per load, on identical ground
    truth. It is deliberately **not** a bitwise check -- see the module docstring.
    """
    saved = join(campaign_dir, 'fold_{}'.format(fold), 'result', '{}_pred.npy'.format(load_id))
    if not exists(saved):
        return dict(BLANK_SAVED)

    reference = np.load(saved)
    if reference.shape != pred.shape:
        return dict(BLANK_SAVED, saved_disagree_pts='shape mismatch')

    metrics = metrics_from_counts(
        *intersection_and_union(reference, segment, num_classes, ignore_index))
    n_different = int((reference != pred).sum())
    return dict(saved_disagree_pts=n_different,
                saved_disagree_pct=100.0 * n_different / pred.size,
                saved_acc=metrics['acc'], saved_miou=metrics['miou'])


# ------------------------------------------------------------------------------------- one fold

def run_fold(args, fold: int, folds: dict, log) -> list:
    """Every load of every requested set, at this fold's checkpoint."""
    fold_dir = join(args.campaign, 'fold_{}'.format(fold))
    config_file = join(fold_dir, 'config.py')
    weight = join(fold_dir, 'model', args.checkpoint)
    if not exists(config_file):
        raise FileNotFoundError('no saved configuration at {}'.format(config_file))

    # The saved configuration, read rather than re-parsed: re-parsing would create directories and
    # rewrite the config inside a finished training run, and nothing here should write into one.
    cfg = Config.fromfile(config_file)
    set_seed(args.seed if args.seed is not None else cfg.seed)

    num_classes = cfg.data.num_classes
    ignore_index = cfg.data.ignore_index

    model, epoch = build_fold_model(cfg, weight)
    log('fold {}: {} (epoch {})'.format(fold, weight, epoch))

    held_out = 'fold_{}'.format(fold)
    train_splits = tuple('fold_{}'.format(k) for k in sorted(folds, key=int) if int(k) != fold)

    rows = []
    for set_name in args.sets:
        split = held_out if set_name == 'val' else train_splits
        dataset = build_dataset(dataset_config(cfg, args.grid_mode, split))
        n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
        log('  {} set: {} load(s){}'.format(
            set_name, n, '' if args.limit is None else ' (limited from {})'.format(len(dataset))))

        # Once per (fold, set), not once per pass: the kernel-selection cost it absorbs is paid by
        # the process, not by the pass
        warmup(model, dataset, args.grid_mode, args.warmup)
        runner = run_sample_test_mode if args.grid_mode == 'test' else run_sample_train_mode

        for rep in range(args.repeats):
            if args.repeats > 1:
                log('    pass {}/{}'.format(rep + 1, args.repeats))

            for index in range(n):
                result = runner(model, dataset, index, num_classes, ignore_index)
                intersection, union, target = result.pop('counts')
                pred = result.pop('pred')
                segment = result.pop('segment')

                row = dict(fold=fold, rep=rep, set=set_name, grid_mode=args.grid_mode,
                           campaign=campaign_of(result['load_id']), checkpoint=args.checkpoint,
                           seed=args.seed if args.seed is not None else cfg.seed, **result)

                # The held-out set only, and the first pass only: a training load is predicted by a
                # model that saw it and would collide across folds, and a later pass differs on the
                # handful of points where the forward is not reproducible
                if args.save_pred and set_name == 'val' and rep == 0:
                    save_prediction(args.save_pred, result['load_id'], pred)

                row.update(metrics_from_counts(intersection, union, target))
                for j, name in enumerate(CLASS_NAMES):
                    row['inter_' + name] = int(intersection[j])
                    row['union_' + name] = int(union[j])
                    row['target_' + name] = int(target[j])
                row.update(score_saved(args.campaign, fold, result['load_id'], pred, segment,
                                       num_classes, ignore_index)
                           if rep == 0 else dict(BLANK_SAVED))
                rows.append(row)

                log('    [{}/{}] {:<16} {:>9} pts  {:>6.2f} s fwd  {:>6.2f} s total  '
                    'loss {:.4f}  acc {:.4f}  mIoU {:.4f}{}'.format(
                        index + 1, n, row['load_id'], row['n_points'], row['t_forward_s'],
                        row['t_total_s'], row['loss'], row['acc'], row['miou'],
                        '' if row['saved_disagree_pct'] == '' else
                        '  stored-pred d_mIoU {:+.5f}, {:.3f} % of points differ'.format(
                            row['miou'] - row['saved_miou'], row['saved_disagree_pct'])))

        del dataset
    del model
    torch.cuda.empty_cache()
    return rows


# --------------------------------------------------------------------------------------- summary

def summarise(rows: list, sets, grid_mode: str, with_rep: bool = False) -> list:
    """One row per (fold, set), or per (fold, set, pass) when more than one pass was run.

    The `rep="all"` rows pool every pass, which is the form a timing study quotes; the per-pass
    rows are what show whether the machine drifted while it ran.
    """
    if not with_rep:
        return _summarise(rows, sets, grid_mode)

    summary = []
    for rep in sorted({r['rep'] for r in rows}) + ['all']:
        group = rows if rep == 'all' else [r for r in rows if r['rep'] == rep]
        summary += [dict(entry, rep=rep) for entry in _summarise(group, sets, grid_mode)]
    return summary


def _summarise(rows: list, sets, grid_mode: str) -> list:
    """One row per (fold, set), plus an `all` row per set.

    Three kinds of number sit side by side and they are **not interchangeable**:

        mean_ / sd_                   over the individual truckloads
        pooled_                       from summed intersection/union/target -- the only form that
                                      reproduces a training campaign's tables, because pooling
                                      weights by cloud size
        mean_over_folds_ / sd_over_   mean and SD of the per-fold means, which is how a campaign
                                      summary builds its own mean row. On the `all` rows only.
    """
    folds = sorted({r['fold'] for r in rows})
    summary = []

    for set_name in sets:
        set_rows = [r for r in rows if r['set'] == set_name]
        if not set_rows:
            continue
        fold_means = {m: [] for m in METRICS}

        for fold in folds + ['all']:
            group = set_rows if fold == 'all' else [r for r in set_rows if r['fold'] == fold]
            if not group:
                continue

            out = dict(fold=fold, set=set_name, grid_mode=grid_mode, n_samples=len(group))
            for metric in METRICS:
                mean, sd = mean_sd([r[metric] for r in group])
                out['mean_' + metric] = mean
                out['sd_' + metric] = sd
                if fold != 'all':
                    fold_means[metric].append(mean)

            pooled = metrics_from_counts(
                [sum(r['inter_' + n] for r in group) for n in CLASS_NAMES],
                [sum(r['union_' + n] for r in group) for n in CLASS_NAMES],
                [sum(r['target_' + n] for r in group) for n in CLASS_NAMES])
            out['pooled_acc'] = pooled['acc']
            out['pooled_miou'] = pooled['miou']
            for name in CLASS_NAMES:
                out['pooled_iou_' + name] = pooled['iou_' + name]

            for metric in METRICS:
                mean, sd = mean_sd(fold_means[metric]) if fold == 'all' else (None, None)
                out['mean_over_folds_' + metric] = '' if mean is None else mean
                out['sd_over_folds_' + metric] = '' if sd is None else sd
            summary.append(out)

    return summary


# ------------------------------------------------------------------------------------- the CSVs

def per_sample_columns(with_rep: bool = False) -> list:
    columns = ['fold'] + (['rep'] if with_rep else [])
    columns += ['set', 'grid_mode', 'load_id', 'campaign', 'n_points', 'n_voxels', 'n_fragments',
                't_transform_s', 't_forward_s', 't_total_s', 'gpu_peak_mib',
                'loss', 'acc', 'miou', 'macc', 'macrof1']
    for prefix in ('iou_', 'acc_', 'f1_', 'inter_', 'union_', 'target_'):
        columns += [prefix + name for name in CLASS_NAMES]
    return columns + ['saved_disagree_pts', 'saved_disagree_pct', 'saved_acc', 'saved_miou',
                      'checkpoint', 'seed']


def summary_columns(with_rep: bool = False) -> list:
    columns = ['fold'] + (['rep'] if with_rep else []) + ['set', 'grid_mode', 'n_samples']
    for metric in METRICS:
        columns += ['mean_' + metric, 'sd_' + metric]
    columns += ['pooled_acc', 'pooled_miou'] + ['pooled_iou_' + n for n in CLASS_NAMES]
    for metric in METRICS:
        columns += ['mean_over_folds_' + metric, 'sd_over_folds_' + metric]
    return columns


def write_csv(path: str, rows: list, columns: list) -> None:
    os.makedirs(dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def write_timing_csv(path: str, rows: list) -> None:
    """The per-load segmentation time, in the shape `semantictruck-timing` joins on.

    `t_external_s` is the whole cost of segmenting one truckload -- transform and forward -- which
    is what an end-to-end figure adds to the geometry time. Written separately from the full
    per-sample table so the timing study needs no knowledge of this one's schema.
    """
    held_out = [r for r in rows if r['set'] == 'val']
    write_csv(path, [dict(load_id=r['load_id'], rep=r['rep'], t_external_s=r['t_total_s'])
                     for r in held_out], ['load_id', 'rep', 't_external_s'])


def read_csv(path: str) -> list:
    if not exists(path):
        return []
    with open(path) as handle:
        return list(csv.DictReader(handle))
