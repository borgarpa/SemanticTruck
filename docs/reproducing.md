# Reproducing the published results

To apply the same pipeline described in the manuscript format your dataset as described in `docs/input-data.md`. Also, export the path to your data.

```bash
export SEMANTICTRUCK_DATA=/path/to/dataset
```

## The main results

```bash
semantictruck-pipeline --out results/pipeline
```

Writes the linear regression model `Vs = a·Vg + b`, as well as per-truckload gross and predicted net volumes with both
in-data and leave-one-out errors, the bay-split score, and per-step timings. With no `--config` it
runs the manuscript's configuration.

To score an automated segmentation as well, put its per-point label files in
`<data root>/predictions/` and add `--arms reference predicted`.

## The tables

```bash
semantictruck-errors --table results/pipeline/results_per_truck.csv --by-campaign
semantictruck-tables --run results/pipeline --out results/pipeline/tables
```

`semantictruck-tables` renders manuscript **Table 1** (gross and net volume by
campaign), **Table 3** (leave-one-out error by segmentation method) and **Table 4** (the same by
campaign).

## The figures

```bash
# Figure 3 -- cross-validation curves, from a training campaign's metrics_per_epoch.csv
semantictruck-plot-curves --csv <campaign dir> --metric miou acc loss

# Figure 4 -- solid against gross volume
semantictruck-plot-volume --table results/pipeline/results_per_truck.csv --fit-line

# Figure 5 -- computation time against size
semantictruck-plot-time --table results/time_results.xlsx
```

## The sensitivity analysis

```bash
semantictruck-sensitivity --out results/sensitivity
```

Eighty-four settings across four reconstruction campaigns, plus the supplementary figures. `--resume` continues an interrupted run from its checkpoints; `--analyse-only` rebuilds the tables
and figures without re-running anything.

## Segmentation model

### Training Point Transformer V3

Needs **Pointcept**, in its own environment; see the README's *Pointcept, for training the segmentation model* section.

```bash
conda activate semantictruck-train
export POINTCEPT_ROOT=/path/to/Pointcept
export SEMANTICTRUCK_DATA=/path/to/dataset

semantictruck-train --out results/segmentation
```

Five folds, each training on 24 truckloads and predicting the 6 it never saw. The fold assignment is **read** from `<data root>/folds.json`. Inference uses the **last** checkpoint on the validation fold.

Outputs generated under `--out` folder:

| | |
|---|---|
| `predictions/` | one out-of-fold prediction per truckload, which **will be fed into the reconstruction pipeline**; `semantictruck-predict` regenerates it |
| `metrics_per_epoch.csv` | one row per (fold, epoch), what `semantictruck-plot-curves` draws as Figure 3 |
| `metrics_summary.csv` | one row per fold, plus the mean over folds |
| `fold_<i>/` | the per-fold Pointcept run: checkpoints, logs, TensorBoard scalars |

### Segmentation inference

`semantictruck-train` already predicts labels for each fold's test loads at the end of training and stores them into `<out>/predictions/`. However, `semantictruck-predict` runs inference alone, and is what produces the **per-truckload** record.

Same environment as training: Pointcept, and a CUDA device.

```bash
conda activate semantictruck-train
export POINTCEPT_ROOT=/path/to/Pointcept

# per-load time, loss, accuracy and mIoU, and the masks the geometry stage reads
semantictruck-predict --campaign results/segmentation --sets val --grid-mode test \
    --save-pred results/segmentation/predictions --timing-csv
```

### Feeding predictions to the main pipeline

```bash
conda activate semantictruck            # back to the geometry environment
semantictruck-pipeline --arms reference predicted \
    --pred-dir results/segmentation/predictions --pred-wood-label 0
```

`--pred-wood-label 0` is the convention we followed. We recommend **declaring it explicitly**.


### Feeding the timing study

```bash
semantictruck-predict --campaign results/segmentation --sets val --grid-mode test \
    --repeats 5 --timing-csv results/segmentation/inference_timing.csv

conda activate semantictruck            # back to the geometry environment
semantictruck-timing --repeats 5 \
    --external-timings results/segmentation/inference_timing.csv
```

`--timing-csv` writes `load_id, rep, t_external_s` (the whole cost of segmenting one truckload, transform, and forward).

### Inference outputs

| | |
|---|---|
| `inference_per_sample[_test].csv` | one row per (fold, set, truckload, pass): time, loss, accuracy, mIoU, per-class counts, GPU peak |
| `inference_summary[_test].csv` | per (fold, set): means and SDs over loads, **pooled** metrics, and the mean over folds |
| `inference_timing[_test].csv` | the per-load cost, for the timing study |
| `inference[_test].log` | everything printed, including the check |

The summary carries three kinds of number side by side and they are **not interchangeable**: `mean_`/`sd_` are over individual truckloads, `pooled_` comes from summed counts, and `mean_over_folds_` is the mean of the per-fold means.

## The computation-time study

```bash
semantictruck-timing --repeats 5
```

**Segmentation timing is computed through** `semantictruck-predict --timing-csv`. Both results can then be joined using `--external-timings` with a CSV of per-load segmentation times keyed on `load_id`.

---

## A note on arm names

For generality's sake, in this package we call the two arms `reference` and `predicted`. Tables written in previous versions named these arms `manual` (`reference`) and `ptv3` (`predicted`); the columns are otherwise the same.
