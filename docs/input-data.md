# Input data

The truckload scans used in this study are proprietary to Forico Pty Limited, so no data is included in the repository. Those interested in replicating the experiment can reach the corresponding author using this email: [borja.garcia.pascual@uef.fi](borja.garcia.pascual@uef.fi). 

This page specifies what the appropriate dataset format is, so that users can create their own from other scans.

## Layout

```
<data root>/
├── clouds/                 one .laz per truckload
│   ├── Truck_1_2016.laz
│   └── ...
├── predictions/            optional: one .npy per truckload, for the `predicted` arm
│   └── Truck_1_2016_pred.npy
└── reference.csv           measured solid volume
```

The root is `$SEMANTICTRUCK_DATA`, or `--data-root`, or a local `data/` directory. Every path can also be overridden individually (`--clouds`, `--reference`, `--pred-dir`).

## The point clouds

One LAZ file per truckload, holding the **vehicle and its cargo together** after ground removal. The original clouds were generated using Structure-from-Motion photogrammetry on UAS imagery, with ground
points removed by a cloth simulation filter.

| Field | Type | Meaning |
|---|---|---|
| `x`, `y`, `z` | double | metres, in any consistent local frame |
| `red`, `green`, `blue` | uint16 | colour, full 16-bit range |
| `label` | extra dimension | **1 = roundwood, 2 = vehicle** |
| `bay` | extra dimension, optional | **bay id, 0 = no bay** |

`label` is the roundwood/vehicle segmentation and is what the `reference` arm reads. `bay` is a manual segmentation of the cargo into separate bays; only used to evaluate automatic bay individualisation.

### File names

A load whose name contains `_2021` is processed as a **multi-bay** load (k-means split, then DBSCAN refinement); anything else is processed as a single bay. The tag is `--multi-bay-tag`, so a dataset
naming its loads differently can be processed without renaming anything.

## Segmentation (optional)

### Training data

Pointcept reads arrays, not LAZ, so the training stage reads `<data root>/training/{train,test}/`
as `N × 7` `.npy` files, with columns `[x, y, z, label, r, g, b]`. The **the label must be located at column 3**, colours should range between 0–255, and labels `1 = roundwood, 2 = vehicle`.

### Predicted labels

To evaluate the semantic segmentation of some model, generate one `.npy` per truckload, named `<load_id>_pred.npy`. Each point has its own integer label.

`--pred-wood-label` specifies which value corresponds to _roundwood_, and defaults to `0`. `semantictruck-sensitivity --check-predictions` reports each load's predicted mask with the reference
one.

## The reference table

A CSV of measured solid wood volume, used to calibrate and to score. In the published work every log was measured for mid-diameter and length and its volume computed by Huber's formula.

| Column | Meaning |
|---|---|
| `File` | load id, matching the cloud's file name without its extension |
| `Solid_vol` | measured solid wood volume, m³ |

A load whose `Solid_vol` is blank is **ignored during regression model fitting but kept in the tables**.
