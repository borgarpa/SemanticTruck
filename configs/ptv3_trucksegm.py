"""PTv3 semantic segmentation of truckloads (wood cargo against vehicle), 5-fold.

A **Pointcept** configuration, in Pointcept's own format, executed by its config loader. It is not
a `Params` file and has nothing to do with `configs/manuscript.json`, which configures the geometry.

Derived from Pointcept's `configs/s3dis/semseg-pt-v3m1-0-base.py`. The model, optimiser and
schedule are PTv3's own tuned recipe, unchanged; what differs is the data plumbing -- two classes,
no normals, fold-based splits -- and the batch/accumulation split needed to fit a single 8 GB card.

**This file is self-contained.** The upstream configuration inherits Pointcept's
`configs/_base_/default_runtime.py` through a relative path into its config tree, which cannot
resolve from outside that tree. Those defaults are inlined below instead, so the whole
configuration is readable in one place and the file can live in this repository. The cost is that
it no longer tracks Pointcept's defaults: check this file against `default_runtime.py` after
upgrading Pointcept.

`semantictruck-train` overrides the splits, the data root and the folds manifest per fold, so the
values here are only what a bare `tools/train.py` invocation would see.
"""

# --------------------------------------------------------------------------- runtime
# Inlined from Pointcept's configs/_base_/default_runtime.py. Only the keys this configuration
# does not set for itself are here; anything it overrides below is left out to avoid stating a
# value twice.
weight = None                   # path to a checkpoint to load
resume = False
evaluate = True                 # evaluate after each epoch
test_only = False
seed = None                     # drawn and recorded per run if left None
save_path = "exp/default"       # overridden per fold
batch_size_val = None           # auto: batch size 1 per GPU
batch_size_test = None
clip_grad = None
sync_bn = False
amp_dtype = "float16"
empty_cache_per_epoch = False
find_unused_parameters = False
wandb_project = "semantictruck"
wandb_key = None

train = dict(type="DefaultTrainer")
test = dict(type="SemSegTester", verbose=True)

# --------------------------------------------------------------------- campaign settings
# batch_size 12 (the S3DIS value) does not fit in 8 GB, so it is split into 2 x 6 accumulation
# steps, which keeps the effective batch -- and therefore the tuned learning rate -- meaningful.
batch_size = 2
gradient_accumulation_steps = 6
num_worker = 8
mix_prob = 0.8
empty_cache = False
enable_amp = True
# TensorBoard only: `semantictruck-train` reads the scalars back to build its metric CSVs.
enable_wandb = False

# model settings
model = dict(
    type="DefaultSegmentorV2",
    num_classes=2,
    backbone_out_channels=64,
    backbone=dict(
        type="PT-v3m1",
        # feat_keys=("coord", "color") below -> 3 + 3. This data has no normals; coord carries the
        # height cue that the KPConvX stage fed as its separate z channel.
        in_channels=6,
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(64, 64, 128, 256),
        dec_num_head=(4, 4, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        shuffle_orders=True,
        pre_norm=True,
        enable_rpe=False,
        # If flash_attn is unavailable, set enable_flash=False and drop enc/dec_patch_size to 128
        # (Pointcept README); the model auto-clamps patch_size to the point count in that path.
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        enc_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D"),
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
)

# scheduler settings
# default_config_parser sets data.train.loop = epoch // eval_epoch (= 30) and the trainer runs
# eval_epoch real epochs. default_setup asserts epoch % eval_epoch == 0.
epoch = 1500
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.006, 0.0006],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0006)]

# hooks: identical to default_runtime.py except PreciseEvaluator, which defaults to test_last=False
# and would therefore auto-test model_best. In this cross-validation the validation fold *is* the
# held-out fold, so selecting a checkpoint on it would leak. Always report the last epoch.
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="ModelHook"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    # Frozen no-grad pass over the TRAINING loads with the end-of-epoch weights, no augmentation,
    # no crop, per original point. Pointcept otherwise reports only a running per-iteration train
    # loss on augmented crops, which is not comparable with the validation curve. Runs before
    # SemSegEvaluator purely for log readability; it never touches current_metric_value.
    dict(type="TrainSetEvaluator", write_cls_metrics=True),
    # write_cls_metrics defaults to False upstream, which silently drops the per-class
    # val/cls_*_IoU and val/cls_*_F1 scalars and leaves eval_val_iou_* empty in the CSV.
    dict(type="SemSegEvaluator", write_cls_metrics=True),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=True),
]

# dataset settings
dataset_type = "SemanticTruckDataset"
# Supplied by `semantictruck-train`, which overrides them per fold. Left unset here so that a
# forgotten --dataset/--folds fails with the dataset's own message rather than reading
# somebody else's directory.
data_root = None
fold_file = None
grid_size = 0.02

data = dict(
    num_classes=2,
    ignore_index=-1,
    names=["wood", "truck"],
    train=dict(
        type=dataset_type,
        split=("fold_0", "fold_1", "fold_2", "fold_3"),
        data_root=data_root,
        fold_file=fold_file,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis="z", center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="x", p=0.5),
            dict(type="RandomRotate", angle=[-1 / 64, 1 / 64], axis="y", p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            # Whole clouds are 417k-839k points at 0.02 m, well past what 8 GB can backprop
            # through. ScanNet's 102400 cap is used rather than S3DIS's 204800.
            dict(type="SphereCrop", sample_rate=0.6, mode="random"),
            dict(type="SphereCrop", point_max=102400, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        type=dataset_type,
        split="fold_4",
        data_root=data_root,
        fold_file=fold_file,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    # Same loads as `train`, but carrying the `val` transform pipeline: no augmentation, no crop,
    # and the inverse mapping back to the original points. Consumed by the TrainSetEvaluator hook
    # so the training curve is produced by the same code as the validation one.
    train_eval=dict(
        type=dataset_type,
        split=("fold_0", "fold_1", "fold_2", "fold_3"),
        data_root=data_root,
        fold_file=fold_file,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
                return_inverse=True,
            ),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "origin_segment", "inverse"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        type=dataset_type,
        split="fold_4",
        data_root=data_root,
        fold_file=fold_file,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            # Single identity pass rather than S3DIS's 10-way scale/flip TTA: each aug multiplies
            # inference cost by one full pass over every cloud, and the held-out set is re-predicted
            # once per fold. Re-enable if the extra accuracy is worth the time.
            aug_transform=[[dict(type="RandomScale", scale=[1, 1])]],
        ),
    ),
)
