_base_ = (
    "../../configs/yolov5/yolov5_s-v61_syncbn_8xb16-300e_coco.py"
)

metainfo = dict(
    classes=(
        "PC",
        "PC-truck",
        "dozer",
        "dump-truck",
        "excavator",
        "mixer",
        "people-helmet",
        "people-no-helmet",
        "roller",
        "wheel-loader",
    )
)
num_classes=10
data_root = "/root/autodl-tmp"
resume = False
work_dir = "work_dirs"
model = dict(
    bbox_head=dict(
        head_module=dict(
            num_classes=num_classes),
        loss_cls=dict(
            loss_weight=_base_.loss_cls_weight *
            (num_classes / 80 * 3 / _base_.num_det_layers))
    ),
)

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file="annotations/train.json",
        backend_args=None,
        data_prefix=dict(img="train"),
        data_root=data_root,
        metainfo=metainfo,
    ),
    num_workers=10,
)

default_hooks = dict(
    param_scheduler=dict(
        lr_factor=0.01,
        max_epochs=100,
        scheduler_type='linear')
)
train_cfg = dict(
    dynamic_intervals=[
        (
            90,
            1,
        ),
    ],
    max_epochs=100,
    type='EpochBasedTrainLoop',
    val_interval=10
)
test_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file="annotations/test.json",
        backend_args=None,
        data_prefix=dict(img="test"),
        data_root=data_root,
        metainfo=metainfo,
    ),
)
test_evaluator = dict(ann_file=data_root + "/annotations/test.json")

val_dataloader = dict(
    batch_size=2,
    dataset=dict(
        ann_file="annotations/val.json",
        backend_args=None,
        data_prefix=dict(img="val"),
        data_root=data_root,
        metainfo=metainfo,
    ),
)
val_evaluator = dict(
    ann_file=data_root + "/annotations/val.json",
)
