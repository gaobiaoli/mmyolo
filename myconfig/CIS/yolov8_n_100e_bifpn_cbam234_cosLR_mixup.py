_base_ = "../../configs/yolov8/yolov8_n_syncbn_fast_8xb16-500e_coco.py"

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
# data_root = "/CV/gaobiaoli/dataset/CIS-Dataset"
data_root = "/root/autodl-tmp"

resume = False
work_dir = "work_dirs"
neck_in_channels=[ch * _base_.widen_factor for ch in [256, 512, _base_.last_stage_out_channels]]
custom_imports = dict(imports=['mmyolo.models.necks.bifpn'], allow_failed_imports=False)
model = dict(
    backbone=dict(
        plugins=[
            dict(
                cfg=dict(
                    type='CBAM'),
                stages=(False, True, True, True))
        ],
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        deepen_factor=_base_.deepen_factor,
        last_stage_out_channels=1024,
        norm_cfg=dict(eps=0.001, momentum=0.03, type='BN'),
        type='YOLOv8CSPDarknet',
        widen_factor=_base_.widen_factor),
    bbox_head=dict(
        head_module=dict(
            num_classes=10,
            in_channels=[512 for _ in range(3)])),
    neck=dict(
            _delete_=True,
            type='mmyolo.models.necks.bifpn.BiFPN',
            num_stages=6,
            in_channels=neck_in_channels,
            out_channels=512 * _base_.widen_factor,
            norm_cfg=_base_.norm_cfg),
        # dict(type="ASFFNeck", widen_factor=_base_.widen_factor, use_att="ASFF_sim"),
    train_cfg=dict(assigner=dict(num_classes=10)),
)

train_pipeline = [
    *_base_.pre_transform,
    dict(
        type='Mosaic',
        img_scale=_base_.img_scale,
        pad_val=114.0,
        pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        # img_scale is (width, height)
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    dict(
        type='YOLOv5MixUp',
        use_cached=True,
        max_cached_images=20),
    *_base_.last_transform
]

train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        ann_file="annotations/train.json",
        backend_args=None,
        data_prefix=dict(img="train"),
        data_root=data_root,
        metainfo=metainfo,
        pipeline=train_pipeline
    ),
    num_workers=16,
)

# auto_scale_lr = dict(enable=True, base_batch_size=8 * 16)
default_hooks = dict(
    param_scheduler=dict(lr_factor=0.01, max_epochs=100, scheduler_type="mix")
)
train_cfg = dict(
    dynamic_intervals=[
        (
            90,
            1,
        ),
    ],
    max_epochs=100,
    type="EpochBasedTrainLoop",
    val_interval=10,
)
custom_hooks = [
    dict(
        switch_epoch=90,
        switch_pipeline=[
            dict(backend_args=None, type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                scale=(
                    640,
                    640,
                ),
                type="YOLOv5KeepRatioResize",
            ),
            dict(
                allow_scale_up=True,
                pad_val=dict(img=114.0),
                scale=(
                    640,
                    640,
                ),
                type="LetterResize",
            ),
            dict(
                border_val=(
                    114,
                    114,
                    114,
                ),
                max_aspect_ratio=100,
                max_rotate_degree=0.0,
                max_shear_degree=0.0,
                scaling_ratio_range=(
                    0.5,
                    1.5,
                ),
                type="YOLOv5RandomAffine",
            ),
            dict(
                bbox_params=dict(
                    format="pascal_voc",
                    label_fields=[
                        "gt_bboxes_labels",
                        "gt_ignore_flags",
                    ],
                    type="BboxParams",
                ),
                keymap=dict(gt_bboxes="bboxes", img="image"),
                transforms=[
                    dict(p=0.01, type="Blur"),
                    dict(p=0.01, type="MedianBlur"),
                    dict(p=0.01, type="ToGray"),
                    dict(p=0.01, type="CLAHE"),
                ],
                type="mmdet.Albu",
            ),
            dict(type="YOLOv5HSVRandomAug"),
            dict(prob=0.5, type="mmdet.RandomFlip"),
            dict(
                meta_keys=(
                    "img_id",
                    "img_path",
                    "ori_shape",
                    "img_shape",
                    "flip",
                    "flip_direction",
                ),
                type="mmdet.PackDetInputs",
            ),
        ],
        type="mmdet.PipelineSwitchHook",
    ),
]

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
