from autolightning import cc

world_size = 1
batch_size_per_device = 64

config = dict(
    model=cc(
        "autolightning.lm.self_supervised.DINO",
        dict(
            student_backbone="resnet50",
            student_head=cc(
                "lightly.models.modules.DINOProjectionHead",
                input_dim=2048, hidden_dim=512, bottleneck_dim=64
            ),
            criterion=cc(
                "lightly.loss.DINOLoss",
                output_dim=2048, warmup_teacher_temp_epochs=5
            ),
            momentum_bounds=(0.996, 1.0)
        )
    ),
    data=cc(
        "autolightning.AutoDataModule",
        dict(
            dataset=cc(
                "torchvision.datasets.VOCDetection",
                dict("datasets/pascal_voc", download=True)
            ),
            transforms=cc(
                "lightly.transforms.dino_transform.DINOTransform",
                global_crop_scale=(0.14, 1), local_crop_scale=(0.05, 0.14)
            )
        )
    ),
    optimizer=cc(
        "SGD",
        lr=0.03 * batch_size_per_device * world_size / 256,
        momentum=0.9,
        weight_decay=1e-4
    ),
    trainer=dict(
        gradient_clip_val=3.0,
        gradient_clip_algorithm="norm"
    )
)
