from typing import Optional, Union, Tuple, List, Literal

import torch
from torch import Tensor
import torch.nn as nn

from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead

from autolightning.lm.self_supervised.lightly import SSL
from autolightning.types import Unpack, Phase, AutoModuleKwargsNoNetCriterion


# TODO! add support for in-loop evaluation
# TODO! add support for in-loop head training
# TODO! Create supported datasets with related transforms!


LOSS_NAME = "dino_loss"


class DINO(SSL):
    def __init__(self,
                 student_backbone: Union[nn.Module, str] = "resnet18",
                 student_head: Union[DINOProjectionHead, Tuple[int, int, int]] = (512, 512, 64),
                 teacher_head: Union[DINOProjectionHead, Tuple[int, int, int], Literal["same"]] = "same",
                 criterion: Union[nn.Module, Literal["dino_loss"]] = LOSS_NAME,
                 output_dim: Optional[int] = 2048,
                 warmup_teacher_temp_epochs: Optional[int] = 5,
                 momentum_bounds: Tuple[float, float] = (0.996, 1.0),
                 **kwargs: Unpack[AutoModuleKwargsNoNetCriterion]):
        """Initialize a DINO (self-distillation with no labels) model.

        Args:
            student_backbone (Union[nn.Module, str]): Either a nn.Module backbone or string name of a torchvision model
                to use as the student backbone. Defaults to "resnet18".
            student_head (Union[DINOProjectionHead, Tuple[int, int, int]]): Either a DINOProjectionHead or tuple of (hidden_dim, bottleneck_dim, out_dim)
                to construct the student head. Defaults to (512, 512, 64).
            teacher_head (Union[DINOProjectionHead, Tuple[int, int, int]]): Either a DINOProjectionHead or tuple of (hidden_dim, bottleneck_dim, out_dim)
                to construct the teacher head. Defaults to (512, 512, 64).
            criterion (Union[nn.Module, str]): Either a custom loss module or "dinoloss" to use the default DINO loss.
                Defaults to "dinoloss".
            output_dim (int): Output dimension of the projection heads. Only used if criterion="dinoloss"
                or using tuple configs for heads. Defaults to 2048.
            warmup_teacher_temp_epochs (Optional[int]): Number of warmup epochs for teacher temperature when using
                default DINO loss. Must be None if using custom criterion. Defaults to 5.
            momentum_bounds (Tuple[float, float]): Tuple of (min, max) values for the momentum teacher EMA updates.
                Defaults to (0.996, 1.0).
            **kwargs: Additional arguments passed to AutoModule base class.
        """

        if criterion != LOSS_NAME:
            assert warmup_teacher_temp_epochs is None, 'warmup_teacher_temp_epochs must be None if criterion module is provided'

            if output_dim is not None:
                assert isinstance(student_head, nn.Module) and isinstance(teacher_head, nn.Module), 'output_dim must only be provided if using custom head or criterion'

        super().__init__(
            criterion=criterion,
            default_criterion=(LOSS_NAME, DINOLoss, dict(
                output_dim=output_dim,
                warmup_teacher_temp_epochs=warmup_teacher_temp_epochs
            )),
            projection_head=student_head,
            default_projection_head=(DINOProjectionHead, dict(output_dim=output_dim)),
            backbone=student_backbone,
            copy_backbone_to_teacher=True,
            teacher_projection_head=teacher_head,
            **kwargs
        )

        self.momentum_bounds = momentum_bounds

    def shared_step(self, phase: Phase, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int):
        if phase == 'train':
            momentum = self.update_momentum(*self.momentum_bounds)

            views, targets = batch[0], batch[1]
            global_views = torch.cat(views[:2])
            local_views = torch.cat(views[2:])

            teacher_projections = self.forward_teacher(global_views)
            student_projections = torch.cat([
                self.forward_student(global_views),
                self.forward_student(local_views)
            ])

            return {
                "criterion_args": (teacher_projections, student_projections, self.current_epoch),
                "log_dict": {
                    "ema_momentum": momentum
                },
                "log_kwargs": dict(
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=len(targets),
                )
            }
        elif phase != 'val':
            raise NotImplementedError(f"Phase {phase} not implemented")

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: Union[int, float, None] = None,
        gradient_clip_algorithm: Union[str, None] = None,
    ) -> None:
        self.clip_gradients(
            optimizer=optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )

        self.projection_head.cancel_last_layer_gradients(self.current_epoch)
