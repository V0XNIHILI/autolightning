from typing import Union, Tuple, List, Callable, Dict, Any, Literal, Optional
import copy

import torch.nn as nn
from torch import Tensor
import torchvision

from lightly.models.utils import get_weight_decay_parameters, activate_requires_grad, deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule

from autolightning import AutoModule
from autolightning.types import Unpack, AutoModuleKwargsNoNetCriterion


class SSL(AutoModule):
    def __init__(self,
                 criterion: Union[nn.Module, str],
                 default_criterion: Tuple[str, Callable[..., nn.Module], Dict[str, Any]],
                 projection_head: Union[nn.Module, List[int]],
                 default_projection_head: Tuple[Callable[..., nn.Module], Dict[str, Any]],
                 backbone: Union[nn.Module, str] = "resnet50",
                 copy_backbone_to_teacher: bool = False,
                 teacher_projection_head: Optional[Union[nn.Module, List[int], Literal["same"]]] = None,
                 disable_student_grad: bool = False,
                 **kwargs: Unpack[AutoModuleKwargsNoNetCriterion]):
        criterion_name, criterion_class, criterion_kwargs = default_criterion

        if criterion == criterion_name:
            criterion = criterion_class(**criterion_kwargs)

        super().__init__(net=None, criterion=criterion, **kwargs)

        self.disable_student_grad = disable_student_grad

        if not isinstance(backbone, nn.Module):
            resnet = getattr(torchvision.models, backbone)(pretrained=False)
            backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove final classification layer

        self.backbone = backbone
        self.teacher_backbone = copy.deepcopy(backbone) if copy_backbone_to_teacher else None

        head_class, head_kwargs = default_projection_head

        if type(projection_head) is list or type(projection_head) is tuple:
            self.projection_head = head_class(*projection_head, **head_kwargs)
        else:
            self.projection_head = projection_head

        if teacher_projection_head == "same":
            self.teacher_projection_projehead = head_class(*projection_head, **head_kwargs)
        elif type(teacher_projection_head) is list:
            self.teacher_projection_head = head_class(*teacher_projection_head, **head_kwargs)
        else:
            self.teacher_projection_head = teacher_projection_head

    def on_train_start(self) -> None:
        if self.disable_student_grad:
            deactivate_requires_grad(self.backbone)
            deactivate_requires_grad(self.projection_head)
        else:
            super().on_train_start()

    def on_train_end(self) -> None:
        if self.disable_student_grad:
            activate_requires_grad(self.backbone)
            activate_requires_grad(self.projection_head)
        else:
            super().on_train_end()

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x).flatten(start_dim=1)

        return y
    
    def forward_student(self, x: Tensor) -> Tensor:
        features = self.forward(x)
        projections = self.projection_head(features)

        return projections

    def forward_teacher(self, x):
        features = self.teacher_backbone(x).flatten(start_dim=1)
        projections = self.teacher_projection_head(features)

        return projections
    
    def update_momentum(self, start_value: float, end_value: float):
        momentum = cosine_schedule(
            step=self.trainer.global_step,
            max_steps=self.trainer.estimated_stepping_batches,
            start_value=start_value,
            end_value=end_value
        )

        update_momentum(self.backbone, self.teacher_backbone, m=momentum)
        update_momentum(self.projection_head, self.teacher_projection_head, m=momentum)

        return momentum

    def parameters_for_optimizer(self, recurse: bool = True):
        params, params_no_weight_decay = get_weight_decay_parameters(
            [self.backbone, self.projection_head]
        )
        return [
            {"name": "params_weight_decay", "params": params},
            {"name": "params_no_weight_decay", "params": params_no_weight_decay, "weight_decay": 0.0}
        ]
