from typing import Union, Tuple, List, Literal, Optional

import torch
from torch import Tensor
import torch.nn as nn

from lightly.loss import BarlowTwinsLoss
from lightly.models.modules import BarlowTwinsProjectionHead
from lightly.transforms import BYOLTransform

from autolightning.lm.self_supervised.lightly import SSL
from autolightning.types import Unpack, Phase, AutoModuleKwargsNoNetCriterion


LOSS_NAME = "barlowtwins_loss"


class BarlowTwins(SSL):
    def __init__(self,
        backbone: Union[nn.Module, str] = "resnet50",
        projection_head: Union[BarlowTwinsProjectionHead, Tuple[int, int]] = (2048, 8192),
        criterion: Union[nn.Module, Literal["barlowtwins_loss"]] = LOSS_NAME,
        output_dim: Optional[int] = 2048,
        **kwargs: Unpack[AutoModuleKwargsNoNetCriterion]
    ) -> None:
        # TODO!!!!!!!! gather_distributed
        super().__init__(
            criterion=criterion,
            default_criterion=(LOSS_NAME, BarlowTwinsLoss, dict(lambda_param=5e-3, gather_distributed=True)),
            projection_head=projection_head,
            default_projection_head=(BarlowTwinsProjectionHead, dict(output_dim=output_dim)),
            backbone=backbone,
            **kwargs
        )

    def shared_step(self, phase: Phase, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int) -> Tensor:
        assert phase == "train"

        # Forward pass and loss calculation.
        views, targets = batch[0], batch[1]
        z = self.forward_student(torch.cat(views))
        z0, z1 = z.chunk(len(views))

        return {
            "criterion_args": (z0, z1),
            "log_kwargs": dict(
                prog_bar=True,
                sync_dist=True,
                batch_size=len(targets),
            )
        }


# BarlowTwins uses same transform as BYOL.
transform = BYOLTransform()
