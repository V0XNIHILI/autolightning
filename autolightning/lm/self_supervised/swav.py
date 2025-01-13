from typing import Union, Tuple, List, Literal, Optional

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F

from lightly.loss.swav_loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.models.modules.memory_bank import MemoryBankModule

from autolightning.lm.self_supervised.lightly import SSL
from autolightning.types import Unpack, Phase, AutoModuleKwargsNoNetCriterion


@torch.no_grad()
def _update_queue(
    projections: List[Tensor],
    queues: nn.ModuleList
):
    """Adds the high resolution projections to the queues and returns the queues."""

    if len(projections) != len(queues):
        raise ValueError(
            f"The number of queues ({len(queues)}) should be equal to the number of high "
            f"resolution inputs ({len(projections)})."
        )

    # Get the queue projections
    queue_projections = []
    for i in range(len(queues)):
        _, queue_proj = queues[i](projections[i], update=True)
        # Queue projections are in (num_ftrs X queue_length) shape, while the high res
        # projections are in (batch_size_per_device X num_ftrs). Swap the axes for interoperability.
        queue_proj = torch.permute(queue_proj, (1, 0))
        queue_projections.append(queue_proj)

    return queue_projections


class SwAV(SSL):
    def __init__(
        self,
        backbone: Union[nn.Module, str] = "resnet50",
        projection_head: Union[SwaVProjectionHead, Tuple[int, int]] = (2048, 2048),
        prototypes: Union[SwaVPrototypes, int] = 3000,
        criterion: Union[nn.Module, Literal["swav_loss"]] = "swav_loss",
        output_dim: Optional[int] = 2048,
        crop_counts: Tuple[int, int] = (2, 6),
        start_queue_at_epoch: int = 15,
        n_batches_in_queue: int = 15,
        **kwargs: Unpack[AutoModuleKwargsNoNetCriterion]
    ):
        # TODO!!!!!!!!!! sinkhorn_gather_distributed
        super().__init__(
            criterion=criterion,
            default_criterion=("swav_loss", SwaVLoss, dict(sinkhorn_gather_distributed=True)),
            projection_head=projection_head,
            default_projection_head=(SwaVProjectionHead, dict(output_dim=output_dim)),
            backbone=backbone,
            **kwargs
        )

        self.prototypes = SwaVPrototypes(output_dim, prototypes, n_steps_frozen_prototypes=1) if isinstance(prototypes, int) else prototypes

        # Use a queue for small batch sizes (<= 256).
        self.start_queue_at_epoch = start_queue_at_epoch
        self.n_batches_in_queue = n_batches_in_queue
        self.queues = nn.ModuleList(
            [
                MemoryBankModule(size=(n_batches_in_queue * kwargs['batch_size_per_device'], 128))
                for _ in range(crop_counts[0])
            ]
        )

    def project(self, x: Tensor) -> Tensor:
        x = self.projection_head(x)
        return F.normalize(x, dim=1, p=2)

    def shared_step(
        self, phase: Phase, batch: Tuple[List[Tensor], Tensor, List[str]], batch_idx: int
    ):
        assert phase == "train"

        # Normalize prototypes for unit sphere alignment.
        self.prototypes.normalize()

        multi_crops, targets = batch[0], batch[1]
        multi_crop_projections = [self.project(self.forward(crops)) for crops in multi_crops]

        queue_crop_logits = None

        with torch.no_grad():
            if self.current_epoch >= self.start_queue_at_epoch:
                queue_crop_projections = _update_queue(
                    projections=multi_crop_projections[:self.crop_counts[0]],
                    queues=self.queues,
                )
                if batch_idx > self.n_batches_in_queue:
                    queue_crop_logits = [self.prototypes(projections, step=self.current_epoch)
                                         for projections in queue_crop_projections]

        multi_crop_logits = [self.prototypes(projections, step=self.current_epoch)
                             for projections in multi_crop_projections]

        return {
            "criterion_args":
            (multi_crop_logits[:self.crop_counts[0]], multi_crop_logits[self.crop_counts[0]:], queue_crop_logits),
            "log_kwargs":
            dict(
                prog_bar=True,
                sync_dist=True,
                batch_size=len(targets),
            )
        }
