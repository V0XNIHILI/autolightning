from typing import Optional, Any, Union

import torch
import torch.nn as nn


class SoftDistillationLoss(nn.Module):
    def __init__(self, soft_target_loss_weight=0.5, ce_loss_weight=0.5, T=1.0, **kwargs):
        super().__init__()

        self.soft_target_loss_weight = soft_target_loss_weight
        self.ce_loss_weight = ce_loss_weight
        self.T = T
        self.ce_loss = nn.CrossEntropyLoss(**kwargs)

    def forward(self, student_logits, teacher_logits, labels):
        # Soften the student logits by applying softmax first and log() second
        soft_targets = nn.functional.softmax(teacher_logits / self.T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / self.T, dim=-1)

        # Calculate the soft targets loss. Scaled by T**2 as suggested by the authors of the paper "Distilling the knowledge in a neural network"
        soft_targets_loss = torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (self.T**2)

        # Calculate the true label loss
        label_loss = self.ce_loss(student_logits, labels)

        loss = self.soft_target_loss_weight * soft_targets_loss + self.ce_loss_weight * label_loss

        return loss


class CosineDistillationLoss(nn.Module):
    def __init__(self,
                 hidden_rep_loss_weight=0.5,
                 ce_loss_weight=0.5,
                 margin = 0.0,
                 label_smoothing = 0.0,
                 ignore_index = -100,
                 weight: Optional[torch.Tensor] = None,
                 size_average: Union[Any, None] = None,
                 reduce: Union[Any, None] = None,
                 reduction: str = 'mean'):
        super().__init__()

        self.hidden_rep_loss_weight = hidden_rep_loss_weight
        self.ce_loss_weight = ce_loss_weight

        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing, ignore_index=ignore_index, weight=weight, size_average=size_average, reduce=reduce, reduction=reduction)
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=margin, size_average=size_average, reduce=reduce, reduction=reduction)

    def forward(self, student_hidden_representation, teacher_hidden_representation, student_logits, labels):
        # Calculate the cosine loss. Target is a vector of ones. From the loss formula above we can see that is the case where loss minimization leads to cosine similarity increase.
        hidden_rep_loss = self.cosine_loss(student_hidden_representation, teacher_hidden_representation, torch.ones(student_hidden_representation.size(0)).to(student_hidden_representation.device))

        label_loss = self.ce_loss(student_logits, labels)

        loss = self.hidden_rep_loss_weight * hidden_rep_loss + self.ce_loss_weight * label_loss

        return loss
