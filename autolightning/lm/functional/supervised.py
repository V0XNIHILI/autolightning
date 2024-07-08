import torch.nn as nn

from autolightning import AutoModule

from torch_mate.utils import calc_accuracy


def compute_loss(criterion: nn.Module, output, target):
    if isinstance(output, tuple):
        return criterion(*output, target)
    else:
        return criterion(output, target)


def process_supervised_batch(model: nn.Module, batch, criterion: nn.Module):
    x, y = batch

    output = model(x)
    loss = compute_loss(criterion, output, y)

    return output, loss


def shared_step(module: AutoModule, batch, batch_idx, stage: str):
    output, loss = process_supervised_batch(module, batch, module.criteria)

    prog_bar = stage == 'val'

    module.log(module.log_key(stage, 'loss'), loss, prog_bar=prog_bar)

    if module.hparams.learner.get("cfg", {}).get("classification", False) == True:
        if "topk" in module.hparams.learner["cfg"]:
            for i, k in enumerate(module.hparams.learner["cfg"]["topk"]):
                module.log(module.log_key(stage, 'accuracy', f"@{k}"), calc_accuracy(output, batch[1], k), prog_bar=(i == 0 and prog_bar))
        else:
            module.log(module.log_key(stage, 'accuracy'), calc_accuracy(output, batch[1]), prog_bar=prog_bar)

    return loss
