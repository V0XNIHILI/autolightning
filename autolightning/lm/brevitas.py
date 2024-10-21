from typing import Optional, List, Any, Dict, Union
from contextlib import nullcontext

import torch
import torch.nn as nn

from lightning.pytorch.utilities.types import STEP_OUTPUT

from brevitas.quant_tensor import QuantTensor
from brevitas.nn.mixin.parameter import WeightQuantType, BiasQuantType
from brevitas.nn.mixin.act import ActQuantType
from brevitas.graph.calibrate import bias_correction_mode, calibration_mode, norm_correction_mode

from brevitas_utils import create_qat_ready_model
from brevitas_utils.creation import create_quant_class

from . import Supervised, Classifier, Prototypical
from ..types import AutoModuleKwargs, Unpack


def __getitem__(self, indices):
    # Only allow indexing on QuantTensors with scalar scale
    if self.scale == None or self.scale.shape == torch.Size([]):
        return QuantTensor(self.value[indices], self.scale, self.zero_point, self.bit_width, self.signed, self.training)

    # Do not yet support indexing on QuantTensors with scale per channel, as it is not directly clear how to handle this
    raise RuntimeError("QuantTensor with scale of shape {} is not supported.".format(self.scale.shape))


def get_first_context(contexts_to_enter: List[str], contexts_exited: List[str]):
    for context in contexts_to_enter:
        if context not in contexts_exited:
            return context

    return None


def weight_quantizer(class_paths: List[str], init_args: Dict[str, Any]) -> WeightQuantType:
    return create_quant_class(class_paths, init_args)


def act_quantizer(class_paths: List[str], init_args: Dict[str, Any]) -> ActQuantType:
    return create_quant_class(class_paths, init_args)


def bias_quantizer(class_paths: List[str], init_args: Dict[str, Any]) -> BiasQuantType:
    return create_quant_class(class_paths, init_args)


class BrevitasMixin:
    # Using quoated types as the base class of these types are
    # dynamically generated and hard to type check using 
    # jsonargparse. I get this error mainly:
    # - 'Injector' can not resolve attribute '__origin__'

    def __init__(self,
                 weight_quant: Optional["WeightQuantType"] = None,
                 act_quant: Optional["ActQuantType"] = None,
                 bias_quant: Optional["BiasQuantType"] = None,
                 in_quant: Optional["ActQuantType"] = None,
                 out_quant: Optional["ActQuantType"] = None,
                 load_float_weights_into_model: bool = True,
                 remove_dropout_layers: bool = True,
                 fold_batch_norm_layers: bool = True,
                 allow_quant_tensor_slicing: bool = False,
                 calibrate: bool = False,
                 correct_biases: bool = False,
                 correct_norms: bool = False,
                 skip_modules: Optional[List[Union[type[nn.Module], str]]] = None,
                 limit_calibration_batches: Optional[int] = None,
                 **kwargs: Unpack[AutoModuleKwargs]):
        super().__init__(**kwargs)

        if allow_quant_tensor_slicing:
            QuantTensor.__getitem__ = __getitem__

        self.weight_quant = weight_quant
        self.act_quant = act_quant
        self.bias_quant = bias_quant
        self.in_quant = in_quant
        self.out_quant = out_quant
        self.load_float_weights_into_model = load_float_weights_into_model
        self.remove_dropout_layers = remove_dropout_layers
        self.fold_batch_norm_layers = fold_batch_norm_layers

        self.calibrate = calibrate
        self.correct_biases = correct_biases
        self.correct_norms = correct_norms
        self.skip_modules = skip_modules

        self.limit_calibration_batches = limit_calibration_batches

        self.prepare_model()

    def prepare_model(self):
        assert self.net != None, "Default model to quantize ('self.net') is not set."

        self.net = create_qat_ready_model(
            self.net,
            weight_quant_cfg=self.weight_quant,
            act_quant_cfg=self.act_quant,
            bias_quant_cfg=self.bias_quant,
            in_quant_cfg=self.in_quant,
            out_quant_cfg=self.out_quant,
            load_float_weights_into_model=self.load_float_weights_into_model,
            remove_dropout_layers=self.remove_dropout_layers,
            fold_batch_norm_layers=self.fold_batch_norm_layers,
            skip_modules=self.skip_modules
        )

        self.calibrate_context = calibration_mode(self.net) if self.calibrate else nullcontext()
        self.correct_biases_context = bias_correction_mode(self.net) if self.correct_biases else nullcontext()
        self.correct_norms_context = norm_correction_mode(self.net) if self.correct_norms else nullcontext()

        self.contexts_to_enter = []

        if self.calibrate:
            self.contexts_to_enter.append("calibrate_context")

        if self.correct_biases:
            self.contexts_to_enter.append("correct_biases_context")

        if self.correct_norms:
            self.contexts_to_enter.append("correct_norms_context")

        self.contexts_exited = []
        self.current_context = None

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        out = super().on_train_batch_start(batch, batch_idx)

        if self.current_context is None:
            context_name = get_first_context(self.contexts_to_enter, self.contexts_exited)

            if context_name is not None:
                context = getattr(self, context_name)
                context.__enter__()
                self.current_context = context

                print(f"Starting {context_name[:-8]} operation.")

        return out
    
    def exit_quant_context_if_exists(self):
        if self.current_context is not None:
            self.current_context.__exit__(None, None, None)
            self.current_context = None

            current_context_name = get_first_context(self.contexts_to_enter, self.contexts_exited)
            self.contexts_exited.append(current_context_name)

            print(f"Completed {current_context_name[:-8]} operation.")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if self.limit_calibration_batches is not None:
            if batch_idx == self.limit_calibration_batches:
                self.exit_quant_context_if_exists()

    def on_train_epoch_end(self) -> None:
        super().on_train_epoch_end()
        
        self.exit_quant_context_if_exists()

    def training_step(self, *args, **kwargs):
        output = super().training_step(*args, **kwargs)

        if self.current_context is None:
            return output
        
        return None


class BrevitasSupervised(BrevitasMixin, Supervised):
    pass


class BrevitasClassifier(BrevitasMixin, Classifier):
    pass


class BrevitasPrototypical(BrevitasMixin, Prototypical):
    pass
