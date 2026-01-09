import re
from typing import Optional, List, Any, Dict, Union
from contextlib import nullcontext

import torch
import torch.nn as nn

from lightning.pytorch.utilities.types import STEP_OUTPUT

from brevitas.inject import _InjectorType
from brevitas.graph.calibrate import (
    bias_correction_mode,
    calibration_mode,
    norm_correction_mode,
)

import brevitas
from brevitas_utils import (
    create_qat_ready_model,
    allow_quant_tensor_slicing as allow_slicing,
)
from brevitas_utils.creation import create_quantizer
from brevitas_utils.bias_correction import add_zero_bias_to_linear

from . import Supervised, Classifier, Prototypical, ICLClassifier
from ..utils import _import_module
from ..types import AutoModuleKwargs, Unpack


_BRACKET_RE = re.compile(r"^[^\[\]]*\[(\d+)\][^\[\]]*$")

def extract_bracket_index(name: str, raise_on_error: bool = True) -> Optional[int]:
    match = _BRACKET_RE.match(name)
    if match is None:
        if "[" in name or "]" in name:
            if raise_on_error:
                raise ValueError(f"Invalid bracket usage in '{name}'")
        return None
    return int(match.group(1))


def _get_first_context(contexts_to_enter: List[str], contexts_exited: List[str]):
    for context in contexts_to_enter:
        if context not in contexts_exited:
            return context

    return None


def quantizer(class_paths: List[str], init_args: Optional[Dict[str, Any]] = None) -> _InjectorType:
    return create_quantizer(class_paths, init_args)


class BrevitasMixin:
    # Using _InjectorType as quant type, since the base class
    # of these types are dynamically generated and hard to
    # type check using jsonargparse. I get this error mainly:
    # - 'Injector' can not resolve attribute '__origin__'

    def __init__(
        self,
        weight_quant: Optional[_InjectorType] = None,
        act_quant: Optional[_InjectorType] = None,
        bias_quant: Optional[_InjectorType] = None,
        in_quant: Optional[_InjectorType] = None,
        out_quant: Optional[_InjectorType] = None,
        load_float_weights_into_model: bool = True,
        remove_dropout_layers: bool = True,
        fold_batch_norm_layers: bool = True,
        allow_quant_tensor_slicing: bool = False,
        enable_brevitas_jit: bool = False,
        calibrate: bool = False,
        correct_biases: bool = False,
        correct_norms: bool = False,
        skip_modules: Optional[List[Union[type[nn.Module], str]]] = None,
        limit_calibration_batches: Optional[int] = None,
        net_attrs_to_quant: Optional[List[str]] = None,
        **kwargs: Unpack[AutoModuleKwargs],
    ):
        super().__init__(**kwargs)

        if enable_brevitas_jit:
            brevitas.config.JIT_ENABLED = 1

        if allow_quant_tensor_slicing:
            allow_slicing()

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

        if limit_calibration_batches is not None:
            assert self.calibrate or self.correct_biases or self.correct_norms, (
                "Calibration or bias correction or norm correction must be enabled to limit calibration batches."
            )

        if self.calibrate or self.correct_biases or self.correct_norms:
            assert self.limit_calibration_batches is not None, (
                "Limit calibration batches must be set if calibration or bias correction or norm correction is enabled."
            )
            assert self.limit_calibration_batches > 0, "Limit calibration batches must be greater than 0."


        if net_attrs_to_quant is None:
            net_attrs_to_quant = ['net']

        self.prepare_model(net_attrs_to_quant)

    def quantize_model(self, model: nn.Module, skip_modules: Optional[List[type[nn.Module]]] = None):
        model = create_qat_ready_model(
            model,
            weight_quant=self.weight_quant,
            act_quant=self.act_quant,
            bias_quant=self.bias_quant,
            in_quant=self.in_quant,
            out_quant=self.out_quant,
            load_float_weights_into_model=self.load_float_weights_into_model,
            remove_dropout_layers=self.remove_dropout_layers,
            fold_batch_norm_layers=self.fold_batch_norm_layers,
            skip_modules=skip_modules,
        )

        if self.correct_biases:
            model = add_zero_bias_to_linear(model)

        return model

    def prepare_model(self, net_attrs_to_quant: List[str]):
        if self.skip_modules is not None:
            skip_modules = [
                _import_module(module) if isinstance(module, str) else module for module in self.skip_modules
            ]
        else:
            skip_modules = None

        self.calibrate_context = []
        self.correct_biases_context = []
        self.correct_norms_context = []

        for net_attr in net_attrs_to_quant:
            net_attr_sub_index = extract_bracket_index(net_attr)

            if net_attr_sub_index is not None:
                base_net = getattr(self, net_attr.split('[')[0])
                sub_net = base_net[net_attr_sub_index]
                quant_net = self.quantize_model(sub_net, skip_modules=skip_modules)
                base_net[net_attr_sub_index] = quant_net
            else:
                net = getattr(self, net_attr)
                quant_net = self.quantize_model(net, skip_modules=skip_modules)
                setattr(self, net_attr, quant_net)

            self.calibrate_context.append(calibration_mode(quant_net) if self.calibrate else nullcontext())
            self.correct_biases_context.append(bias_correction_mode(quant_net) if self.correct_biases else nullcontext())
            self.correct_norms_context.append(norm_correction_mode(quant_net) if self.correct_norms else nullcontext())

        self.contexts_to_enter = []

        if self.calibrate:
            self.contexts_to_enter.append("calibrate_context")

        if self.correct_biases:
            self.contexts_to_enter.append("correct_biases_context")

        if self.correct_norms:
            self.contexts_to_enter.append("correct_norms_context")

        self.contexts_exited = []
        self.current_context = None

        self.no_grad_context = torch.no_grad()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        out = super().on_train_batch_start(batch, batch_idx)

        # In case a quantized checkpoint is reloaded, we do not want to start the calibration again
        if self.limit_calibration_batches is not None and batch_idx + 1 >= len(self.contexts_to_enter) * self.limit_calibration_batches:
            return out

        if self.current_context is None and self.limit_calibration_batches is not None:
            context_name = _get_first_context(self.contexts_to_enter, self.contexts_exited)

            if context_name is not None:
                if context_name == "calibrate_context":
                    # Only enter no_grad context if the calibrate context is entered
                    self.no_grad_context.__enter__()

                context = getattr(self, context_name)
                for ctx in context:
                    ctx.__enter__()
                self.current_context = context

                print(f"Starting {context_name[:-8]} operation...")

        return out

    def exit_quant_context_if_exists(self):
        if self.current_context is not None:
            for ctx in self.current_context:
                ctx.__exit__(None, None, None)
            self.current_context = None

            current_context_name = _get_first_context(self.contexts_to_enter, self.contexts_exited)
            self.contexts_exited.append(current_context_name)

            if current_context_name == "calibrate_context":
                self.no_grad_context.__exit__(None, None, None)

            print(f"Completed {current_context_name[:-8]} operation!")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        super().on_train_batch_end(outputs, batch, batch_idx)

        if self.limit_calibration_batches is not None:
            if (batch_idx + 1) % self.limit_calibration_batches == 0:
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


class BrevitasICLClassifier(BrevitasMixin, ICLClassifier):
    def __init__(self, **kwargs):
        if kwargs.get('net_attrs_to_quant', None) is None:
            kwargs['net_attrs_to_quant'] = ['net', 'sample_embedder']

            if kwargs.get('query_sample_embedder', None) is not None:
                kwargs['net_attrs_to_quant'].append('query_sample_embedder')

        super().__init__(**kwargs)

