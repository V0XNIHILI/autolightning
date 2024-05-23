from typing import Dict

import inspect
from functools import wraps

import lightning as L


def create_factory(cls, pre_applied_first_arg, return_annotation):
    # Get the signature of the __init__ method, excluding 'self'
    init_signature = inspect.signature(cls.__init__)
    parameters = list(init_signature.parameters.values())[2:]  # Exclude 'self' and 'cfg'
    
    # Create a new signature for the factory function
    new_signature = inspect.Signature(parameters, return_annotation=return_annotation)
    
    # Create the factory function
    @wraps(cls.__init__)
    def factory(*args, **kwargs):
        # Create an instance of the class with the pre-applied first argument
        return cls(pre_applied_first_arg, *args, **kwargs)
    
    # Apply the new signature to the factory function
    factory.__signature__ = new_signature
    
    return factory

def pre_cli(module, cfg: Dict):
    # Get the method resolution order (MRO) of the class
    mro = inspect.getmro(module)

    if L.LightningDataModule in mro:
        return_annotation = L.LightningDataModule
    else:
        return_annotation = L.LightningModule

    return create_factory(module, cfg, return_annotation)
