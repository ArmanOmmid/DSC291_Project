import __init__

import sys
import os
from collections import Counter

import torch
import torch.nn as nn
import torchvision

import src.utility.util as util
import src.arch as arch

def get_weight_initializer():
    def init_weights(module):
        # children_size = len([None for _ in module.children()]) # Its a non-container module if this is 0; but we don't need this
        # module_name = module.__class__.__name__ # For if you want to check which module this is 
        # list(module.parameters()) # This will list out the associated parameters. For non-containers, this is usually between 0-2 (weights and bias)
        invalid_layers = ["LayerNorm", "BatchNorm2d", "BatchNorm1d"]
        if module.__class__.__name__ in invalid_layers: return
        try:
            if hasattr(module, 'weight') and module.weight is not None and module.weight.requires_grad:
                torch.nn.init.xavier_uniform_(module.weight.data)
            if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                torch.nn.init.normal_(module.bias.data) # xavier not applicable for biases
        except Exception as E:
            print("Invalid Layer for Xavier (Please Register It): ", module.__class__.__name__)
            raise E
    return init_weights

def build_model(config, classes):

    class_count = classes if isinstance(classes, int) else len(classes)

                                   
    if config.model == 'dsc_arch':
        model = arch.dsc_arch.Smoother(num_classes=class_count, config=config)
    elif config.model == 'dsc_prototype':
        model = arch.dsc_prototype.Smoother(num_classes=class_count, config=config)
    elif config.model == 'dsc_margin':
        model = arch.dsc_margin.Smoother(num_classes=class_count, config=config)
    else:
        raise NotImplementedError("Model Architecture Not Found")

    # init_weights = get_weight_initializer()
    # model.apply(init_weights)

    return model
