import torch
import torch.nn as nn
from mars.utils import dict2func
import numpy as np
from .configuration import Configuration
config = Configuration()
del Configuration
from mars.utils import dict2func

__all__ = ['DynamicsNet','ConstantDynamicsNet']

class DynamicsNet(nn.Module):
    """A pytorch based neural network"""

    def __init__(self, input_dim, layer_dims, activations, initializer):
        super(DynamicsNet, self).__init__()
        self.input_dim = input_dim
        self.num_layers = len(layer_dims)
        activation_dict = {'relu': torch.relu, 'tanh': torch.tanh, 'identity':nn.Identity()}
        self.activations = list(map(dict2func(activation_dict), activations))
        self.initializer = initializer
        self.output_dims = layer_dims

        for i in range(self.num_layers):
            if i == 0:
                layer_tmp = torch.nn.Linear(self.input_dim, self.output_dims[i], bias=True, dtype=config.ptdtype)
                self.initializer(layer_tmp.weight)
                setattr(self, 'layer_{}'.format(i), layer_tmp)
            else:
                layer_tmp = torch.nn.Linear(self.output_dims[i-1], self.output_dims[i], bias=True, dtype=config.ptdtype)
                self.initializer(layer_tmp.weight)
                setattr(self, 'layer_{}'.format(i), layer_tmp)
        
        
    def forward(self, points):
        """Build the evaluation graph."""
        if isinstance(points, np.ndarray):
            points = torch.tensor(points, dtype = config.ptdtype, device= config.device)
        net = points
        for i in range(self.num_layers):
            layer_tmp = getattr(self, 'layer_{}'.format(i))
            layer_output = layer_tmp(net)
            net = self.activations[i](layer_output)
        
        return net

class ConstantDynamicsNet(nn.Module):
    """A pytorch based neural network"""

    def __init__(self, output_dim, initializer):
        super(ConstantDynamicsNet, self).__init__()
        self.output_dim = output_dim
        self.initializer = initializer
        self.W = torch.nn.Parameter(torch.zeros([1, self.output_dim], dtype=config.ptdtype), requires_grad=True)

    def forward(self, points):

        return torch.broadcast_to(self.W, (points.shape[0], self.output_dim))