import torch
import torch.nn as nn
import numpy as np
from torch.autograd import grad
from utils.util import AverageMeter, get_logger, eval_metrix
import os
from .efficient_KAN import KAN as eKAN

class KAN(nn.Module):
    def __init__(self, in_dim=2):
        super(KAN, self).__init__()
        self.encoder = eKAN(layers_hidden=[in_dim, 2, 1], grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        # self.encoder = eKAN(layers_hidden=[17, 60, 32], grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])
        # self.predictor = eKAN(layers_hidden=[32, 32, 1], grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0, base_activation=torch.nn.SiLU, grid_eps=0.02, grid_range=[-1, 1])

    def forward(self,x):
        x = self.encoder(x)
        # x = self.predictor(x)
        return x

