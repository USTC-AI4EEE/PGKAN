import torch
from Model.kan.utils import SYMBOLIC_LIB, fit_params
import numpy as np

def symbolic_regression_single(layer, in_idx, out_idx, x_range=[-1,1], resolution=200, verbose=True):
    """
    Perform symbolic regression on one efficient_KAN activation specified by in_idx and out_idx.
    Return the best symbolic function, its parameters, and R-squared.
    """
    in_features = layer.in_features
    out_features = layer.out_features
    x = torch.linspace(x_range[0], x_range[1], resolution)
    x_input = torch.zeros(resolution, in_features)
    x_input[:, in_idx] = x
    with torch.no_grad():
        y = layer(x_input)[:, out_idx].cpu()
    best_r2 = -float('inf')
    best_name = None
    best_params = None
    for name, (fun, _) in SYMBOLIC_LIB.items():
        try:
            params, r2 = fit_params(x, y, fun, verbose=False)
            if r2 > best_r2:
                best_r2 = r2
                best_name = name
                best_params = params
        except Exception as e:
            continue
    if verbose:
        print(f"in {in_idx} -> out {out_idx}: best_func={best_name}, r2={best_r2:.4f}, params={best_params}")
    return {'in': in_idx, 'out': out_idx, 'best_func': best_name, 'params': best_params, 'r2': best_r2}
