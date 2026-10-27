import argparse
import json
import os
from argparse import Namespace
from collections import defaultdict

import numpy as np
import torch

from Model.kan.utils import SYMBOLIC_LIB
from dataloader.dataloader import HUSTdata


def apply_symbolic_function(fun_name, input_val):
    eps = 1e-8
    if fun_name in ['arcsin', 'arccos', 'arctanh']:
        input_val = input_val.clamp(-1 + eps, 1 - eps)
    elif fun_name in ['log', 'sqrt', '1/sqrt(x)', '1/x', '1/x^2', '1/x^3', '1/x^4']:
        input_val = input_val.clamp_min(eps)

    fun = SYMBOLIC_LIB[fun_name][0]
    fun_val = fun(input_val)
    return torch.nan_to_num(fun_val, nan=0.0, posinf=1e6, neginf=-1e6)


def symbolic_formula(x, params_all_layers):
    layers = defaultdict(list)
    for params in params_all_layers:
        layers[params['layer']].append(params)

    h = x
    for layer_id in sorted(layers):
        params_list = layers[layer_id]
        output_indices = sorted({params['out'] for params in params_list})
        outputs = []
        for output_idx in output_indices:
            output = torch.zeros(h.shape[0], dtype=h.dtype, device=h.device)
            for params in params_list:
                if params['out'] != output_idx:
                    continue
                input_idx = params['in']
                if input_idx < 0 or input_idx >= h.shape[1]:
                    raise ValueError(
                        f'Layer {layer_id} input index {input_idx} is out of range '
                        f'for input width {h.shape[1]}'
                    )
                if params['best_func'] is None or params['params'] is None:
                    raise ValueError(
                        f'No symbolic fit for layer {layer_id}, '
                        f'edge {input_idx}->{output_idx}'
                    )
                a, b, c, d = params['params']
                transformed_input = a * h[:, input_idx] + b
                function_output = apply_symbolic_function(
                    params['best_func'], transformed_input
                )
                output = output + c * function_output + d
            outputs.append(output.reshape(-1, 1))
        if not outputs:
            raise ValueError(f'Layer {layer_id} has no valid outputs')
        h = torch.cat(outputs, dim=1)

    return h[:, 0] if h.shape[1] == 1 else h


def load_hust_test_data(data_root, args):
    test_ids = {
        '1-4', '1-8', '2-4', '2-8', '3-4', '3-8', '4-4', '4-8',
        '5-4', '5-7', '6-4', '6-8', '7-4', '7-8', '8-4', '8-8',
        '9-4', '9-8', '10-4', '10-8',
    }
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f'HUST data directory not found: {data_root}')

    dataset = HUSTdata(root=data_root, args=args)
    train_paths = []
    test_paths = []
    for filename in sorted(os.listdir(data_root)):
        file_path = os.path.join(data_root, filename)
        name, _ = os.path.splitext(filename)
        if name in test_ids:
            test_paths.append(file_path)
        else:
            train_paths.append(file_path)

    train_loader = dataset.read_all(specific_path_list=train_paths)
    test_loader = dataset.read_all(specific_path_list=test_paths)
    return test_loader['test_3']


def r2_score(y_true, y_pred):
    residual_sum = np.sum((y_true - y_pred) ** 2)
    total_sum = np.sum((y_true - np.mean(y_true)) ** 2)
    if total_sum == 0:
        raise ValueError('Cannot calculate R-squared for constant target values')
    return 1 - residual_sum / total_sum


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate symbolic-regression JSON results on the HUST test split.'
    )
    parser.add_argument('--json', required=True, help='Path to encoder symbolic results JSON.')
    parser.add_argument('--data-root', default='data/HUST data', help='HUST dataset directory.')
    parser.add_argument(
        '--output-dir',
        default='results_of_symbolic_eval',
        help='Directory used by the HUST data loader for temporary outputs.',
    )
    args = parser.parse_args()

    with open(args.json, 'r', encoding='utf-8') as file:
        symbolic_results = json.load(file)
    if not isinstance(symbolic_results, list):
        raise ValueError('Symbolic-regression JSON must contain a list of edge records')
    required_fields = {'layer', 'in', 'out', 'best_func', 'params'}
    for record_index, record in enumerate(symbolic_results):
        if not isinstance(record, dict) or not required_fields.issubset(record):
            raise ValueError(
                f'Symbolic-regression JSON record {record_index} is missing '
                f'required fields: {sorted(required_fields)}'
            )

    os.makedirs(args.output_dir, exist_ok=True)
    data_args = Namespace(
        normalization_method='min-max',
        log_dir='logging.txt',
        save_folder=args.output_dir,
        batch_size=512,
    )
    test_loader = load_hust_test_data(args.data_root, data_args)

    features = []
    labels = []
    for x, _, y, _ in test_loader:
        features.append(x[:, [4, 16]].numpy())
        labels.append(y.numpy())
    if not features:
        raise ValueError(f'No HUST test samples found under {args.data_root}')

    x_test = np.concatenate(features, axis=0)
    y_test = np.concatenate(labels, axis=0)
    predictions = symbolic_formula(
        torch.from_numpy(x_test),
        symbolic_results,
    ).numpy().reshape(-1, 1)

    print(f'Symbolic equation test R-squared: {r2_score(y_test, predictions):.4f}')


if __name__ == '__main__':
    main()
