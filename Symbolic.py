import argparse
import json
import os

import torch

from Model.eKAN_for_symbolic import KAN
from Model.eKAN_symbolic import symbolic_regression_single


def load_input_ranges(ranges_path):
    layer_input_ranges = {}
    layer_pair_ranges = {}
    if not os.path.exists(ranges_path):
        return layer_input_ranges, layer_pair_ranges

    with open(ranges_path, 'r', encoding='utf-8') as file:
        ranges_data = json.load(file)
    if not isinstance(ranges_data, list):
        raise ValueError(f'Expected a list of layer ranges in {ranges_path}')

    for item in ranges_data:
        layer_id = int(item['layer'])
        input_ranges = item.get('input_ranges') or []
        pair_ranges = item.get('pair_ranges') or []
        layer_input_ranges[layer_id] = {
            int(entry['in']): {
                'min': entry.get('min'),
                'p1': entry.get('p1'),
                'p99': entry.get('p99'),
                'max': entry.get('max'),
            }
            for entry in input_ranges
        }
        layer_pair_ranges[layer_id] = pair_ranges
    return layer_input_ranges, layer_pair_ranges


def main():
    parser = argparse.ArgumentParser(
        description='Fit symbolic functions to each edge in a trained two-input KAN.'
    )
    parser.add_argument(
        '--experiment-dir',
        required=True,
        help='Directory containing the KAN model.pth checkpoint.',
    )
    parser.add_argument(
        '--ranges',
        default=None,
        help='Optional JSON file containing per-layer input ranges.',
    )
    parser.add_argument(
        '--output',
        default=None,
        help='Output JSON path (defaults to encoder_symbolic_results.json in experiment-dir).',
    )
    parser.add_argument('--resolution', type=int, default=200)
    args = parser.parse_args()

    experiment_dir = os.path.abspath(args.experiment_dir)
    checkpoint_path = os.path.join(experiment_dir, 'model.pth')
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f'KAN checkpoint not found: {checkpoint_path}')

    ranges_path = args.ranges or os.path.join(
        experiment_dir, 'encoder_symbolic_input_ranges.json'
    )
    output_path = args.output or os.path.join(
        experiment_dir, 'encoder_symbolic_results.json'
    )

    model = KAN()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    layer_input_ranges, layer_pair_ranges = load_input_ranges(ranges_path)

    results = []
    for layer_idx, layer in enumerate(model.encoder.layers):
        for in_idx in range(layer.in_features):
            x_range = [-1.0, 1.0]
            previous_layer_ranges = layer_pair_ranges.get(layer_idx - 1, [])
            relevant_ranges = [
                item for item in previous_layer_ranges
                if int(item['out']) == in_idx
            ]
            lows = []
            highs = []
            for item in relevant_ranges:
                low = item.get('p1')
                high = item.get('p99')
                if low is None:
                    low = item.get('min')
                if high is None:
                    high = item.get('max')
                if low is not None and high is not None:
                    lows.append(float(low))
                    highs.append(float(high))

            if lows and highs:
                low = min(lows)
                high = max(highs)
                margin = max((high - low) * 0.05, abs(low) * 0.01, 0.01)
                x_range = [low - margin, high + margin]
            else:
                input_range = layer_input_ranges.get(layer_idx, {}).get(in_idx)
                if input_range is not None:
                    low = input_range['p1']
                    high = input_range['p99']
                    if low is None:
                        low = input_range['min']
                    if high is None:
                        high = input_range['max']
                    if low is not None and high is not None:
                        low = float(low)
                        high = float(high)
                        margin = max((high - low) * 0.05, abs(low) * 0.01, 0.01)
                        x_range = [low - margin, high + margin]

            for out_idx in range(layer.out_features):
                result = symbolic_regression_single(
                    layer,
                    in_idx,
                    out_idx,
                    x_range=x_range,
                    resolution=args.resolution,
                    verbose=True,
                )
                result['layer'] = layer_idx
                if isinstance(result['params'], torch.Tensor):
                    result['params'] = result['params'].detach().cpu().tolist()
                if isinstance(result['r2'], torch.Tensor):
                    result['r2'] = float(result['r2'].detach().cpu())
                results.append(result)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
    print(f'Symbolic regression results saved to {output_path}')


if __name__ == '__main__':
    main()
