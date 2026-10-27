import argparse
import os
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import shap
import torch
import matplotlib.pyplot as plt

from Model.PGKAN import Solution_u
from dataloader.dataloader import HUSTdata, MITdata, TJUdata, XJTUdata


# Update this base path when moving the datasets or experiment result folders.
BASE_DIR = r'./'
NAMES = [
    'voltage mean', 'voltage std', 'voltage kurtosis', 'voltage skewness',
    'CC Q', 'CC charge time', 'voltage slope', 'voltage entropy',
    'current mean', 'current std', 'current kurtosis', 'current skewness',
    'CV Q', 'CV charge time', 'current slope', 'current entropy'
]

DATASET_CONFIGS = {
    'MIT': {
        'root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-MIT_results degradation sensing with pde'),
        'legacy_root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-MIT_results rba with pde'),
        'bold': ['CC Q', 'CC charge time', 'voltage kurtosis'],
        'data_root': os.path.join(BASE_DIR, 'data', 'MIT data'),
    },
    'HUST': {
        'root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-HUST_results degradation sensing with pde'),
        'legacy_root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-HUST_results rba with pde'),
        'bold': ['CC Q', 'voltage entropy', 'voltage skewness'],
        'data_root': os.path.join(BASE_DIR, 'data', 'HUST data'),
    },
    'XJTU': {
        'root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-XJTU_results degradation sensing with pde'),
        'legacy_root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-XJTU_results rba with pde'),
        'bold': ['CV charge time', 'current slope'],
        'data_root': os.path.join(BASE_DIR, 'data', 'XJTU data'),
    },
    'TJU': {
        'root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-TJU_results degradation sensing with pde'),
        'legacy_root': os.path.join(BASE_DIR, 'discover_PDE', '0202_1436-TJU_results rba with pde'),
        'bold': ['CV charge time', 'CV Q'],
        'data_root': os.path.join(BASE_DIR, 'data', 'TJU data'),
    },
}


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ['solution_u', 'model', 'state_dict', 'model_state_dict']:
            if key in ckpt:
                v = ckpt[key]
                if isinstance(v, dict):
                    return v
        if any(k.startswith('encoder') or k.startswith('predictor') for k in ckpt.keys()):
            return ckpt
        if 'best_model' in ckpt and isinstance(ckpt['best_model'], dict):
            inner = ckpt['best_model']
            if 'solution_u' in inner:
                return inner['solution_u']
    return ckpt


def load_solution_u_from_checkpoint(model_path, device='cpu'):
    """Restore Solution_u from a checkpoint."""
    ckpt = torch.load(model_path, map_location=device)
    state = _extract_state_dict(ckpt)

    net = Solution_u().to(device)
    if isinstance(state, dict):
        try:
            net.load_state_dict(state, strict=True)
        except RuntimeError:
            net.load_state_dict(state, strict=False)
    else:
        raise TypeError(f'Unsupported checkpoint type: {type(state)} for {model_path}')

    net.eval()
    return net


def _make_dataset_args():
    return SimpleNamespace(
        normalization_method='min-max',
        batch_size=512,
        save_folder='.',
        log_dir=None,
        xjtu_batch='2C',
        tju_batch=0,
        tju_train_batch=-1,
        tju_test_batch=-1,
        in_same_batch=True,
    )


def _get_dataset_loader(dataset_name):
    dataset_name = dataset_name.upper()
    args = _make_dataset_args()
    if dataset_name == 'HUST':
        return HUSTdata(root=os.path.join(BASE_DIR, 'data', 'HUST data'), args=args)
    if dataset_name == 'MIT':
        return MITdata(root=os.path.join(BASE_DIR, 'data', 'MIT data'), args=args)
    if dataset_name == 'XJTU':
        return XJTUdata(root=os.path.join(BASE_DIR, 'data', 'XJTU data'), args=args)
    if dataset_name == 'TJU':
        return TJUdata(root=os.path.join(BASE_DIR, 'data', 'TJU data'), args=args)
    raise ValueError(f'Unsupported dataset: {dataset_name}')


def _pad_or_trim_features(X, in_dim=17):
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] >= in_dim:
        X = X[:, :in_dim]
    else:
        pad = np.zeros((X.shape[0], in_dim - X.shape[1]), dtype=np.float32)
        X = np.hstack([X, pad])
    return X.astype(np.float32)


def _load_real_background_inputs(dataset_name, in_dim=17, n_samples=None):
    dataset = _get_dataset_loader(dataset_name)
    if dataset_name.upper() == 'TJU':
        file_list = []
        for batch in getattr(dataset, 'batchs', []):
            batch_root = os.path.join(dataset.root, batch)
            if not os.path.isdir(batch_root):
                continue
            for file_name in sorted(os.listdir(batch_root)):
                file_path = os.path.join(batch_root, file_name)
                if file_name.lower().endswith('.csv'):
                    file_list.append(file_path)
        if not file_list:
            raise ValueError(f'No CSV files found for dataset {dataset_name}')
        loader = dataset.read_all(specific_path_list=file_list)
    else:
        loader = dataset.read_all()

    all_x = []
    for batch in loader.get('train_2', loader.get('test_3', [])):
        x1, x2, y1, y2 = batch
        all_x.append(x1)
    if not all_x:
        raise ValueError(f'No training inputs found for dataset {dataset_name}')
    X = torch.cat(all_x, dim=0).cpu().numpy()
    X = _pad_or_trim_features(X, in_dim=in_dim)
    if n_samples is not None:
        if X.shape[0] < n_samples:
            reps = int(np.ceil(n_samples / X.shape[0]))
            X = np.tile(X, (reps, 1))[:n_samples]
        return X[:n_samples].astype(np.float32)
    return X.astype(np.float32)


def _load_or_generate_inputs(folder, dataset_name=None, in_dim=17, n_samples=1024):
    if dataset_name is not None:
        try:
            return _load_real_background_inputs(dataset_name, in_dim=in_dim, n_samples=n_samples)
        except Exception:
            pass

    sample_x_path = os.path.join(folder, 'example_inputs.npy')
    if os.path.exists(sample_x_path):
        X = np.load(sample_x_path)
        X = _pad_or_trim_features(X, in_dim=in_dim)
        if X.shape[0] < n_samples:
            reps = int(np.ceil(n_samples / X.shape[0]))
            X = np.tile(X, (reps, 1))[:n_samples]
        return X[:n_samples].astype(np.float32)

    X = np.random.uniform(-1, 1, size=(n_samples, in_dim)).astype(np.float32)
    return X


def _compute_single_shap_importance(model, X_bg, X_test):
    explainer = shap.GradientExplainer(model, X_bg)
    shap_values = explainer.shap_values(X_test)

    arr = np.asarray(shap_values)
    if isinstance(shap_values, list):
        arr = np.asarray(shap_values[0])
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2 and arr.shape[1] > 16:
        arr = arr[:, :16]
    if arr.ndim == 1 and arr.size > 16:
        arr = arr[:16]

    if arr.ndim == 1:
        imp = np.abs(arr).astype(np.float64)
    elif arr.ndim == 2:
        imp = np.abs(arr).mean(axis=0).astype(np.float64)
    else:
        raise ValueError(f'Unexpected SHAP output shape: {arr.shape}')

    imp = imp[:16]
    if imp.size < 16:
        imp = np.pad(imp, (0, 16 - imp.size), mode='constant', constant_values=0.0)
    return imp


def calc_shap_importance_per_input(model_path, dataset_name='MIT', background_size=256, test_size=64, n_bootstrap=10):
    """Use real training data as the SHAP background and validate with bootstrap replications."""
    model = load_solution_u_from_checkpoint(model_path)
    X_full = _load_real_background_inputs(dataset_name, in_dim=17, n_samples=None)
    total = X_full.shape[0]
    if total == 0:
        raise ValueError(f'No real training samples for dataset {dataset_name}')

    rng = np.random.default_rng(0)
    test_idx = rng.choice(total, size=min(test_size, total), replace=False)
    X_test = torch.tensor(X_full[test_idx], dtype=torch.float32)

    boot_importances = []
    for _ in range(n_bootstrap):
        bg_idx = rng.choice(total, size=min(background_size, total), replace=True)
        X_bg = torch.tensor(X_full[bg_idx], dtype=torch.float32)
        imp = _compute_single_shap_importance(model, X_bg, X_test)
        boot_importances.append(imp)

    arr = np.stack(boot_importances, axis=0)
    mean_imp = arr.mean(axis=0)
    std_imp = arr.std(axis=0, ddof=1 if arr.shape[0] > 1 else 0)
    return mean_imp, std_imp


def calc_detailed_spline_importance(model_path, device='cpu'):
    net = load_solution_u_from_checkpoint(model_path, device=device)

    def _extract_five(layer):
        scaler = layer.spline_scaler.detach().cpu().numpy()
        sw = layer.spline_weight.detach().cpu().numpy()
        vals = [(sw[:, :, k + 1] + sw[:, :, k + 2]) / 2 for k in range(5)]
        five = np.stack(vals, axis=-1)
        return five * scaler[:, :, None]

    enc_fives = [_extract_five(l) for l in net.encoder.layers]
    pred_fives = [_extract_five(l) for l in net.predictor.layers]

    imp = pred_fives[-1].squeeze()
    if imp.ndim == 3:
        imp = np.sum(imp, axis=0)

    for five in reversed(pred_fives[:-1]):
        new_imp = [np.sum(five[:, j, :] * imp, axis=0) for j in range(five.shape[1])]
        imp = np.stack(new_imp, axis=0)

    for five in reversed(enc_fives):
        new_imp = []
        for j in range(five.shape[1]):
            if five.shape[0] == imp.shape[0]:
                new_v = np.sum(five[:, j, :] * imp, axis=0)
            else:
                new_v = np.sum(five[:, j, :], axis=0)
            new_imp.append(new_v)
        imp = np.stack(new_imp, axis=0)

    imp = np.abs(imp)
    result = imp.sum(axis=1)[:16]
    if result.max() > 0:
        result = result / result.max()
    return result.astype(np.float64)


def calc_feature_correlation_matrix(X, feat_dim=16):
    feats = X[:, :feat_dim]
    corr = np.corrcoef(feats, rowvar=False)
    return corr


def main():
    parser = argparse.ArgumentParser(description='Compare feature importance for the dataset-specific SR-PDE model.')
    parser.add_argument('--dataset', choices=['MIT', 'HUST', 'XJTU', 'TJU'], default='XJTU',
                        help='dataset that defines the result root, bold features, and SHAP background data')
    parser.add_argument('--root', default=None, help='Override the experiment root folder for the selected dataset')
    parser.add_argument('--showplt', action='store_true', help='Display the feature-importance plot')
    args = parser.parse_args()

    dataset_name = args.dataset.upper()
    config = DATASET_CONFIGS[dataset_name]
    root = args.root or config['root']
    if args.root is None and not os.path.exists(root) and os.path.exists(config['legacy_root']):
        root = config['legacy_root']
    bold = config['bold']

    if not os.path.exists(root):
        raise FileNotFoundError(f'Experiment root does not exist: {root}')

    excel_path = os.path.join(root, 'FI_PGKAN_importance_summary.xlsx')

    if os.path.exists(excel_path):
        print(f'Found existing {excel_path}, reading ...')
        df_shap = pd.read_excel(excel_path, sheet_name='GradientExplainer', index_col=0)
        df_spline = pd.read_excel(excel_path, sheet_name='Spline', index_col=0)
        try:
            df_corr = pd.read_excel(excel_path, sheet_name='Correlation', index_col=0)
        except Exception:
            df_corr = None
    else:
        subdirs = [d for d in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, d))]
        shap_results = {}
        spline_results = {}

        for sub in subdirs:
            sub_path = os.path.join(root, sub)
            model_path = os.path.join(sub_path, 'model.pth')
            if not os.path.exists(model_path):
                continue
            try:
                t0 = time.time()
                shap_imp, _ = calc_shap_importance_per_input(model_path, dataset_name=dataset_name)
                spline_imp = calc_detailed_spline_importance(model_path)
                elapsed = time.time() - t0

                shap_results[sub] = shap_imp
                spline_results[sub] = spline_imp
                print(f'{sub}: OK, time={elapsed:.2f}s')
            except Exception as e:
                print(f'{sub}: error -> {e}')

        if not shap_results:
            print('No valid result under current root.')
            return

        df_shap = pd.DataFrame(shap_results, index=NAMES)
        df_spline = pd.DataFrame(spline_results, index=NAMES)

        first_sub = next(iter(shap_results.keys()))
        first_folder = os.path.join(root, first_sub)
        X_corr = _load_or_generate_inputs(first_folder, dataset_name=dataset_name, in_dim=17, n_samples=1024)
        df_corr = pd.DataFrame(calc_feature_correlation_matrix(X_corr, feat_dim=16), index=NAMES, columns=NAMES)

        df_shap['sum'] = df_shap.sum(axis=1)
        df_spline['sum'] = df_spline.sum(axis=1)

        with pd.ExcelWriter(excel_path) as writer:
            df_shap.to_excel(writer, sheet_name='GradientExplainer')
            df_spline.to_excel(writer, sheet_name='Spline')
            df_corr.to_excel(writer, sheet_name='Correlation')

        print(f'Saved summary to {excel_path}')

    print('\n=== GradientExplainer (mean sum):', round(float(df_shap['sum'].mean()), 6))
    print('=== Spline (mean sum):', round(float(df_spline['sum'].mean()), 6))

    if df_corr is not None:
        print('\nTop feature correlations:')
        mask = np.triu(np.ones_like(df_corr.to_numpy(), dtype=bool), k=1)
        vals = df_corr.to_numpy()[mask]
        idx = np.abs(vals).argsort()[::-1][:10]
        pairs = []
        rows, cols = np.where(mask)
        for pos in idx:
            i, j = rows[pos], cols[pos]
            val = vals[pos]
            pairs.append((NAMES[i], NAMES[j], round(float(val), 3)))
        for a, b, val in pairs:
            print(f'  {a} <-> {b}: r = {val}')

    mean_shap = df_shap.drop(columns=['sum']).mean(axis=1).values
    mean_spline = df_spline.drop(columns=['sum']).mean(axis=1).values

    order = np.argsort(-mean_shap)
    names_sorted = [NAMES[i] for i in order]
    shap_sorted = mean_shap[order]
    spline_sorted = mean_spline[order]

    def _norm(x):
        m = x.max()
        return x / m if m > 0 else x

    shap_plot = _norm(shap_sorted)
    spline_plot = _norm(spline_sorted)

    names_inv = list(reversed(names_sorted))
    shap_plot_inv = shap_plot[::-1]
    spline_plot_inv = spline_plot[::-1]
    x_inv = np.arange(len(names_inv))

    plt.rcParams['text.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'
    plt.rcParams['axes.labelcolor'] = 'black'

    fig, axs = plt.subplots(1, 2, figsize=(8, 6), sharey=True)
    fig.subplots_adjust(left=0.25)

    axs[0].barh(x_inv, shap_plot_inv, color='#274690', alpha=0.9)
    axs[0].set_title('SHAP analysis', fontsize=14)
    axs[0].set_yticks(x_inv)
    axs[0].set_yticklabels(names_inv, fontsize=14)
    axs[0].xaxis.set_label_position('top')
    axs[0].xaxis.tick_top()
    axs[0].set_xlabel('variable importance', fontsize=12)
    axs[0].set_xlim(0, 1)
    axs[0].set_xticks(np.linspace(0, 1, 6))
    axs[0].tick_params(axis='x', labelsize=14)
    axs[0].tick_params(axis='y', labelsize=14)
    for lbl in axs[0].get_yticklabels():
        if lbl.get_text() in bold:
            lbl.set_fontweight('bold')

    axs[1].barh(x_inv, spline_plot_inv, color='#6A9CCB')
    axs[1].set_title('spline function analysis', fontsize=14)
    axs[1].set_yticks(x_inv)
    axs[1].set_yticklabels(names_inv, fontsize=14)
    axs[1].xaxis.set_label_position('top')
    axs[1].xaxis.tick_top()
    axs[1].set_xlabel('variable importance', fontsize=12)
    axs[1].set_xlim(0, 1)
    axs[1].set_xticks(np.linspace(0, 1, 6))
    axs[1].tick_params(axis='x', labelsize=14)
    axs[1].tick_params(axis='y', labelsize=14)
    for lbl in axs[1].get_yticklabels():
        if lbl.get_text() in bold:
            lbl.set_fontweight('bold')

    plt.tight_layout()
    if args.showplt:
        plt.show()
    else:
        plt.close(fig)


if __name__ == '__main__':
    main()
