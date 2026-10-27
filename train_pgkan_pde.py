"""Train PDE-discovery models on one selected battery dataset using CPU."""

import torch
import numpy as np
import os
import argparse
import pandas as pd
from datetime import datetime
from Model.PGKAN import PINN_PDE_Discovery
# from Model.pde_ablation import PINN_PDE_Discovery
from dataloader.dataloader import XJTUdata, HUSTdata, MITdata, TJUdata
from result_analysis.XJTU_merged_results import Results as XJTUResults
from result_analysis.TJU_merged_results import Results as TJUResults
from result_analysis.HUST_results import Results as HUSTResults
from result_analysis.MIT_results import Results as MITResults


# GPU device will be set in each training function
# No need to set globally when using multiprocessing

def get_xjtu_args():
    """XJTU dataset parameters"""
    parser = argparse.ArgumentParser(description='PINN with PDE Discovery for XJTU')
    parser.add_argument('--dataset', choices=['XJTU', 'TJU', 'HUST', 'MIT'], default='XJTU')
    parser.add_argument('--experiments', type=int, default=10, help='number of repeated experiments')
    
    # Data related
    parser.add_argument('--data', type=str, default='XJTU', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max or z-score')
    parser.add_argument('--xjtu_batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    
    # Training scheduler
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=0.001, help='learning rate for lambda_w')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iterations per epoch')
    
    # Model related
    parser.add_argument('--F_layers_num', type=int, default=3, help='unused for PDE discovery')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='unused for PDE discovery')
    
    # PDE discovery parameters
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order for library')
    parser.add_argument('--l1_coef', type=float, default=1e-6, help='L1 regularization for lambda_w (auxiliary penalty in SINDy mode)')
    parser.add_argument('--reg_method', type=str, default='L1', choices=['L1', 'L0', 'ElasticNet', 'SINDy', 'MLP'],
                        help="regularization for dynamics: 'L1'=L1 sparse penalty, 'L0'=L0 sparsity (continuous relaxation + optional hard threshold), 'ElasticNet'=L1+L2 penalty, 'SINDy'=sequential hard thresholding (STLSQ), 'MLP'=neural network dynamical_F")
    parser.add_argument('--l0_coef', type=float, default=1e-6, help='L0 regularization coefficient (L0 mode)')
    parser.add_argument('--l0_beta', type=float, default=100.0, help='L0 relaxation steepness: 2*sigmoid(beta*|lambda|)-1')
    parser.add_argument('--l0_threshold', type=float, default=0.0, help='L0 hard-threshold: |lambda|<thr zeroed out every l0_threshold_freq steps (0=disabled)')
    parser.add_argument('--l0_threshold_freq', type=int, default=1, help='L0 hard-thresholding frequency (every N iterations)')
    parser.add_argument('--en_l1_coef', type=float, default=1e-6, help='ElasticNet L1 coefficient')
    parser.add_argument('--en_l2_coef', type=float, default=1e-6, help='ElasticNet L2 coefficient')
    parser.add_argument('--sindy_threshold', type=float, default=1e-2, help='SINDy threshold: coefficients below this are zeroed out')
    parser.add_argument('--sindy_threshold_freq', type=int, default=1, help='SINDy thresholding frequency (every N iterations)')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=0.7, help='weight for PDE loss')
    parser.add_argument('--beta', type=float, default=0.2, help='weight for physics constraint')
    parser.add_argument('--ds_eta', type=float, default=0.001, help='degradation-sensing weight scale')
    
    # Saving
    parser.add_argument('--save_folder', type=str, default='results/XJTU_PDE', help='save folder')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log file name')
    parser.add_argument('--gpu', type=str, default='2', help='gpu id')
    
    return parser.parse_args()

def get_tju_args():
    """TJU dataset parameters"""
    parser = argparse.ArgumentParser(description='PINN with PDE Discovery for TJU')
    parser.add_argument('--dataset', choices=['XJTU', 'TJU', 'HUST', 'MIT'], default='TJU')
    parser.add_argument('--experiments', type=int, default=10, help='number of repeated experiments')
    
    parser.add_argument('--data', type=str, default='TJU', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max or z-score')
    parser.add_argument('--in_same_batch', type=bool, default=True)
    parser.add_argument('--tju_batch', type=int, default=0, choices=[0,1,2])
    
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=0.002, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=0.01, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=0.0002, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=0.001, help='learning rate for lambda_w')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iterations per epoch')
    
    parser.add_argument('--F_layers_num', type=int, default=3, help='layers num of dynamical_F (MLP mode)')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='hidden dim of dynamical_F (MLP mode)')
    
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order for library')
    parser.add_argument('--l1_coef', type=float, default=1e-6, help='L1 regularization for lambda_w (auxiliary penalty in SINDy mode)')
    parser.add_argument('--reg_method', type=str, default='L1', choices=['L1', 'L0', 'ElasticNet', 'SINDy', 'MLP'],
                        help="regularization for dynamics: 'L1'=L1 sparse penalty, 'L0'=L0 sparsity (continuous relaxation + optional hard threshold), 'ElasticNet'=L1+L2 penalty, 'SINDy'=sequential hard thresholding (STLSQ), 'MLP'=neural network dynamical_F")
    parser.add_argument('--l0_coef', type=float, default=1e-6, help='L0 regularization coefficient (L0 mode)')
    parser.add_argument('--l0_beta', type=float, default=100.0, help='L0 relaxation steepness: 2*sigmoid(beta*|lambda|)-1')
    parser.add_argument('--l0_threshold', type=float, default=0.0, help='L0 hard-threshold: |lambda|<thr zeroed out every l0_threshold_freq steps (0=disabled)')
    parser.add_argument('--l0_threshold_freq', type=int, default=1, help='L0 hard-thresholding frequency (every N iterations)')
    parser.add_argument('--en_l1_coef', type=float, default=1e-6, help='ElasticNet L1 coefficient')
    parser.add_argument('--en_l2_coef', type=float, default=1e-6, help='ElasticNet L2 coefficient')
    parser.add_argument('--sindy_threshold', type=float, default=1e-2, help='SINDy threshold: coefficients below this are zeroed out')
    parser.add_argument('--sindy_threshold_freq', type=int, default=1, help='SINDy thresholding frequency (every N iterations)')
    
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for PDE loss')
    parser.add_argument('--beta', type=float, default=0.05, help='weight for physics constraint')
    parser.add_argument('--ds_eta', type=float, default=0.01, help='degradation-sensing weight scale')
    
    parser.add_argument('--save_folder', type=str, default='results/TJU_PDE', help='save folder')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log file name')
    parser.add_argument('--gpu', type=str, default='1', help='gpu id')
    
    return parser.parse_args()

def get_hust_args():
    """HUST dataset parameters"""
    parser = argparse.ArgumentParser(description='PINN with PDE Discovery for HUST')
    parser.add_argument('--dataset', choices=['XJTU', 'TJU', 'HUST', 'MIT'], default='HUST')
    parser.add_argument('--experiments', type=int, default=10, help='number of repeated experiments')
    
    parser.add_argument('--data', type=str, default='HUST', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max or z-score')
    
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=5e-4, help='learning rate for lambda_w')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iterations per epoch')
    
    parser.add_argument('--F_layers_num', type=int, default=3, help='unused for PDE discovery')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='unused for PDE discovery')
    
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order for library')
    parser.add_argument('--l1_coef', type=float, default=1e-6, help='L1 regularization for lambda_w (auxiliary penalty in SINDy mode)')
    parser.add_argument('--reg_method', type=str, default='L1', choices=['L1', 'L0', 'ElasticNet', 'SINDy', 'MLP'],
                        help="regularization for dynamics: 'L1'=L1 sparse penalty, 'L0'=L0 sparsity (continuous relaxation + optional hard threshold), 'ElasticNet'=L1+L2 penalty, 'SINDy'=sequential hard thresholding (STLSQ), 'MLP'=neural network dynamical_F")
    parser.add_argument('--l0_coef', type=float, default=1e-6, help='L0 regularization coefficient (L0 mode)')
    parser.add_argument('--l0_beta', type=float, default=100.0, help='L0 relaxation steepness: 2*sigmoid(beta*|lambda|)-1')
    parser.add_argument('--l0_threshold', type=float, default=0.0, help='L0 hard-threshold: |lambda|<thr zeroed out every l0_threshold_freq steps (0=disabled)')
    parser.add_argument('--l0_threshold_freq', type=int, default=1, help='L0 hard-thresholding frequency (every N iterations)')
    parser.add_argument('--en_l1_coef', type=float, default=1e-6, help='ElasticNet L1 coefficient')
    parser.add_argument('--en_l2_coef', type=float, default=1e-6, help='ElasticNet L2 coefficient')
    parser.add_argument('--sindy_threshold', type=float, default=1e-2, help='SINDy threshold: coefficients below this are zeroed out')
    parser.add_argument('--sindy_threshold_freq', type=int, default=1, help='SINDy thresholding frequency (every N iterations)')
    
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for PDE loss')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for physics constraint')
    parser.add_argument('--ds_eta', type=float, default=0.001, help='degradation-sensing weight scale')
    
    parser.add_argument('--save_folder', type=str, default='results/HUST_PDE', help='save folder')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log file name')
    parser.add_argument('--gpu', type=str, default='0', help='gpu id')
    
    return parser.parse_args()

def get_mit_args():
    """MIT dataset parameters"""
    parser = argparse.ArgumentParser(description='PINN with PDE Discovery for MIT')
    parser.add_argument('--dataset', choices=['XJTU', 'TJU', 'HUST', 'MIT'], default='MIT')
    parser.add_argument('--experiments', type=int, default=10, help='number of repeated experiments')
    
    parser.add_argument('--data', type=str, default='MIT', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max or z-score')
    
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=10, help='early stopping patience')
    parser.add_argument('--warmup_epochs', type=int, default=30, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=2e-3, help='warmup learning rate')
    parser.add_argument('--lr', type=float, default=1e-2, help='base learning rate')
    parser.add_argument('--final_lr', type=float, default=2e-4, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate for lambda_w')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iterations per epoch')
    
    parser.add_argument('--F_layers_num', type=int, default=3, help='layers num of dynamical_F (MLP mode)')
    parser.add_argument('--F_hidden_dim', type=int, default=60, help='hidden dim of dynamical_F (MLP mode)')
    
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order for library')
    parser.add_argument('--l1_coef', type=float, default=1e-6, help='L1 regularization for lambda_w (auxiliary penalty in SINDy mode)')
    parser.add_argument('--reg_method', type=str, default='L1', choices=['L1', 'L0', 'ElasticNet', 'SINDy', 'MLP'],
                        help="regularization for dynamics: 'L1'=L1 sparse penalty, 'L0'=L0 sparsity (continuous relaxation + optional hard threshold), 'ElasticNet'=L1+L2 penalty, 'SINDy'=sequential hard thresholding (STLSQ), 'MLP'=neural network dynamical_F")
    parser.add_argument('--l0_coef', type=float, default=1e-6, help='L0 regularization coefficient (L0 mode)')
    parser.add_argument('--l0_beta', type=float, default=100.0, help='L0 relaxation steepness: 2*sigmoid(beta*|lambda|)-1')
    parser.add_argument('--l0_threshold', type=float, default=0.0, help='L0 hard-threshold: |lambda|<thr zeroed out every l0_threshold_freq steps (0=disabled)')
    parser.add_argument('--l0_threshold_freq', type=int, default=1, help='L0 hard-thresholding frequency (every N iterations)')
    parser.add_argument('--en_l1_coef', type=float, default=1e-6, help='ElasticNet L1 coefficient')
    parser.add_argument('--en_l2_coef', type=float, default=1e-6, help='ElasticNet L2 coefficient')
    parser.add_argument('--sindy_threshold', type=float, default=1e-2, help='SINDy threshold: coefficients below this are zeroed out')
    parser.add_argument('--sindy_threshold_freq', type=int, default=1, help='SINDy thresholding frequency (every N iterations)')
    
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for PDE loss')
    parser.add_argument('--beta', type=float, default=0.05, help='weight for physics constraint')
    parser.add_argument('--ds_eta', type=float, default=0.01, help='degradation-sensing weight scale')
    
    parser.add_argument('--save_folder', type=str, default='results/MIT_PDE', help='save folder')
    parser.add_argument('--log_dir', type=str, default='logging.txt', help='log file name')
    parser.add_argument('--gpu', type=str, default='3', help='gpu id')
    
    return parser.parse_args()

def get_args():
    parser = argparse.ArgumentParser(description='PINN with PDE Discovery for Battery SOH Prediction')
    
    # Data related
    parser.add_argument('--data', type=str, default='XJTU', choices=['XJTU', 'HUST', 'MIT', 'TJU'])
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max or z-score')
    
    # XJTU data specific
    parser.add_argument('--xjtu_batch', type=str, default='2C', choices=['2C','3C','R2.5','R3','RW','satellite'])
    
    # TJU data specific
    parser.add_argument('--in_same_batch', type=bool, default=True)
    parser.add_argument('--tju_batch', type=int, default=0, choices=[0,1,2])
    
    # Training scheduler
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for solution_u')
    parser.add_argument('--warmup_epochs', type=int, default=10, help='warmup epochs')
    parser.add_argument('--warmup_lr', type=float, default=5e-4, help='warmup learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='final learning rate')
    parser.add_argument('--lr_F', type=float, default=1e-3, help='learning rate for lambda_w')
    parser.add_argument('--iter_per_epoch', type=int, default=1, help='iterations per epoch')
    
    # PDE discovery parameters
    parser.add_argument('--poly_order', type=int, default=2, help='polynomial order for library construction')
    parser.add_argument('--l1_coef', type=float, default=1e-6, help='L1 regularization for lambda_w sparsity (auxiliary penalty in SINDy mode)')
    parser.add_argument('--reg_method', type=str, default='L1', choices=['L1', 'L0', 'ElasticNet', 'SINDy', 'MLP'],
                        help="regularization for dynamics: 'L1'=L1 sparse penalty, 'L0'=L0 sparsity (continuous relaxation + optional hard threshold), 'ElasticNet'=L1+L2 penalty, 'SINDy'=sequential hard thresholding (STLSQ), 'MLP'=neural network dynamical_F")
    parser.add_argument('--l0_coef', type=float, default=1e-6, help='L0 regularization coefficient (L0 mode)')
    parser.add_argument('--l0_beta', type=float, default=100.0, help='L0 relaxation steepness: 2*sigmoid(beta*|lambda|)-1')
    parser.add_argument('--l0_threshold', type=float, default=0.0, help='L0 hard-threshold: |lambda|<thr zeroed out every l0_threshold_freq steps (0=disabled)')
    parser.add_argument('--l0_threshold_freq', type=int, default=1, help='L0 hard-thresholding frequency (every N iterations)')
    parser.add_argument('--en_l1_coef', type=float, default=1e-6, help='ElasticNet L1 coefficient')
    parser.add_argument('--en_l2_coef', type=float, default=1e-6, help='ElasticNet L2 coefficient')
    parser.add_argument('--sindy_threshold', type=float, default=1e-2, help='SINDy threshold: coefficients below this are zeroed out')
    parser.add_argument('--sindy_threshold_freq', type=int, default=1, help='SINDy thresholding frequency (every N iterations)')
    
    # Loss weights
    parser.add_argument('--alpha', type=float, default=1.0, help='weight for PDE loss')
    parser.add_argument('--beta', type=float, default=1.0, help='weight for physics constraint loss')
    
    # Degradation-sensing parameters
    parser.add_argument('--ds_eta', type=float, default=0.99, help='degradation-sensing weight scale')
    
    # Training control
    parser.add_argument('--early_stop', type=int, default=20, help='early stopping patience')
    
    # Saving
    parser.add_argument('--save_folder', type=str, default='./results/pde_discovery_test', help='folder to save results')
    parser.add_argument('--log_dir', type=str, default='training.log', help='log file name')
    
def get_args():
    """General arguments (kept for backward compatibility)"""
    return get_xjtu_args()


def load_xjtu_data(args):
    """Load XJTU dataset (all batches merged)"""
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    files = os.listdir(root)
    files.sort()
    for file in files:
        if '4' in file or '8' in file:
            test_list.append(os.path.join(root, file))
        else:
            train_list.append(os.path.join(root, file))

    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3']
    }
    return dataloader


def load_tju_data(args):
    """Load TJU dataset (all batches merged)"""
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []
    mod = [(5,9),(4,8),(5,9)]
    batchs = os.listdir(root)
    batchs.sort()
    for batch_idx, batch in enumerate(batchs):
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        files.sort()
        for i, f in enumerate(files):
            id = i + 1
            if id % 10 == 5 or id % 10 == 9:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    dataloader = {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3']
    }
    return dataloader


def load_hust_data(args):
    """Load HUST dataset"""
    test_id = ['1-4','1-8','2-4','2-8','3-4','3-8','4-4','4-8',
               '5-4','5-7','6-4','6-8','7-4','7-8','8-4','8-8',
               '9-4','9-8','10-4','10-8']
    data = HUSTdata(root='data/HUST data', args=args)
    train_list = []
    test_list = []
    files = os.listdir('data/HUST data')
    files.sort()
    for f in files:
        if f[:-4] in test_id:
            test_list.append(f'data/HUST data/{f}')
        else:
            train_list.append(f'data/HUST data/{f}')

    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3']
    }
    return dataloader


def load_mit_data(args):
    """Load MIT dataset"""
    root = 'data/MIT data'
    train_list = []
    test_list = []
    for batch in ['2017-05-12','2017-06-30','2018-04-12']:
        batch_root = os.path.join(root, batch)
        files = os.listdir(batch_root)
        files.sort()
        for f in files:
            id = int(f.split('-')[-1].split('.')[0])
            if id % 5 == 0:
                test_list.append(os.path.join(batch_root, f))
            else:
                train_list.append(os.path.join(batch_root, f))
    
    data = MITdata(root=root, args=args)
    trainloader = data.read_all(specific_path_list=train_list)
    testloader = data.read_all(specific_path_list=test_list)
    dataloader = {
        'train': trainloader['train_2'],
        'valid': trainloader['valid_2'],
        'test': testloader['test_3']
    }
    return dataloader


# Global root folder with timestamp
root0 = f"results/{datetime.now().strftime('%m%d_%H%M')}-"


def train_xjtu(device_name='cpu'):
    """Train the XJTU dataset on the selected device."""
    device = torch.device(device_name)
    print("=" * 50)
    print(f"Starting XJTU training (PDE discovery) on {device}...")
    print("=" * 50)
    
    root = f"{root0}XJTU_results ds with pde/"
    args = get_xjtu_args()
    args.device = str(device)
    args.gpu = str(device)
    
    for e in range(args.experiments):
        save_folder = f'{root}Experiment{e + 1}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        setattr(args, 'save_folder', save_folder)
        
        dataloader = load_xjtu_data(args)
        pinn = PINN_PDE_Discovery(args).to(device)
        pinn.Train(trainloader=dataloader['train'], 
                  validloader=dataloader['valid'], 
                  testloader=dataloader['test'])
    
    # Organize results
    xlsx_path = f'{root}XJTU_results.xlsx'
    writer = pd.ExcelWriter(xlsx_path)
    results = XJTUResults(root)
    df_mean1 = results.get_battery_mean(args.experiments)
    df_mean2 = results.get_experiments_mean(args.experiments)
    df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
    
    # Add average row for 10 experiments
    df_mean2_with_avg = df_mean2.copy()
    avg_row = df_mean2.mean(axis=0)
    avg_row.name = 'average'
    df_mean2_with_avg = df_mean2_with_avg.append(avg_row)
    df_mean2_with_avg.to_excel(writer, sheet_name='experiment_mean_0', index=True)
    
    print(f"XJTU MAPE: {df_mean2['MAPE'].mean():.4f}")
    print(f"XJTU RMSE: {df_mean2['RMSE'].mean():.6f}")
    writer.close()
    
    # Copy discovered equation to root folder
    eq_file = os.path.join(f'{root}Experiment1', 'discovered_equation.txt')
    if os.path.exists(eq_file):
        import shutil
        shutil.copy(eq_file, os.path.join(root, 'discovered_equation_example.txt'))
    
    print(f"XJTU training completed on {device}.")


def train_tju(device_name='cpu'):
    """Train the TJU dataset on the selected device."""
    device = torch.device(device_name)
    print("=" * 50)
    print(f"Starting TJU training (PDE discovery) on {device}...")
    print("=" * 50)
    
    root = f"{root0}TJU_results ds with pde/"
    args = get_tju_args()
    args.device = str(device)
    args.gpu = str(device)
    
    for e in range(args.experiments):
        save_folder = f'{root}Experiment{e + 1}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        setattr(args, 'save_folder', save_folder)
        
        dataloader = load_tju_data(args)
        pinn = PINN_PDE_Discovery(args).to(device)
        pinn.Train(trainloader=dataloader['train'], 
                  validloader=dataloader['valid'], 
                  testloader=dataloader['test'])
    
    # Organize results
    xlsx_path = f'{root}TJU_results.xlsx'
    writer = pd.ExcelWriter(xlsx_path)
    results = TJUResults(root, gap=0.07)
    df_mean1 = results.get_battery_average(args.experiments)
    df_mean2 = results.get_experiments_mean(args.experiments)
    df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
    
    df_mean2_with_avg = df_mean2.copy()
    avg_row = df_mean2.mean(axis=0)
    avg_row.name = 'average'
    df_mean2_with_avg = df_mean2_with_avg.append(avg_row)
    df_mean2_with_avg.to_excel(writer, sheet_name='experiment_mean_0', index=True)
    
    print(f"TJU MAPE: {df_mean2['MAPE'].mean():.4f}")
    print(f"TJU RMSE: {df_mean2['RMSE'].mean():.6f}")
    writer.close()
    
    # Copy discovered equation
    eq_file = os.path.join(f'{root}Experiment1', 'discovered_equation.txt')
    if os.path.exists(eq_file):
        import shutil
        shutil.copy(eq_file, os.path.join(root, 'discovered_equation_example.txt'))
    
    print(f"TJU training completed on {device}.")


def train_hust(device_name='cpu'):
    """Train the HUST dataset on the selected device."""
    device = torch.device(device_name)
    print("=" * 50)
    print(f"Starting HUST training (PDE discovery) on {device}...")
    print("=" * 50)
    
    root = f"{root0}HUST_results ds with pde/"
    args = get_hust_args()
    args.device = str(device)
    args.gpu = str(device)
    
    for e in range(args.experiments):
        save_folder = f'{root}Experiment{e + 1}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        setattr(args, 'save_folder', save_folder)
        
        dataloader = load_hust_data(args)
        pinn = PINN_PDE_Discovery(args).to(device)
        pinn.Train(trainloader=dataloader['train'], 
                  validloader=dataloader['valid'], 
                  testloader=dataloader['test'])
    
    # Organize results
    xlsx_path = f'{root}HUST_results.xlsx'
    writer = pd.ExcelWriter(xlsx_path)
    results = HUSTResults(root)
    
    df_battery_mean = results.get_battery_mean(args.experiments)
    df_experiment_mean = results.get_experiments_mean(args.experiments)
    df_battery_mean.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_experiment_mean.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    
    print(f"HUST MAPE: {df_experiment_mean['MAPE'].mean():.4f}")
    print(f"HUST RMSE: {df_experiment_mean['RMSE'].mean():.6f}")
    writer.close()
    
    # Copy discovered equation
    eq_file = os.path.join(f'{root}Experiment1', 'discovered_equation.txt')
    if os.path.exists(eq_file):
        import shutil
        shutil.copy(eq_file, os.path.join(root, 'discovered_equation_example.txt'))
    
    print(f"HUST training completed on {device}.")


def train_mit(device_name='cpu'):
    """Train the MIT dataset on the selected device."""
    device = torch.device(device_name)
    print("=" * 50)
    print(f"Starting MIT training (PDE discovery) on {device}...")
    print("=" * 50)
    
    root = f"{root0}MIT_results ds with pde/"
    args = get_mit_args()
    args.device = str(device)
    args.gpu = str(device)
    
    for e in range(args.experiments):
        save_folder = f'{root}Experiment{e + 1}'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        setattr(args, 'save_folder', save_folder)
        
        dataloader = load_mit_data(args)
        pinn = PINN_PDE_Discovery(args).to(device)
        pinn.Train(trainloader=dataloader['train'], 
                  validloader=dataloader['valid'], 
                  testloader=dataloader['test'])
    
    # Organize results
    xlsx_path = f'{root}MIT_results.xlsx'
    writer = pd.ExcelWriter(xlsx_path)
    results = MITResults(root)
    
    df_mean1 = results.get_battery_average(args.experiments)
    df_mean2 = results.get_experiment_average(args.experiments)
    
    df_mean1.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_mean2.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    
    print(f"MIT MAPE: {df_mean2['MAPE'].mean():.4f}")
    print(f"MIT RMSE: {df_mean2['RMSE'].mean():.6f}")
    writer.close()
    
    # Copy discovered equation
    eq_file = os.path.join(f'{root}Experiment1', 'discovered_equation.txt')
    if os.path.exists(eq_file):
        import shutil
        shutil.copy(eq_file, os.path.join(root, 'discovered_equation_example.txt'))
    
    print(f"MIT training completed on {device}.")


def main():
    """Train one selected dataset serially on CPU."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--dataset', choices=['XJTU', 'TJU', 'HUST', 'MIT'], default='XJTU')
    selected, _ = parser.parse_known_args()
    train_functions = {
        'XJTU': train_xjtu,
        'TJU': train_tju,
        'HUST': train_hust,
        'MIT': train_mit,
    }
    train_functions[selected.dataset]('cpu')


if __name__ == '__main__':
    main()
