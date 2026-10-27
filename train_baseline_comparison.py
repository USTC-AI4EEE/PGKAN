import importlib
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse

from utils.util import AverageMeter, get_logger
from Model.eKAN_for_symbolic import KAN
from Model.Compare_Models import MLP, CNN, LSTM, GRU
from Model.Model import LR_Scheduler
from dataloader.dataloader import XJTUdata, HUSTdata, MITdata, TJUdata


DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class Trainer:
    def __init__(self, model, train_loader, valid_loader, test_loader, args, device=None):
        self.device = torch.device(device or DEFAULT_DEVICE)
        self.model = model.to(self.device)
        self.args = args
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.save_dir = args.save_folder
        os.makedirs(self.save_dir, exist_ok=True)
        self.epochs = args.epochs
        self.logger = get_logger(os.path.join(args.save_folder, args.log_dir))
        self.loss_meter = AverageMeter()
        self.loss_func = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.warmup_lr)
        self.scheduler = LR_Scheduler(
            optimizer=self.optimizer,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            num_epochs=args.epochs,
            base_lr=args.lr,
            final_lr=args.final_lr,
        )

    def clear_logger(self):
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        self.logger.handlers.clear()

    def train(self):
        min_loss = 100
        early_stop = 0
        for epoch in range(1, self.epochs + 1):
            early_stop += 1
            self.train_one_epoch(epoch)
            self.scheduler.step()
            valid_loss = self.valid(epoch)
            if valid_loss < min_loss and self.test_loader is not None:
                min_loss = valid_loss
                self.test()
                torch.save(self.model.state_dict(), os.path.join(self.save_dir, 'model.pth'))
                early_stop = 0
            if early_stop > 10:
                break
        self.clear_logger()


def load_XJTU_data(args, small_sample=None):
    root = 'data/XJTU data'
    data = XJTUdata(root=root, args=args)
    train_list = []
    test_list = []
    for filename in os.listdir(root):
        path = os.path.join(root, filename)
        if '4' in filename or '8' in filename:
            test_list.append(path)
        else:
            train_list.append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3'],
    }


def load_TJU_data(args, small_sample=None):
    root = 'data/TJU data'
    data = TJUdata(root=root, args=args)
    train_list = []
    test_list = []
    for batch in sorted(os.listdir(root)):
        batch_root = os.path.join(root, batch)
        for index, filename in enumerate(sorted(os.listdir(batch_root)), start=1):
            path = os.path.join(batch_root, filename)
            if index % 10 in (5, 9):
                test_list.append(path)
            else:
                train_list.append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3'],
    }


def load_MIT_data(args, small_sample=None):
    root = 'data/MIT data'
    train_list = []
    test_list = []
    for batch in ('2017-05-12', '2017-06-30', '2018-04-12'):
        batch_root = os.path.join(root, batch)
        for filename in sorted(os.listdir(batch_root)):
            battery_id = int(filename.split('-')[-1].split('.')[0])
            path = os.path.join(batch_root, filename)
            (test_list if battery_id % 5 == 0 else train_list).append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]
    data = MITdata(root=root, args=args)
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3'],
    }


def load_HUST_data(args, small_sample=None):
    root = 'data/HUST data'
    test_ids = {
        f'{group}-{cycle}'
        for group in range(1, 11)
        for cycle in (4, 8)
    }
    test_ids.add('5-7')
    data = HUSTdata(root=root, args=args)
    train_list = []
    test_list = []
    for filename in sorted(os.listdir(root)):
        path = os.path.join(root, filename)
        (test_list if filename[:-4] in test_ids else train_list).append(path)
    if small_sample is not None:
        train_list = train_list[:small_sample]
    train_loader = data.read_all(specific_path_list=train_list)
    test_loader = data.read_all(specific_path_list=test_list)
    return {
        'train': train_loader['train_2'],
        'valid': train_loader['valid_2'],
        'test': test_loader['test_3'],
    }


def get_args():
    parser = argparse.ArgumentParser('Sequential model comparison')
    parser.add_argument('--model', type=str, default='KAN', choices=['KAN', 'MLP', 'CNN', 'LSTM', 'GRU'])
    parser.add_argument('--dataset', type=str, default='XJTU', choices=['XJTU', 'HUST', 'MIT', 'TJU'])
    parser.add_argument('--normalization_method', type=str, default='min-max', help='min-max or z-score')
    parser.add_argument('--xjtu_batch', type=str, default='2C', choices=['2C', '3C', 'R2.5', 'R3', 'RW', 'satellite'])
    parser.add_argument('--in_same_batch', type=bool, default=True)
    parser.add_argument('--tju_batch', type=int, default=0, choices=[0, 1, 2])
    parser.add_argument('--tju_train_batch', type=int, default=-1, choices=[-1, 0, 1, 2])
    parser.add_argument('--tju_test_batch', type=int, default=-1, choices=[-1, 0, 1, 2])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=int, default=30)
    parser.add_argument('--warmup_lr', type=float, default=2e-3)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--final_lr', type=float, default=2e-4)
    parser.add_argument('--lr_F', type=float, default=5e-4)
    parser.add_argument('--save_folder', type=str, default='./results of reviewer/')
    parser.add_argument('--log_dir', type=str, default='logging.txt')
    parser.add_argument('--batch_size', type=int, default=512)
    args = parser.parse_args()
    os.makedirs(args.save_folder, exist_ok=True)
    return args


def load_model(args, device):
    model_classes = {
        'KAN': KAN,
        'MLP': MLP,
        'CNN': CNN,
        'LSTM': LSTM,
        'GRU': GRU,
    }
    return model_classes[args.model]().to(device)


DATASETS = {
    'XJTU': {
        'loader': load_XJTU_data,
        'results_module': 'XJTU_KAN_results',
        'summary_methods': ('get_battery_mean', 'get_experiments_mean'),
    },
    'TJU': {
        'loader': load_TJU_data,
        'results_module': 'TJU_KAN_results',
        'summary_methods': ('get_battery_average', 'get_experiments_mean'),
    },
    'MIT': {
        'loader': load_MIT_data,
        'results_module': 'MIT_KAN_results',
        'summary_methods': ('get_battery_average', 'get_experiment_average'),
    },
    'HUST': {
        'loader': load_HUST_data,
        'results_module': 'HUST_KAN_results',
        'summary_methods': ('get_battery_mean', 'get_experiments_mean'),
    },
}


class SequentialTrainer(Trainer):
    def _model_inputs(self, features):
        if self.args.model == 'KAN':
            return features[:, [4, 16]]
        return features

    def train_one_epoch(self, epoch):
        self.model.train()
        self.loss_meter.reset()
        for features, _, labels, _ in self.train_loader:
            features = self._model_inputs(features.to(self.device))
            labels = labels.to(self.device)
            predictions = self.model(features)
            loss = self.loss_func(predictions, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.loss_meter.update(loss.item())

        self.logger.info(
            '[Train] epoch:{:0>3d}, data loss:{:.6f}'.format(
                epoch, self.loss_meter.avg
            )
        )
        return self.loss_meter.avg

    def valid(self, epoch):
        self.model.eval()
        self.loss_meter.reset()
        with torch.no_grad():
            for features, _, labels, _ in self.valid_loader:
                features = self._model_inputs(features.to(self.device))
                labels = labels.to(self.device)
                predictions = self.model(features)
                self.loss_meter.update(self.loss_func(predictions, labels).item())

        self.logger.info(
            '[Valid] epoch:{:0>3d}, data loss:{:.6f}'.format(
                epoch, self.loss_meter.avg
            )
        )
        return self.loss_meter.avg

    def test(self):
        self.model.eval()
        true_labels = []
        predicted_labels = []
        with torch.no_grad():
            for features, _, labels, _ in self.test_loader:
                features = self._model_inputs(features.to(self.device))
                predictions = self.model(features)
                true_labels.append(labels.cpu().numpy())
                predicted_labels.append(predictions.cpu().numpy())

        true_labels = np.concatenate(true_labels, axis=0)
        predicted_labels = np.concatenate(predicted_labels, axis=0)
        if self.save_dir is not None:
            np.save(os.path.join(self.save_dir, 'true_label.npy'), true_labels)
            np.save(os.path.join(self.save_dir, 'pred_label.npy'), predicted_labels)
        return true_labels, predicted_labels


def run_dataset(dataset, args, timestamp):
    config = DATASETS[dataset]
    args.dataset = dataset
    if dataset == 'XJTU':
        args.xjtu_batch = '2C'
    elif dataset == 'TJU':
        args.tju_batch = 0

    result_root = os.path.join('results', f'{timestamp}-{dataset}-{args.model} results')
    for experiment in range(1, 11):
        args.save_folder = os.path.join(result_root, f'Experiment{experiment}')
        os.makedirs(args.save_folder, exist_ok=True)
        model = load_model(args, DEFAULT_DEVICE)
        loaders = config['loader'](args)
        trainer = SequentialTrainer(
            model,
            loaders['train'],
            loaders['valid'],
            loaders['test'],
            args,
            DEFAULT_DEVICE,
        )
        trainer.train()

    try:
        results_class = importlib.import_module(config['results_module']).Results
        results = results_class(result_root)
        battery_method, experiment_method = config['summary_methods']
        with pd.ExcelWriter(
            os.path.join(result_root, f'{dataset}_{args.model}_results.xlsx')
        ) as writer:
            getattr(results, battery_method)().to_excel(
                writer, sheet_name='battery_mean_0', index=True
            )
            getattr(results, experiment_method)().to_excel(
                writer, sheet_name='experiment_mean_0', index=False
            )
    except Exception as error:
        print(f'[{dataset}] result aggregation failed: {error}')


def main():
    args = get_args()
    timestamp = datetime.now().strftime('%m%d_%H%M')
    print(f'Using device: {DEFAULT_DEVICE}; model: {args.model}')

    for dataset in ('XJTU', 'TJU', 'MIT', 'HUST'):
        run_dataset(dataset, args, timestamp)


if __name__ == '__main__':
    main()