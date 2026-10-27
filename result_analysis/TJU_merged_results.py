"""Analyze merged TJU dataset results without distinguishing batches."""
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix

class Results:
    def __init__(self, root='results/TJU results/', gap=0.07):
        self.root = root
        self.gap = gap
        self.experiments = [d for d in os.listdir(root) if d.startswith('Experiment')]
        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiment(1)

    def _update_experiment(self, experiment=1):
        subfolder = f'Experiment{experiment}'
        self.log_dir = os.path.join(self.root, subfolder, 'logging.txt')
        self.pred_label = os.path.join(self.root, subfolder, 'pred_label.npy')
        self.true_label = os.path.join(self.root, subfolder, 'true_label.npy')

    def parser_label(self):
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        RMSE_list = []
        diff = np.diff(true_label)
        split_point = np.where(diff > self.gap)[0]
        local_minima = np.concatenate((split_point, [len(true_label)]))
        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            [MAE_i, MAPE_i, _, RMSE_i] = eval_metrix(pred_i, true_i)
            start = end + 1
            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            RMSE_list.append(RMSE_i)
        results_dict = {
            'pred_label': pred_label_list,
            'true_label': true_label_list,
            'MAE': MAE_list,
            'MAPE': MAPE_list,
            'RMSE': RMSE_list
        }
        return results_dict

    def get_test_results(self, e=1):
        self._update_experiment(experiment=e)
        return self.parser_label()

    def get_battery_average(self, total_experiment=10):
        df_mean_values = []
        for e in range(1, 1+total_experiment):
            res = self.get_test_results(e)
            df = pd.DataFrame(res)
            df = df[['MAE', 'MAPE', 'RMSE']]
            df_i_mean = df.mean(axis=0)
            df_mean_values.append(df_i_mean.values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE'])
        df_mean.insert(0, column='experiment', value=np.arange(1, 1+total_experiment))
        return df_mean

    def get_experiments_mean(self, total_experiment=10):
        df_value_list = []
        for e in range(1, 1+total_experiment):
            res = self.get_test_results(e)
            df = pd.DataFrame(res)
            df = df[['MAE', 'MAPE', 'RMSE']]
            df_value_list.append(df.values)
        columns = ['MAE', 'MAPE', 'RMSE']
        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array, axis=0)
        df_mean = pd.DataFrame(np_mean, columns=columns)
        return df_mean
