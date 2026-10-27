"""Analyze HUST dataset results."""
import pandas as pd
import numpy as np
import os
from utils.util import eval_metrix

class Results:
    def __init__(self,root='../results/HUST results/'):
        self.root = root
        self.experiments = os.listdir(root)

        self.log_dir = None
        self.pred_label = None
        self.true_label = None
        self._update_experiments(1)

    def _update_experiments(self,e):
        experiment = 'Experiment' + str(e)
        self.log_dir = os.path.join(self.root, experiment,'logging.txt')
        self.pred_label = os.path.join(self.root, experiment,'pred_label.npy')
        self.true_label = os.path.join(self.root, experiment,'true_label.npy')

    def parser_log(self):
        """Parse data from the log generated during training.

        Returns:
            A dictionary of parsed log data.
        """
        data_dict = {}

        with open(self.log_dir, 'r') as f:
            lines = f.readlines()

        # Parse hyperparameters logged at the CRITICAL level.
        for line in lines:
            if 'CRITICAL' in line:
                params = line.split('\t')[-1].split('\n')[0]
                k, v = params.split(':')
                data_dict[k] = v

        # Parse losses recorded during training, validation, and testing.
        train_total_loss = []
        valid_data_loss = []

        test_mse = []
        test_epoch = []

        for i in range(len(lines)):
            line = lines[i]
            if '[train] epoch:1 iter:1 data' in line:
                train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Train]' in line:
                train_total_loss.append(float(line.split('total loss:')[1].split('\n')[0]))
            elif '[Valid]' in line:
                valid_data_loss.append(float(line.split('MSE:')[1].split('\n')[0]))
            elif '[Test]' in line:
                test_mse.append(float(line.split('MSE:')[1].split(',')[0]))
                test_epoch.append(int(lines[i - 1].split('epoch:')[1].split(',')[0]))


        data_dict['train_total_loss'] = train_total_loss
        data_dict['valid_data_loss'] = valid_data_loss
        data_dict['test_mse'] = test_mse
        data_dict['test_epoch'] = test_epoch

        # Parse data paths.
        line1 = lines[1]
        if '.csv' in line1:
            line = line1[1:-2]
            line_list = line.replace('data/HUST data/', '').replace('.csv','').replace('\'','').split(', ')
            data_dict['IDs_1'] = line_list

        line2 = lines[3]
        if '.csv' in line2:
            line = line2[1:-2]
            line_list = line.replace('data/HUST data/', '').replace('.csv', '').replace('\'', '').split(', ')
            data_dict['IDs_2'] = line_list

        return data_dict

    def parser_label(self):
        """Parse prediction results."""
        pred_label = np.load(self.pred_label).reshape(-1)
        true_label = np.load(self.true_label).reshape(-1)
        # [MAE, MAPE, MSE, RMSE, R2] = eval_metrix(pred_label, true_label)
        [MAE, MAPE, MSE, RMSE] = eval_metrix(pred_label, true_label)


        # Store prediction results for each battery.
        pred_label_list = []
        true_label_list = []
        MAE_list = []
        MAPE_list = []
        MSE_list = []
        RMSE_list = []
        # R2_list = []

        diff = np.diff(true_label)
        split_point = np.where(diff>0.1)[0]
        local_minima = np.concatenate((split_point,[len(true_label)]))
        # Split the concatenated labels into individual batteries.
        start = 0
        end = 0
        for i in range(len(local_minima)):
            end = local_minima[i]
            pred_i = pred_label[start:end]
            true_i = true_label[start:end]
            # [MAE_i, MAPE_i, MSE_i, RMSE_i, R2_i] = eval_metrix(pred_i, true_i)
            [MAE_i, MAPE_i, MSE_i, RMSE_i] = eval_metrix(pred_i, true_i)
            #print('battery {} MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(i+1,MAE_i, MAPE_i, MSE_i, RMSE_i,R2_i))
            start = end+1

            pred_label_list.append(pred_i)
            true_label_list.append(true_i)
            MAE_list.append(MAE_i)
            MAPE_list.append(MAPE_i)
            MSE_list.append(MSE_i)
            RMSE_list.append(RMSE_i)
            # R2_list.append(R2_i)
        #print('Mean  MAE:{:.4f}, MAPE:{:.4f}, MSE:{:.6f}, RMSE:{:.4f}, R2:{:.4f}'.format(MAE, MAPE, MSE, RMSE,R2))
        results_dict = {}
        results_dict['pred_label'] = pred_label_list
        results_dict['true_label'] = true_label_list
        results_dict['MAE'] = MAE_list
        results_dict['MAPE'] = MAPE_list
        #results_dict['MSE'] = MSE_list
        results_dict['RMSE'] = RMSE_list
        # results_dict['R2'] = R2_list
        return results_dict


    def get_test_results(self,e):
        """Parse battery IDs from the training and test data.

        Args:
            e: Experiment ID.
        """
        self._update_experiments(e)
        log_dict = self.parser_log()
        results_dict = self.parser_label()
        results_dict['channel'] = log_dict['IDs_2']
        return results_dict

    def get_battery_mean(self, total_experiment=10):
        df_mean_values = []
        for e in range(1, total_experiment + 1):
            res = self.get_test_results(e=e)
            df = pd.DataFrame(res)
            df = df[['MAE', 'MAPE', 'RMSE']]
            df_i_mean = df.mean(axis=0)
            df_mean_values.append(df_i_mean.values)
        df_mean_values = np.array(df_mean_values)
        df_mean = pd.DataFrame(df_mean_values, columns=['MAE', 'MAPE', 'RMSE'])
        df_mean.insert(0, column='experiment', value=np.arange(1, total_experiment + 1))
        # Calculate the mean of the MAE, MAPE, and RMSE columns.
        average_values = df_mean[['MAE', 'MAPE', 'RMSE']].mean(axis=0).values
        average_row = ['average'] + average_values.tolist()

        # Append the mean values as the final row.
        df_mean.loc[len(df_mean)] = average_row
        print(df_mean)
        return df_mean

    def get_experiments_mean(self, total_experiment=10):
        """Return each test battery's metrics averaged across experiments.

        Returns:
            A DataFrame with one row per battery.
        """
        df_value_list = []
        for i in range(1, total_experiment + 1):
            res = self.get_test_results(i)
            df = pd.DataFrame(res)
            df = df[['channel','MAE', 'MAPE', 'RMSE']]
            df = df.sort_values(by='channel')
            df.reset_index(drop=True, inplace=True)
            df_value_list.append(df[['MAE','MAPE','RMSE']].values)
        channel = df['channel']
        columns = ['MAE', 'MAPE', 'RMSE']

        np_array = np.array(df_value_list)
        np_mean = np.mean(np_array,axis=0)
        df_mean = pd.DataFrame(np_mean,columns=columns)
        df_mean.insert(0,column='channel',value=channel)
        #df_mean['channel'] = df_mean['channel'].astype(str)
        df_mean['channel'] = df_mean['channel'].apply(lambda x: x.replace('-','--'))
        print(df_mean)
        return df_mean



if __name__ == '__main__':
    root = '../results2.5/HUST results PIKAN/'
    writer = pd.ExcelWriter('../results2.5/HUST_results_PIKAN.xlsx')
    results = Results(root)

    df_battery_mean = results.get_battery_mean()
    df_experiment_mean = results.get_experiments_mean()
    df_battery_mean.to_excel(writer, sheet_name='battery_mean_0', index=False)
    df_experiment_mean.to_excel(writer, sheet_name='experiment_mean_0', index=False)
    #
    writer.close()


