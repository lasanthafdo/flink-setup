# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from glob import glob
from sklearn.metrics import auc

pd.set_option('display.max_columns', None)


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "/home/m34ferna/flink-tests/data"
    exp_date_id = "jun-20"
    file_date_default = "2022_06_20"
    file_date_adaptive = "2022_06_20"
    metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
    lrb_default_num_out_file = data_dir + "/" + exp_date_id + \
                               "/" + metric_name + "_lrb_default_" + file_date_default + ".csv"
    lrb_adaptive_num_out_file = data_dir + "/" + exp_date_id + \
                                "/" + metric_name + "_lrb_adaptive_" + file_date_adaptive + ".csv"
    lrb_replicating_num_out_file = data_dir + "/" + exp_date_id + \
                                "/" + metric_name + "_lrb_replicating_" + file_date_adaptive + ".csv"

    upper_time_threshold = 540
    lower_time_threshold = 60
    plot_tp = True

    if plot_tp:
        print("Reading file : " + lrb_default_num_out_file)
        col_list = ["name", "time", "operator_name", "task_name", "subtask_index", "count", "rate"]
        # col_list = ["name", "time", "operator_name", "subtask_index", "rate"]
        lrb_default_df = pd.read_csv(lrb_default_num_out_file, usecols=col_list)
        print(lrb_default_df.columns)
        print(lrb_default_df)
        print(lrb_default_df["name"].unique())
        print(lrb_default_df['operator_name'].unique())

        lrb_default_src_df = lrb_default_df[lrb_default_df['operator_name'].str.contains('Source:')].drop(
            ['name'], axis=1).groupby(['time'])[['rate', 'count']].sum().reset_index()
        lrb_default_src_df['rel_time'] = lrb_default_src_df['time'].subtract(lrb_default_src_df['time'].min()).div(
            1_000_000_000)
        lrb_default_src_df = lrb_default_src_df.loc[
            (lrb_default_src_df['rel_time'] > lower_time_threshold) & (lrb_default_src_df['rel_time'] < upper_time_threshold)]
        lrb_default_avg = np.mean(lrb_default_src_df['rate'])

        print("Printing values for LRB-Default")
        print(lrb_default_src_df)

        print("Reading file : " + lrb_replicating_num_out_file)
        lrb_replicating_df = pd.read_csv(lrb_replicating_num_out_file, usecols=col_list)
        print(lrb_replicating_df.columns)
        print(lrb_replicating_df)
        print(lrb_replicating_df["name"].unique())
        print(lrb_replicating_df['operator_name'].unique())

        replicating_offset = 105
        lrb_replicating_src_df = \
            lrb_replicating_df[lrb_replicating_df['operator_name'].str.contains('Source:')].drop(['name'], axis=1).groupby(
                ['time'])[['rate', 'count']].sum().reset_index()
        lrb_replicating_src_df['rel_time'] = lrb_replicating_src_df['time'].subtract(lrb_replicating_src_df['time'].min()).div(
            1_000_000_000).subtract(replicating_offset)
        lrb_replicating_src_df = lrb_replicating_src_df.loc[
            (lrb_replicating_src_df['rel_time'] > lower_time_threshold) & (lrb_replicating_src_df['rel_time'] < upper_time_threshold)]
        lrb_replicating_avg = np.mean(lrb_replicating_src_df['rate'])

        print("Printing values for LRB-Adaptive")
        print(lrb_replicating_src_df)

        print("Reading file : " + lrb_adaptive_num_out_file)
        lrb_adaptive_df = pd.read_csv(lrb_adaptive_num_out_file, usecols=col_list)
        print(lrb_adaptive_df.columns)
        print(lrb_adaptive_df)
        print(lrb_adaptive_df["name"].unique())
        print(lrb_adaptive_df['operator_name'].unique())

        adaptive_offset = 100
        lrb_adaptive_src_df = \
            lrb_adaptive_df[lrb_adaptive_df['operator_name'].str.contains('Source:')].drop(['name'], axis=1).groupby(
                ['time'])[['rate', 'count']].sum().reset_index()
        lrb_adaptive_src_df['rel_time'] = lrb_adaptive_src_df['time'].subtract(lrb_adaptive_src_df['time'].min()).div(
            1_000_000_000).subtract(adaptive_offset)
        lrb_adaptive_src_df = lrb_adaptive_src_df.loc[
            (lrb_adaptive_src_df['rel_time'] > lower_time_threshold) & (lrb_adaptive_src_df['rel_time'] < upper_time_threshold)]
        lrb_adaptive_avg = np.mean(lrb_adaptive_src_df['rate'])

        print("Printing values for LRB-Adaptive")
        print(lrb_adaptive_src_df)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(lrb_default_src_df["rel_time"], lrb_default_src_df["rate"], label="LRB-Default")
        ax.plot(lrb_replicating_src_df["rel_time"], lrb_replicating_src_df["rate"], label="LRB-Replicating")
        ax.plot(lrb_adaptive_src_df["rel_time"], lrb_adaptive_src_df["rate"], label="LRB-Adaptive")
        plt.axhline(y=lrb_default_avg, ls='--', color='c', label="LRB-Default-Avg")
        plt.axhline(y=lrb_replicating_avg, ls='--', color='m', label="LRB-Replicating-Avg")
        plt.axhline(y=lrb_adaptive_avg, ls='--', color='r', label="LRB-Adaptive-Avg")
        plt.text(100, lrb_default_avg - 6000, 'Default Avg. TP = ' + f'{lrb_default_avg:,.2f}')
        plt.text(45, lrb_replicating_avg + 4000, 'Replicating Avg. TP = ' + f'{lrb_replicating_avg:,.2f}')
        plt.text(360, lrb_adaptive_avg + 4000, 'Adaptive Avg. TP = ' + f'{lrb_adaptive_avg:,.2f}')

        #ax.set_ylim(bottom=0)
        ax.set(xlabel="Time (sec)", ylabel="Throughput (event/sec)", title="Throughput")
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
        plt.savefig("throughput_" + exp_date_id + ".png")
        plt.show()

        count_fig, count_ax = plt.subplots(figsize=(12, 6))

        count_ax.plot(lrb_default_src_df["rel_time"], lrb_default_src_df["count"], label="LRB-Default")
        count_ax.plot(lrb_replicating_src_df["rel_time"], lrb_replicating_src_df["count"], label="LRB-Replicating")
        count_ax.plot(lrb_adaptive_src_df["rel_time"], lrb_adaptive_src_df["count"], label="LRB-Adaptive")
        # plt.axhline(y=lrb_default_avg, ls='--', color='c', label="LRB-Default-Avg")
        # plt.axhline(y=lrb_replicating_avg, ls='--', color='m', label="LRB-Replicating-Avg")
        # plt.axhline(y=lrb_adaptive_avg, ls='--', color='r', label="LRB-Adaptive-Avg")
        # plt.text(100, lrb_default_avg - 8000, 'Default Avg. TP = ' + f'{lrb_default_avg:,.2f}')
        # plt.text(20, lrb_replicating_avg + 5000, 'Replicating Avg. TP = ' + f'{lrb_replicating_avg:,.2f}')
        # plt.text(300, lrb_adaptive_avg + 5000, 'Adaptive Avg. TP = ' + f'{lrb_adaptive_avg:,.2f}')

        #count_ax.set_ylim(bottom=0)
        count_ax.set(xlabel="Time (sec)", ylabel="Total events", title="Event count")
        count_ax.tick_params(axis="x", rotation=0)
        count_ax.legend()
        plt.savefig("count_" + exp_date_id + ".png")
        plt.show()

