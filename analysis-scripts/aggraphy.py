# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)


def get_formatted_tput(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset, policy_name):
    print("Reading file : " + lrb_num_out_file)
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    lrb_src_df = lrb_df[lrb_df['operator_name'].str.contains('Source:')].drop(
        ['name'], axis=1).groupby(['time'])[['rate', 'count']].sum().reset_index()
    lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_src_df = lrb_src_df.loc[
        (lrb_src_df['rel_time'] > lower_threshold) & (
                lrb_src_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_src_df['rate'])
    return lrb_src_df, lrb_avg


def get_filename(data_directory, exp_id, metric_name, file_date, sched_policy, par_level='12'):
    return data_directory + "/" + exp_id + \
           "/" + metric_name + "_" + sched_policy + "_" + par_level + "_" + file_date + ".csv"


def get_grouped_df(col_list, data_file):
    metric_df = pd.read_csv(data_file, usecols=col_list)
    metric_grouped_df = metric_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
    metric_grouped_df['rel_time'] = metric_grouped_df['time'].subtract(
        metric_grouped_df['time'].min()).div(
        1_000_000_000)
    return metric_grouped_df


def get_df_without_groupby(col_list, data_file):
    metric_df = pd.read_csv(data_file, usecols=col_list)
    metric_df['rel_time'] = metric_df['time'].subtract(
        metric_df['time'].min()).div(
        1_000_000_000)
    metric_df.set_index('rel_time', inplace=True)
    return metric_df


def combine_df_without_groupby(original_df, col_list, data_file, sched_policy):
    metric_df = pd.read_csv(data_file, usecols=col_list)
    metric_df['rel_time'] = metric_df['time'].subtract(
        metric_df['time'].min()).div(
        1_000_000_000)
    metric_df['sched_policy'] = sched_policy
    metric_df.set_index('rel_time', inplace=True)
    combined_df = pd.concat([original_df, metric_df])
    print(combined_df)
    return combined_df


def plot_metric(data_df, x_label, y_label, plot_title, group_by_col_name, plot_filename, exp_date_id):
    data_df.groupby(group_by_col_name)['value'].plot(legend=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.savefig(results_dir + "/" + plot_filename + "_" + exp_date_id + ".png")
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "/home/m34ferna/flink-tests/data"
    experiment_date_id = "jul-25-1"
    file_date_default = "2022_07_25"
    file_date_adaptive = "2022_07_25"
    results_dir = "results/" + experiment_date_id
    os.makedirs(results_dir, exist_ok=True)

    upper_time_threshold = 580
    lower_time_threshold = 80
    plot_tp = True
    plot_cpu = False
    plot_mem = False
    plot_busy = True
    plot_idle = True
    plot_backpressure = True
    plot_iq_len = True
    plot_nw = True

    has_replicating_only_metrics = False
    has_scheduling_only_metrics = True
    has_adaptive_metrics = False

    default_offset = 0

    if plot_tp:
        col_list = ["name", "time", "operator_name", "task_name", "subtask_index", "count", "rate"]
        metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
        lrb_avg_all_df = pd.DataFrame(columns=['Scheduling Policy','Parallelism','tp'])
        for parallelism_level in range(4,25,4):
            lrb_default_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                                    "lrb_default",
                                                    str(parallelism_level))
            lrb_default_src_df, lrb_default_avg = get_formatted_tput(lrb_default_num_out_file, col_list,
                                                                     lower_time_threshold, upper_time_threshold,
                                                                     default_offset, "Default")
            lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Default", parallelism_level, lrb_default_avg]
            if has_replicating_only_metrics:
                replicating_offset = 0
                lrb_replicating_num_out_file = get_filename(data_dir, experiment_date_id, metric_name,
                                                            file_date_default,
                                                            "lrb_replicating", str(parallelism_level))
                lrb_replicating_src_df, lrb_replicating_avg = get_formatted_tput(lrb_replicating_num_out_file, col_list,
                                                                                 lower_time_threshold,
                                                                                 upper_time_threshold,
                                                                                 replicating_offset, "Replicating")
                lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Replicating", parallelism_level, lrb_replicating_avg]

            if has_scheduling_only_metrics:
                scheduling_offset = 0
                lrb_scheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                                           "lrb_scheduling", str(parallelism_level))
                lrb_scheduling_src_df, lrb_scheduling_avg = get_formatted_tput(lrb_scheduling_num_out_file, col_list,
                                                                               lower_time_threshold,
                                                                               upper_time_threshold,
                                                                               scheduling_offset, "Replicating")
                lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Scheduling", parallelism_level, lrb_scheduling_avg]

            if has_adaptive_metrics:
                adaptive_offset = 0
                lrb_adaptive_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                                         "lrb_adaptive",
                                                         str(parallelism_level))
                lrb_adaptive_src_df, lrb_adaptive_avg = get_formatted_tput(lrb_adaptive_num_out_file, col_list,
                                                                           lower_time_threshold, upper_time_threshold,
                                                                           adaptive_offset, "Adaptive")
                lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Adaptive", parallelism_level, lrb_adaptive_avg]

        print(lrb_avg_all_df.dtypes)
        pivoted_lrb_avg_all_df = lrb_avg_all_df.pivot(index='Parallelism',columns='Scheduling Policy', values='tp')
        print(pivoted_lrb_avg_all_df)

        ax = pivoted_lrb_avg_all_df.plot.bar(rot=0)
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_ylabel('Events/sec')
        ax.legend(loc="lower left")
        plt.title('LRB Throughput - 4 source replicas')
        plt.tight_layout()
        plt.savefig(results_dir + "/throughput_all_" + experiment_date_id + ".png")
        plt.show()
        exit(0)

        count_fig, count_ax = plt.subplots(figsize=(12, 6))

        count_ax.plot(lrb_default_src_df["rel_time"], lrb_default_src_df["count"], label="LRB-Default")
        if has_replicating_only_metrics:
            count_ax.plot(lrb_replicating_src_df["rel_time"], lrb_replicating_src_df["count"], label="LRB-Replicating")
        if has_adaptive_metrics:
            count_ax.plot(lrb_adaptive_src_df["rel_time"], lrb_adaptive_src_df["count"], label="LRB-Adaptive")
        if has_scheduling_only_metrics:
            count_ax.plot(lrb_scheduling_src_df["rel_time"], lrb_scheduling_src_df["count"], label="LRB-Scheduling")

        # count_ax.set_ylim(bottom=0)
        count_ax.set(xlabel="Time (sec)", ylabel="Total events", title="Event count")
        count_ax.tick_params(axis="x", rotation=0)
        count_ax.legend()
        plt.savefig(results_dir + "/count_" + parallelism_level + "_" + experiment_date_id + ".png")
        plt.show()