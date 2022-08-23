# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_formatted_tput(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset):
    print("Reading file : " + lrb_num_out_file)
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    lrb_src_df = lrb_df[lrb_df['operator_name'].str.contains('Source:')].drop(
        ['name'], axis=1).groupby(['time'])[['rate', 'count']].sum().reset_index()
    lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_src_df = lrb_src_df.loc[
        (lrb_src_df['rel_time'] > lower_threshold) & (lrb_src_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_src_df['rate'])
    return lrb_src_df, lrb_avg


def get_formatted_latency(lrb_latency_file, column_list, lower_threshold, upper_threshold, offset, target_op_id,
                          target_stat):
    print("Reading file : " + lrb_latency_file)
    lrb_latency_df = pd.read_csv(lrb_latency_file, usecols=column_list)
    lrb_sink_latency_df = lrb_latency_df[lrb_latency_df['operator_id'] == target_op_id].drop(
        ['name'], axis=1).groupby(['time'])[['mean', 'p50', 'p95', 'p99']].mean().reset_index()
    lrb_sink_latency_df['rel_time'] = lrb_sink_latency_df['time'].subtract(lrb_sink_latency_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_sink_latency_df = lrb_sink_latency_df.loc[
        (lrb_sink_latency_df['rel_time'] > lower_threshold) & (lrb_sink_latency_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_sink_latency_df[target_stat])
    return lrb_sink_latency_df, lrb_avg


def get_filename(data_directory, exp_id, metric_name, file_date, sched_policy, par_level='12', sched_period='0',
                 num_parts='1'):
    return data_directory + "/" + exp_id + \
           "/" + metric_name + "_" + sched_policy + "_" + sched_period + "ms_" + par_level + "_" + num_parts + "parts_" + file_date + ".csv"


def get_grouped_df(col_list, data_file):
    metric_df = pd.read_csv(data_file, usecols=col_list)
    metric_grouped_df = metric_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
    metric_grouped_df['rel_time'] = metric_grouped_df['time'].subtract(
        metric_grouped_df['time'].min()).div(
        1_000_000_000)
    metric_grouped_df.set_index('rel_time', inplace=True)
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


def get_op_name_id_mapping(lrb_tp_file):
    op_name_id_mapping_df = pd.read_csv(lrb_tp_file, usecols=['operator_name', 'operator_id']).drop_duplicates()
    op_name_id_dict = dict(zip(op_name_id_mapping_df['operator_name'], op_name_id_mapping_df['operator_id']))
    return op_name_id_dict


def get_pivoted_latency(lrb_latency_file, column_list, target_stat, op_to_id_dict):
    lrb_all_latency_for_sched_mode = pd.read_csv(lrb_latency_file, usecols=column_list)
    join_dict_df = pd.DataFrame(op_to_id_dict.items(), columns=['operator_name', 'operator_id'])
    lrb_all_latency_for_sched_mode = pd.merge(lrb_all_latency_for_sched_mode, join_dict_df, on="operator_id")
    lrb_all_latency_for_sched_mode = lrb_all_latency_for_sched_mode.groupby(['time', 'operator_name'])[
        [target_stat]].mean().reset_index()
    lrb_pivoted_latency_df = lrb_all_latency_for_sched_mode.pivot(index='time', columns='operator_name',
                                                                  values=target_stat)
    lrb_pivoted_latency_df.columns = [''.join(col).strip() for col in lrb_pivoted_latency_df.columns]
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.reset_index()[
        ['time', 'fil_1', 'tsw_1', 'prj_1', 'vehicle_win_1', 'speed_win_1', 'acc_win_1', 'toll_win_1',
         'toll_acc_win_1', 'Sink: sink_1']]
    lrb_pivoted_latency_df['rel_time'] = lrb_pivoted_latency_df['time'].subtract(
        lrb_pivoted_latency_df['time'].min()).div(1_000_000_000)
    # lrb_pivoted_latency_df.set_index('rel_time', inplace=True)
    print(lrb_pivoted_latency_df)
    return lrb_pivoted_latency_df


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "/home/m34ferna/flink-tests/data"
    experiment_date_id = "aug-22-1"
    file_date_default = "2022_08_22"
    file_date_adaptive = "2022_08_22"
    parallelism_level = "12"
    results_dir = "results/" + experiment_date_id + "/par_" + parallelism_level
    os.makedirs(results_dir, exist_ok=True)
    scheduling_period = "5"

    upper_time_threshold = 580
    lower_time_threshold = 0
    plot_tp = True
    plot_latency = True
    plot_cpu = False
    plot_mem = False
    plot_busy = False
    plot_idle = False
    plot_backpressure = False
    plot_iq_len = False
    plot_nw = False

    has_pseudo_default_metrics = False
    has_replicating_only_metrics = False
    has_scheduling_only_metrics = False
    has_adaptive_metrics = False

    default_offset = 0

    default_id_str = "lrb_default"
    default_sched_period = "5"
    num_parts = "6"
    default_sched_periods = ["1", "2", "3", "4", "5"]

    if plot_tp:
        col_list = ["name", "time", "operator_name", "operator_id", "task_name", "subtask_index", "count", "rate"]
        metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
        lrb_default_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default, default_id_str,
                                           parallelism_level, default_sched_period, num_parts)
        lrb_adaptive_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_adaptive,
                                            "lrb_adaptive", parallelism_level, scheduling_period, num_parts)
        lrb_replicating_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                               "lrb_replicating", parallelism_level, scheduling_period, num_parts)
        lrb_scheduling_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_adaptive,
                                              "lrb_scheduling", parallelism_level, scheduling_period, num_parts)
        lrb_pseudo_default_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_adaptive,
                                                  "lrb_pd", parallelism_level, "3600000", num_parts)
        lrb_default_src_tp_df, lrb_default_tp_avg = get_formatted_tput(lrb_default_tp_file, col_list,
                                                                       lower_time_threshold,
                                                                       upper_time_threshold,
                                                                       default_offset)

        lrb_default_df_all = []
        lrb_default_src_tp_df_all = []
        lrb_default_tp_avg_all = []
        for default_sched_period in default_sched_periods:
            lrb_default_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                               default_id_str, parallelism_level, default_sched_period, num_parts)
            lrb_default_df_all.append(pd.read_csv(lrb_default_tp_file, usecols=col_list))
            temp_lrb_default_src_tp_df, temp_lrb_default_tp_avg = get_formatted_tput(lrb_default_tp_file, col_list,
                                                                                     lower_time_threshold,
                                                                                     upper_time_threshold,
                                                                                     default_offset)
            lrb_default_src_tp_df_all.append(temp_lrb_default_src_tp_df)
            lrb_default_tp_avg_all.append(temp_lrb_default_tp_avg)

        lrb_default_df = lrb_default_df_all[0]
        src_task_indexes = lrb_default_df[lrb_default_df['operator_name'].str.contains('Source:')][
            'subtask_index'].unique()
        other_task_indexes = lrb_default_df[lrb_default_df['operator_name'].str.contains('tsw')][
            'subtask_index'].unique()
        src_task_indexes.sort()
        other_task_indexes.sort()
        print("Source subtasks: " + str(src_task_indexes))
        print("Other subtasks: " + str(other_task_indexes))

        lrb_default_op_name_id_dict = get_op_name_id_mapping(lrb_default_tp_file)

        if has_pseudo_default_metrics:
            pseudo_default_offset = 0
            lrb_pseudo_default_src_df, lrb_pseudo_default_avg = get_formatted_tput(lrb_pseudo_default_tp_file, col_list,
                                                                                   lower_time_threshold,
                                                                                   upper_time_threshold,
                                                                                   pseudo_default_offset)
            lrb_pseudo_default_op_name_id_dict = get_op_name_id_mapping(lrb_pseudo_default_tp_file)
        else:
            lrb_pseudo_default_src_df = None
            lrb_pseudo_default_avg = None
            lrb_pseudo_default_op_name_id_dict = None

        if has_replicating_only_metrics:
            replicating_offset = 0
            lrb_replicating_src_df, lrb_replicating_avg = get_formatted_tput(lrb_replicating_tp_file, col_list,
                                                                             lower_time_threshold, upper_time_threshold,
                                                                             replicating_offset)
            lrb_replicating_op_name_id_dict = get_op_name_id_mapping(lrb_replicating_tp_file)
        else:
            lrb_replicating_src_df = None
            lrb_replicating_avg = None
            lrb_replicating_op_name_id_dict = None

        if has_scheduling_only_metrics:
            scheduling_offset = 0
            lrb_scheduling_src_df, lrb_scheduling_avg = get_formatted_tput(lrb_scheduling_tp_file, col_list,
                                                                           lower_time_threshold, upper_time_threshold,
                                                                           scheduling_offset)
            lrb_scheduling_op_name_id_dict = get_op_name_id_mapping(lrb_scheduling_tp_file)
        else:
            lrb_scheduling_src_df = None
            lrb_scheduling_avg = None
            lrb_scheduling_op_name_id_dict = None

        if has_adaptive_metrics:
            adaptive_offset = 0
            lrb_adaptive_src_df, lrb_adaptive_avg = get_formatted_tput(lrb_adaptive_tp_file, col_list,
                                                                       lower_time_threshold, upper_time_threshold,
                                                                       adaptive_offset)
            lrb_adaptive_op_name_id_dict = get_op_name_id_mapping(lrb_adaptive_tp_file)
        else:
            lrb_adaptive_src_df = None
            lrb_adaptive_avg = None
            lrb_adaptive_op_name_id_dict = None

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, lrb_default_src_tp_df in enumerate(lrb_default_src_tp_df_all):
            ax.plot(lrb_default_src_tp_df["rel_time"], lrb_default_src_tp_df["rate"], label="LRB-Default" + str(idx))
        if has_pseudo_default_metrics:
            ax.plot(lrb_pseudo_default_src_df["rel_time"], lrb_pseudo_default_src_df["rate"], label="LRB-PD")
        if has_replicating_only_metrics:
            ax.plot(lrb_replicating_src_df["rel_time"], lrb_replicating_src_df["rate"], label="LRB-Replicating")
        if has_adaptive_metrics:
            ax.plot(lrb_adaptive_src_df["rel_time"], lrb_adaptive_src_df["rate"], label="LRB-Adaptive")
        if has_scheduling_only_metrics:
            ax.plot(lrb_scheduling_src_df["rel_time"], lrb_scheduling_src_df["rate"], label="LRB-Scheduling")

        # plt.axhline(y=lrb_default_tp_avg, ls='--', color='c', label="LRB-Default-Avg")
        # plt.text(100, lrb_default_tp_avg + 5000, 'Default Avg. TP = ' + f'{lrb_default_tp_avg:,.2f}')
        if has_pseudo_default_metrics:
            plt.axhline(y=lrb_pseudo_default_avg, ls='--', color='navy', label="LRB-PD-Avg")
            plt.text(360, lrb_pseudo_default_avg + 5000, 'PD Avg. TP = ' + f'{lrb_pseudo_default_avg:,.2f}')
        if has_scheduling_only_metrics:
            plt.axhline(y=lrb_scheduling_avg, ls='--', color='y', label="LRB-Scheduling-Avg")
            plt.text(360, lrb_scheduling_avg + 5000, 'Scheduling Avg. TP = ' + f'{lrb_scheduling_avg:,.2f}')
        if has_replicating_only_metrics:
            plt.axhline(y=lrb_replicating_avg, ls='--', color='m', label="LRB-Replicating-Avg")
            plt.text(200, lrb_replicating_avg + 5000, 'Replicating Avg. TP = ' + f'{lrb_replicating_avg:,.2f}')
        if has_adaptive_metrics:
            plt.axhline(y=lrb_adaptive_avg, ls='--', color='r', label="LRB-Adaptive-Avg")
            plt.text(260, lrb_adaptive_avg + 5000, 'Adaptive Avg. TP = ' + f'{lrb_adaptive_avg:,.2f}')
        if has_scheduling_only_metrics:
            plt.axhline(y=lrb_scheduling_avg, ls='--', color='y', label="LRB-Scheduling-Avg")
            plt.text(360, lrb_scheduling_avg + 5000, 'Scheduling Avg. TP = ' + f'{lrb_scheduling_avg:,.2f}')

        # ax.set_ylim(bottom=0)
        ax.set(xlabel="Time (sec)", ylabel="Throughput (event/sec)",
               title="Throughput [# Kafka partitions (same as # src replicas): " + num_parts + "]")
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
        plt.savefig(
            results_dir + "/throughput_" + parallelism_level + "_" + num_parts + "parts_" + experiment_date_id + ".png")
        plt.show()

        count_fig, count_ax = plt.subplots(figsize=(12, 6))

        count_ax.plot(lrb_default_src_tp_df["rel_time"], lrb_default_src_tp_df["count"],
                      label="LRB-Default")
        if has_pseudo_default_metrics:
            count_ax.plot(lrb_pseudo_default_src_df["rel_time"], lrb_pseudo_default_src_df["count"], label="LRB-PD")
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
        plt.savefig(
            results_dir + "/count_" + parallelism_level + "_" + num_parts + "parts_" + experiment_date_id + ".png")
        plt.show()
    else:
        lrb_default_op_name_id_dict = None
        lrb_replicating_op_name_id_dict = None
        lrb_scheduling_op_name_id_dict = None
        lrb_adaptive_op_name_id_dict = None

    if plot_latency:
        col_list = ["name", "time", "operator_id", "operator_subtask_index", "mean", "p50", "p95", "p99"]
        metric_name = "taskmanager_job_latency_source_id_operator_id_operator_subtask_index_latency"
        target_op_name = 'toll_win_1'
        target_stat = 'mean'
        all_latency_graph_y_top = 500

        print(lrb_default_op_name_id_dict)
        lrb_default_latency_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                                default_id_str, parallelism_level, default_sched_period, num_parts)
        target_op_id = lrb_default_op_name_id_dict[target_op_name]
        lrb_default_latency_df, lrb_default_latency_avg = get_formatted_latency(lrb_default_latency_file, col_list,
                                                                                lower_time_threshold,
                                                                                upper_time_threshold,
                                                                                default_offset,
                                                                                target_op_id, target_stat)

        lrb_default_latency_df_all = []
        lrb_default_latency_avg_all = []
        for default_sched_period in default_sched_periods:
            lrb_default_latency_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_default,
                                                    default_id_str, parallelism_level, default_sched_period, num_parts)
            temp_lrb_default_latency_df, temp_lrb_default_latency_avg = get_formatted_latency(lrb_default_latency_file,
                                                                                              col_list,
                                                                                              lower_time_threshold,
                                                                                              upper_time_threshold,
                                                                                              default_offset,
                                                                                              target_op_id, target_stat)
            lrb_default_latency_df_all.append(temp_lrb_default_latency_df)
            lrb_default_latency_avg_all.append(temp_lrb_default_latency_avg)

        lrb_default_pivoted_latency_df = get_pivoted_latency(lrb_default_latency_file, col_list, target_stat,
                                                             lrb_default_op_name_id_dict)
        fig_def_all, ax_def_all = plt.subplots(figsize=(8, 6))
        lrb_default_pivoted_latency_df.plot(x="rel_time", y=['prj_1', 'vehicle_win_1', 'toll_win_1', 'toll_acc_win_1',
                                                             'Sink: sink_1'], ax=ax_def_all)
        ax_def_all.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                       title="Default Latency (" + target_stat + ") - All Operators ")
        ax_def_all.set_ylim(top=all_latency_graph_y_top)
        plt.savefig(
            results_dir + "/latency_default_" + parallelism_level + "_all_" + target_stat + "_" + num_parts + "parts_" + experiment_date_id + ".png")
        plt.show()

        fig, ax = plt.subplots(figsize=(12, 6))

        for idx, lrb_default_latency_df in enumerate(lrb_default_latency_df_all):
            ax.plot(lrb_default_latency_df["rel_time"], lrb_default_latency_df[target_stat],
                    label="LRB-Default" + str(idx))

        # plt.axhline(y=lrb_default_latency_avg, ls='--', color='c', label="LRB-Default-Avg")
        # plt.text(100, lrb_default_latency_avg + 50,
        #          'Avg. ' + target_stat + ' latency (ms) - Default = ' + f'{lrb_default_latency_avg:,.2f}')

        if has_pseudo_default_metrics:
            pseudo_default_offset = 0
            lrb_pseudo_default_latency_file = get_filename(data_dir, experiment_date_id, metric_name,
                                                           file_date_adaptive, "lrb_pd", parallelism_level, "3600000",
                                                           num_parts)
            target_op_id = lrb_pseudo_default_op_name_id_dict[target_op_name]
            lrb_pseudo_default_sink_latency_df, lrb_pseudo_default_latency_avg = get_formatted_latency(
                lrb_pseudo_default_latency_file, col_list, lower_time_threshold, upper_time_threshold,
                pseudo_default_offset,
                target_op_id, target_stat)

            lrb_pseudo_default_pivoted_latency_df = get_pivoted_latency(lrb_pseudo_default_latency_file, col_list,
                                                                        target_stat,
                                                                        lrb_default_op_name_id_dict)

            ax.plot(lrb_pseudo_default_sink_latency_df["rel_time"], lrb_pseudo_default_sink_latency_df[target_stat],
                    label="LRB-PD")
            plt.axhline(y=lrb_pseudo_default_latency_avg, ls='--', color='y', label="LRB-PD-Avg")
            plt.text(360, lrb_pseudo_default_latency_avg - 50,
                     'Avg. ' + target_stat + ' latency (ms) - PD = ' + f'{lrb_pseudo_default_latency_avg:,.2f}')

        if has_scheduling_only_metrics:
            scheduling_offset = 0
            lrb_scheduling_latency_file = get_filename(data_dir, experiment_date_id, metric_name, file_date_adaptive,
                                                       "lrb_scheduling", parallelism_level, scheduling_period,
                                                       num_parts)
            target_op_id = lrb_scheduling_op_name_id_dict[target_op_name]
            lrb_scheduling_sink_latency_df, lrb_scheduling_latency_avg = get_formatted_latency(
                lrb_scheduling_latency_file, col_list, lower_time_threshold, upper_time_threshold, scheduling_offset,
                target_op_id, target_stat)

            lrb_scheduling_pivoted_latency_df = get_pivoted_latency(lrb_scheduling_latency_file, col_list, target_stat,
                                                                    lrb_default_op_name_id_dict)

            ax.plot(lrb_scheduling_sink_latency_df["rel_time"], lrb_scheduling_sink_latency_df[target_stat],
                    label="LRB-Scheduling")
            plt.axhline(y=lrb_scheduling_latency_avg, ls='--', color='y', label="LRB-Scheduling-Avg")
            plt.text(360, lrb_scheduling_latency_avg - 50,
                     'Avg. ' + target_stat + ' latency (ms) - Scheduling = ' + f'{lrb_scheduling_latency_avg:,.2f}')

        # ax.set_ylim(bottom=0)
        ax.set(xlabel="Time (sec)", ylabel="Latency (ms)",
               title="Latency (" + target_stat + ") - Operator: " + target_op_name + ", # Kafka partitions (same as # src replicas): " + num_parts)
        ax.set_ylim(0, 1000)
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
        plt.savefig(
            results_dir + "/latency_" + parallelism_level + "_" + target_op_name + "_" + target_stat + "_" + num_parts + "parts_" + experiment_date_id + ".png")
        plt.show()

        if has_pseudo_default_metrics:
            fig_pd_all, ax_pd_all = plt.subplots(figsize=(8, 6))
            lrb_pseudo_default_pivoted_latency_df.plot(x="rel_time",
                                                       y=['prj_1', 'vehicle_win_1', 'toll_win_1', 'toll_acc_win_1',
                                                          'Sink: sink_1'], ax=ax_pd_all)
            ax_pd_all.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                          title="PD Latency (" + target_stat + ") - All Operators ")
            ax_pd_all.set_ylim(top=all_latency_graph_y_top)
            plt.savefig(
                results_dir + "/latency_pd_" + parallelism_level + "_all_" + target_stat + "_" + num_parts + "parts_" + experiment_date_id + ".png")
            plt.show()
        if has_scheduling_only_metrics:
            fig_sched_all, ax_sched_all = plt.subplots(figsize=(8, 6))
            lrb_scheduling_pivoted_latency_df.plot(x="rel_time",
                                                   y=['prj_1', 'vehicle_win_1', 'toll_win_1', 'toll_acc_win_1',
                                                      'Sink: sink_1'], ax=ax_sched_all)
            ax_sched_all.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                             title="Scheduling Latency (" + target_stat + ") - All Operators ")
            ax_sched_all.set_ylim(top=all_latency_graph_y_top)
            plt.savefig(
                results_dir + "/latency_scheduling_" + parallelism_level + "_all_" + target_stat + "_" + num_parts + "parts_" + experiment_date_id + ".png")
            plt.show()
