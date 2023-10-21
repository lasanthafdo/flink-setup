import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys


def get_formatted_tput(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset):
    print("Reading file " + lrb_num_out_file + " filtering events between " + str(lower_threshold) + " and " + str(
        upper_threshold) + " seconds")
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    lrb_src_df = lrb_df[lrb_df['operator_name'].str.contains('Source:')].drop(
        ['name'], axis=1).groupby(['time'])[['rate', 'count']].sum().reset_index()
    lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_src_df = lrb_src_df.loc[
        (lrb_src_df['rel_time'] > lower_threshold) & (lrb_src_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_src_df['rate'])
    return lrb_src_df, lrb_avg


def get_filename(data_directory, exp_id, metric_name, file_date, sched_policy, par_level='12', sched_period='0',
                 num_parts='1', iter='0_1_', exp_host='tem104'):
    return data_directory + "/" + exp_id + \
        "/" + metric_name + "_" + exp_host + "_" + sched_policy + "_" + sched_period + \
        "ms_" + par_level + "_" + num_parts + "parts_iter" + iter + file_date + ".csv"


def get_iteration_id(iter, is_global_iter, scheduling_policy):
    return scheduling_policy + " - " + str(iter.split("_")[0] if is_global_iter else iter.split("_")[1])


def get_par_it(parallelism_level, iter_val):
    return "Parallel level " + parallelism_level + " iteration " + str(iter_val)


def get_op_name_id_mapping(lrb_tp_file):
    op_name_id_mapping_df = pd.read_csv(
        lrb_tp_file, usecols=['operator_name', 'operator_id']).drop_duplicates()
    op_name_id_dict = dict(
        zip(op_name_id_mapping_df['operator_name'], op_name_id_mapping_df['operator_id']))
    return op_name_id_dict


def calc_plot_graphs_for_metric(metric_name, scheduling_policy, lrb_offsets, lrb_labels, default_id_str,
                                default_sched_period, scheduling_period, num_iters, target_metric, user_ylabel,
                                simple_metric_name, user_xlabel="Elapsed Time (sec)", exp_host="tem104"):
    lrb_file_names = {}
    lrb_metric_dfs = {}
    lrb_metric_avgs = {}
    lrb_metric_avgs_per_iter = {}

    fig, ax = plt.subplots(figsize=(8, 5))
    for parallelism_level in range(1, 3):
        parallelism_level = str(parallelism_level)
        for iter_val in range(1, num_iters + 1):
            iter = str(iter_val) + "_" + "2" + \
                "_"
            iter_policy_id = iter + scheduling_policy

            lrb_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                          file_date,
                                                          scheduling_policy,
                                                          parallelism_level,
                                                          default_sched_period if scheduling_policy == "lrb_default" else scheduling_period,
                                                          src_parallelism, iter, exp_host)

            lrb_metric_dfs[iter_policy_id], lrb_metric_avgs[iter_policy_id] = get_formatted_tput(
                lrb_file_names[iter_policy_id], col_list,
                lower_time_threshold,
                upper_time_threshold,
                lrb_offsets[scheduling_policy] if lrb_offsets[scheduling_policy] >= 0 else lrb_offsets[
                    "lrb_default"])
            lrb_metric_avgs_per_iter[get_par_it(parallelism_level, iter_val)] = \
                lrb_metric_avgs[iter_policy_id]
            lrb_op_name_id_dicts[iter_policy_id] = get_op_name_id_mapping(
                lrb_file_names[iter_policy_id])
            ax.plot(lrb_metric_dfs[iter_policy_id]["rel_time"], lrb_metric_dfs[iter_policy_id][target_metric],
                    label=get_par_it(parallelism_level, iter_val))

        ax.set(xlabel=user_xlabel, ylabel=user_ylabel,
               title="Flink " + simple_metric_name)
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
    plt.savefig(results_dir + "/" + simple_metric_name.lower() + "_" + "flink_" +
                scheduling_policy + "_" + experiment_date_id + ".png")
    plt.show()

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    for iter, tp_avg in lrb_metric_avgs_per_iter.items():
        ax_bar.bar(iter, tp_avg)

    ax_bar.set(xlabel="Iteration", ylabel=user_ylabel,
               title="Flink " + simple_metric_name)
    ax_bar.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(results_dir + "/" + simple_metric_name.lower() + "_bar_" +
                "flink_" + parallelism_level + "_" + experiment_date_id + ".png")
    plt.show()

    print("Metric avgs" + str(lrb_metric_avgs_per_iter))
    return lrb_file_names, lrb_metric_dfs, ax


if __name__ == '__main__':
    data_dir = "/Users/jayzhou/Desktop/ura/k/flink-data/data"
    flink_metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
    lrb_offsets = {"lrb_default": 0, "lrb_pd": -1, "lrb_schedidling": -1, "lrb_scheduling": -1, "lrb_bpscheduling": -1,
                   "lrb_osdef": -1, "lrb_lqf": -1, "lrb_bposdef": -1, "lrb_bplqf": -1}
    lrb_labels = {"lrb_default": "LRB-Default", "lrb_pd": "LRB-PD", "lrb_scheduling": "LRB-Scheduling",
                  "lrb_schedidling": "LRB-Scheduling with blocking", "lrb_osdef": "LRB-OS default",
                  "lrb_bpscheduling": "LRB-Scheduling BP", "lrb_bposdef": "LRB-OS default BP",
                  "lrb_bplqf": "LRB-Largest Q First BP", "lrb_lqf": "LRB-Largest Q First"}
    experiment_date_id = "oct-20"
    file_date = "2023_10_20"
    src_parallelism = "1"
    default_sched_period = "5"
    scheduling_period = "5"
    num_iters = 5
    exp_host = "tem104"
    default_id_str = "lrb_osdef"
    scheduling_policy = "lrb_osdef"

    results_dir = "results/" + "combined"
    os.makedirs(results_dir, exist_ok=True)

    upper_time_threshold = 300
    lower_time_threshold = 0

    flink_col_list = ["name", "time", "operator_name",
                      "operator_id", "task_name", "subtask_index", "count", "rate"]
    alt_col_list = ["name", "time", "task_name", "subtask_index", "value"]
    flink_metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
    alt_metric_name = "taskmanager_job_task_numRecordsInChannelPerSecond"
    metric_name = flink_metric_name
    col_list = flink_col_list
    is_global_iter = True
    lrb_op_name_id_dicts = {}

    lrb_file_names, lrb_src_tp_dfs, ax = calc_plot_graphs_for_metric(flink_metric_name, scheduling_policy,
                                                                     lrb_offsets, lrb_labels,
                                                                     default_id_str, default_sched_period,
                                                                     scheduling_period, num_iters, "rate",
                                                                     "Rate (events/sec)", "Throughput",
                                                                     exp_host=exp_host)

    # count_fig, count_ax = plt.subplots(figsize=(12, 6))
    # for iter_val in range(1, num_iters + 1):
    #     iter = str(iter_val) + "_" + '2' + "_" + str(iter_val) + "_"
    #     iter_policy_id = iter + scheduling_policy
    #     count_ax.plot(lrb_src_tp_dfs[iter_policy_id]["rel_time"],
    #                   lrb_src_tp_dfs[iter_policy_id]["count"],
    #                   label=iter + lrb_labels[scheduling_policy])

    # # count_ax.set_ylim(bottom=0)
    # count_ax.set(xlabel="Time (sec)",
    #              ylabel="Total events", title="Event count")
    # count_ax.tick_params(axis="x", rotation=0)
    # count_ax.legend()
    # plt.savefig(
    #     results_dir + "/count_" + parallelism_level + "_" + experiment_date_id + ".png")
    # plt.show()
