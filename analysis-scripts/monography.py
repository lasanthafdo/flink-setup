# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import argparse
import json
import os
from statistics import fmean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_formatted_tput(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset,
                       target_op_for_tp):
    print("Reading file " + lrb_num_out_file + " filtering events between " + str(lower_threshold) + " and " + str(
        upper_threshold) + " seconds")
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    lrb_target_op_df = lrb_df[lrb_df['operator_name'].str.contains(target_op_for_tp)].drop(
        ['name'], axis=1).groupby(['time'])[['rate', 'count']].sum().reset_index()
    lrb_target_op_df['rel_time'] = lrb_target_op_df['time'].subtract(lrb_target_op_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_target_op_df = lrb_target_op_df.loc[
        (lrb_target_op_df['rel_time'] > lower_threshold) & (lrb_target_op_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_target_op_df['rate'])
    return lrb_target_op_df, lrb_avg


def get_formatted_alt_tput(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset):
    print("Reading file : " + lrb_num_out_file)
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    lrb_src_df = lrb_df[lrb_df['task_name'].str.contains('Source:')].drop(
        ['name'], axis=1).groupby(['time'])['value'].sum().reset_index()
    lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_src_df = lrb_src_df.loc[
        (lrb_src_df['rel_time'] > lower_threshold) & (lrb_src_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_src_df['value'])
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


def get_formatted_alt_latency(lrb_latency_file, column_list, lower_threshold, upper_threshold, offset, target_task_name,
                              target_stat):
    print("Reading file : " + lrb_latency_file)
    lrb_latency_df = pd.read_csv(lrb_latency_file, usecols=column_list)
    lrb_sink_latency_df = lrb_latency_df[lrb_latency_df['task_name'] == target_task_name].drop(
        ['name'], axis=1).groupby(['time'])[['mean', 'p50', 'p95', 'p99']].mean().reset_index()
    lrb_sink_latency_df['rel_time'] = lrb_sink_latency_df['time'].subtract(lrb_sink_latency_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_sink_latency_df = lrb_sink_latency_df.loc[
        (lrb_sink_latency_df['rel_time'] > lower_threshold) & (lrb_sink_latency_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_sink_latency_df[target_stat])
    return lrb_sink_latency_df, lrb_avg


def get_filename(data_directory, exp_id, metric_name, file_date, sched_policy, par_level='12', sched_period='0',
                 num_parts='1', iter='0_1_', exp_host='tem104'):
    return data_directory + "/" + exp_id + \
        "/" + metric_name + "_" + exp_host + "_" + sched_policy + "_" + sched_period + "ms_" + par_level + "_" + num_parts + "parts_iter" + iter + file_date + ".csv"


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
    return combined_df


def plot_metric(data_df, x_label, y_label, plot_title, group_by_col_name, plot_filename, exp_date_id, iter, y_max=-1):
    data_df.groupby(group_by_col_name)['value'].plot(legend=True)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    if y_max > 0:
        plt.ylim(top=y_max)
    plt.savefig(results_dir + "/" + plot_filename + "_" + exp_date_id + "_iter" + iter + ".png")
    plt.show()


def get_op_name_id_mapping(lrb_tp_file):
    op_name_id_mapping_df = pd.read_csv(lrb_tp_file, usecols=['operator_name', 'operator_id']).drop_duplicates()
    op_name_id_dict = dict(zip(op_name_id_mapping_df['operator_name'], op_name_id_mapping_df['operator_id']))
    return op_name_id_dict


def get_pivoted_latency(lrb_latency_file, column_list, target_stat, op_to_id_dict, upper_threshold, lower_threshold):
    lrb_all_latency_for_sched_mode = pd.read_csv(lrb_latency_file, usecols=column_list)
    join_dict_df = pd.DataFrame(op_to_id_dict.items(), columns=['operator_name', 'operator_id'])
    lrb_all_latency_for_sched_mode = pd.merge(lrb_all_latency_for_sched_mode, join_dict_df, on="operator_id")
    lrb_all_latency_for_sched_mode = lrb_all_latency_for_sched_mode.groupby(['time', 'operator_name'])[
        [target_stat]].mean().reset_index()
    lrb_pivoted_latency_df = lrb_all_latency_for_sched_mode.pivot(index='time', columns='operator_name',
                                                                  values=target_stat)
    lrb_pivoted_latency_df.columns = [''.join(col).strip() for col in lrb_pivoted_latency_df.columns]
    # col_order = ['time', 'fil_1', 'tsw_1', 'prj_1', 'vehicle_win_1', 'speed_win_1', 'acc_win_1', 'toll_win_1',
    #              'toll_acc_win_1', 'Sink: sink_1']
    col_order = ['time', 'fil_1', 'tsw_1', 'prj_1', 'Sink: sink_1']
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.reset_index()[col_order]
    lrb_pivoted_latency_df['rel_time'] = lrb_pivoted_latency_df['time'].subtract(
        lrb_pivoted_latency_df['time'].min()).div(1_000_000_000)
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.loc[
        (lrb_pivoted_latency_df['rel_time'] > lower_threshold) & (lrb_pivoted_latency_df['rel_time'] < upper_threshold)]

    return lrb_pivoted_latency_df


def get_pivoted_alt_latency(lrb_latency_file, column_list, target_stat, upper_threshold, lower_threshold):
    lrb_all_latency_for_sched_mode = pd.read_csv(lrb_latency_file, usecols=column_list)
    lrb_all_latency_for_sched_mode = lrb_all_latency_for_sched_mode.groupby(['time', 'task_name'])[
        [target_stat]].mean().reset_index()
    lrb_pivoted_latency_df = lrb_all_latency_for_sched_mode.pivot(index='time', columns='task_name',
                                                                  values=target_stat)
    lrb_pivoted_latency_df.columns = [''.join(col).strip() for col in lrb_pivoted_latency_df.columns]
    col_order = ['time', 'Sink: sink_1']
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.reset_index()[col_order]
    lrb_pivoted_latency_df['rel_time'] = lrb_pivoted_latency_df['time'].subtract(
        lrb_pivoted_latency_df['time'].min()).div(1_000_000_000)
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.loc[
        (lrb_pivoted_latency_df['rel_time'] > lower_threshold) & (lrb_pivoted_latency_df['rel_time'] < upper_threshold)]

    return lrb_pivoted_latency_df


def calc_plot_graphs_for_metric(metric_name, lrb_scheduling_policies, lrb_offsets, lrb_labels, default_sched_period,
                                scheduling_period, num_iters, target_metric, user_ylabel,
                                simple_metric_name, user_xlabel="Elapsed Time (sec)", exp_host="tem104",
                                target_op='fil_1'):
    lrb_file_names = {}
    lrb_metric_dfs = {}
    lrb_metric_avgs = {}
    metric_avgs_iter_dict = {}
    metric_avgs_policy_dict = {}
    lrb_current_policy_labels = []
    for scheduling_policy in lrb_scheduling_policies:
        current_policy_lbl = lrb_labels[scheduling_policy]
        lrb_current_policy_labels.append(current_policy_lbl)
        metric_avgs_policy_dict[current_policy_lbl] = list()
        fig, ax = plt.subplots(figsize=(8, 5))
        for iter_val in range(1, num_iters + 1):
            if iter_val in iter_to_skip: continue
            iter_lbl = "Iter " + str(iter_val)
            if not iter_lbl in metric_avgs_iter_dict:
                metric_avgs_iter_dict[iter_lbl] = list()
            iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
            iter_policy_id = iter + scheduling_policy
            if skip_default and scheduling_policy == "lrb_default":
                lrb_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                              file_date,
                                                              scheduling_policy,
                                                              parallelism_level,
                                                              default_sched_period if scheduling_policy == "lrb_default" else scheduling_period,
                                                              src_parallelism, iter, exp_host)
                lrb_metric_dfs[iter_policy_id], lrb_metric_avgs[iter_policy_id] = get_formatted_tput(
                    lrb_file_names[iter_policy_id], flink_col_list,
                    lower_time_threshold,
                    upper_time_threshold,
                    lrb_offsets[scheduling_policy] if lrb_offsets[scheduling_policy] >= 0 else lrb_offsets[
                        "lrb_default"], target_op)
                lrb_op_name_id_dicts[iter_policy_id] = get_op_name_id_mapping(lrb_file_names[iter_policy_id])
                ax._get_lines.get_next_color()
            else:
                lrb_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                              file_date,
                                                              scheduling_policy,
                                                              parallelism_level,
                                                              default_sched_period if scheduling_policy == "lrb_default" else scheduling_period,
                                                              src_parallelism, iter, exp_host)

            if not skip_default or scheduling_policy != "lrb_default":
                if use_alt_metrics:
                    lrb_metric_dfs[iter_policy_id], lrb_metric_avgs[iter_policy_id] = get_formatted_alt_tput(
                        lrb_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        lrb_offsets[scheduling_policy] if lrb_offsets[scheduling_policy] >= 0 else lrb_offsets[
                            "lrb_default"])
                    ax.plot(lrb_metric_dfs[iter_policy_id]["rel_time"], lrb_metric_dfs[iter_policy_id]["value"],
                            label=iter + current_policy_lbl)
                else:
                    lrb_metric_dfs[iter_policy_id], lrb_metric_avgs[iter_policy_id] = get_formatted_tput(
                        lrb_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        lrb_offsets[scheduling_policy] if lrb_offsets[scheduling_policy] >= 0 else lrb_offsets[
                            "lrb_default"], target_op)
                    # metric_avgs_iter_dict[get_iteration_id(iter, is_global_iter, lrb_labels[scheduling_policy])] = \
                    #     lrb_metric_avgs[iter_policy_id]
                    metric_avgs_policy_dict[current_policy_lbl].append(lrb_metric_avgs[iter_policy_id])
                    metric_avgs_iter_dict[iter_lbl].append(
                        scheduling_policy + ":" + str(lrb_metric_avgs[iter_policy_id]))
                    lrb_op_name_id_dicts[iter_policy_id] = get_op_name_id_mapping(lrb_file_names[iter_policy_id])
                    ax.plot(lrb_metric_dfs[iter_policy_id]["rel_time"], lrb_metric_dfs[iter_policy_id][target_metric],
                            label=iter + current_policy_lbl)

        ax.set(xlabel=user_xlabel, ylabel=user_ylabel,
               title=("Custom " if use_alt_metrics else "Flink ") + simple_metric_name)
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
        plt.savefig(results_dir + "/" + simple_metric_name.lower() + "_" + (
            "custom_" if use_alt_metrics else "flink_") + scheduling_policy + "_" + parallelism_level + "_" + experiment_date_id + ".png")
        plt.show()

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    print(metric_avgs_iter_dict)

    x = np.arange(len(lrb_scheduling_policies))
    width = 0.8 / num_iters  # the width of the bars
    multiplier = 0
    for itr, encoded_measurements in metric_avgs_iter_dict.items():
        offset = width * multiplier
        derived_measurements = []
        for idx, value in enumerate(encoded_measurements):
            policy = value.split(":")[0]
            assert lrb_scheduling_policies[idx] == policy
            derived_measurements.append(float(value.split(":")[1]))
        ax_bar.bar(x + offset, derived_measurements, width, label=itr)
        # ax_bar.bar_label(rects, padding=3)
        multiplier += 1

    line_colors = ['red', 'black', 'green', 'blue']
    color_cylce = 0
    for pol, measurements in metric_avgs_policy_dict.items():
        avg_metric = round(fmean(measurements))
        ax_bar.axhline(avg_metric, linestyle='--', color=line_colors[color_cylce])
        ax_bar.text(color_cylce, avg_metric + 10000, "Avg (" + pol + ") = " + str(avg_metric))
        color_cylce += 1

    ax_bar.set(xlabel="Scheduling Policy", ylabel=user_ylabel,
               title=("Custom " if use_alt_metrics else "Flink ") + simple_metric_name)
    ax_bar.set_xticks(x + width, lrb_current_policy_labels)
    ax_bar.legend(loc="lower right")
    # ax_bar.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(results_dir + "/" + simple_metric_name.lower() + "_bar_" + (
        "custom_" if use_alt_metrics else "flink_") + parallelism_level + "_" + experiment_date_id + ".png")
    plt.show()

    print("Metric avgs" + str(metric_avgs_iter_dict))
    return lrb_file_names, lrb_metric_dfs


def get_iteration_id(iter, is_global_iter, scheduling_policy):
    return scheduling_policy + " - " + str(iter.split("_")[0] if is_global_iter else iter.split("_")[1])


def load_config():
    try:
        with open("config.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        raise Exception("Configuration file 'config.json' not found. Please create it based on 'config.example.json'.")
    except json.JSONDecodeError:
        raise Exception("Error parsing 'config.json'. Please ensure it's correctly formatted.")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("expdate_id")
    parser.add_argument("filedate")
    parser.add_argument("-p", "--parallelism", default="1")
    parser.add_argument("-sp", "--src_parallelism", default="1")
    parser.add_argument("-i", "--numiters", default=5, type=int)
    parser.add_argument("-def", "--defaultid", default="lrb_osdef")
    parser.add_argument("-pol", "--policies", default="lrb_osdef")
    parser.add_argument("--host", default="tem104")
    parser.add_argument("-scp", "--schedperiod", default="5")
    args = parser.parse_args()
    config = load_config()
    data_dir = config["data_dir"]
    experiment_date_id = args.expdate_id
    file_date = args.filedate
    parallelism_level = args.parallelism
    src_parallelism = args.src_parallelism
    results_dir = "results/" + experiment_date_id + "/par_" + parallelism_level
    os.makedirs(results_dir, exist_ok=True)
    scheduling_period = args.schedperiod

    upper_time_threshold = 3600
    lower_time_threshold = 0
    plot_tp = True
    plot_latency = True
    plot_event_time_latency = True
    plot_cpu = True
    plot_mem = True
    plot_busy = True
    plot_idle = True
    plot_backpressure = True
    plot_iq_len = True
    plot_nw = True

    has_pseudo_default_metrics = False
    has_replicating_only_metrics = False
    has_scheduling_only_metrics = False
    has_fcfsp_metrics = False
    has_lqf_metrics = False
    has_lqsf_metrics = False
    has_lsf_metrics = False
    has_rr_metrics = False
    has_dummy_metrics = False
    has_adaptive_metrics = False

    skip_default = False
    use_alt_metrics = False

    default_id_str = args.defaultid
    num_iters = args.numiters
    exp_host = args.host
    lrb_scheduling_policies = args.policies.split(",")
    default_sched_period = scheduling_period
    lrb_offsets = {"lrb_default": 0, "lrb_def116": 0, "lrb_pd": -1, "lrb_schedidling": -1, "lrb_scheduling": -1,
                   "lrb_bpscheduling": -1,
                   "lrb_osdef": -1, "lrb_lqf": -1, "lrb_bposdef": -1, "lrb_bplqf": -1, "lrb_bpmitigation": -1,
                   "lrb_bplatency": -1, "lrb_dynbuffers": -1, "lrb_osdefbpoff": -1, "lrb_adaptltncy": -1}
    lrb_labels = {"lrb_default": "LRB-Default", "lrb_def116": "LRB-Default (1.16.x)", "lrb_pd": "LRB-PD",
                  "lrb_scheduling": "LRB-Scheduling",
                  "lrb_schedidling": "LRB-Scheduling with blocking", "lrb_osdef": "LRB-OS default",
                  "lrb_bpscheduling": "LRB-Scheduling BP", "lrb_bposdef": "LRB-OS default BP",
                  "lrb_bplqf": "LRB-Largest Q First BP", "lrb_lqf": "LRB-Largest Q First",
                  "lrb_bpmitigation": "LRB-Backpressure Mitigation", "lrb_bplatency": "LRB-BP Latency Mitigation",
                  "lrb_dynbuffers": "LRB-Dynamic Buffers", "lrb_osdefbpoff": "LRB-OS default BP off",
                  "lrb_adaptltncy": "LRB-Adaptive Buffers"}
    lrb_op_name_id_dicts = {}
    iter_to_skip = []
    local_iter_default = "2"
    is_global_iter = True
    tgt_op_for_tp = "fil_1"

    if plot_tp:
        flink_col_list = ["name", "time", "operator_name", "operator_id", "task_name", "subtask_index", "count", "rate"]
        alt_col_list = ["name", "time", "task_name", "subtask_index", "value"]
        flink_metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
        alt_metric_name = "taskmanager_job_task_numRecordsInChannelPerSecond"
        if use_alt_metrics:
            metric_name = alt_metric_name
            col_list = alt_col_list
        else:
            metric_name = flink_metric_name
            col_list = flink_col_list

        lrb_file_names, lrb_src_tp_dfs = calc_plot_graphs_for_metric(flink_metric_name, lrb_scheduling_policies,
                                                                     lrb_offsets, lrb_labels,
                                                                     default_sched_period,
                                                                     scheduling_period, num_iters, "rate",
                                                                     "Rate (events/sec)", "Throughput",
                                                                     exp_host=exp_host, target_op=tgt_op_for_tp)

        if not use_alt_metrics:
            count_fig, count_ax = plt.subplots(figsize=(12, 6))

            for scheduling_policy in lrb_scheduling_policies:
                for iter_val in range(1, num_iters + 1):
                    if iter_val in iter_to_skip: continue
                    iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(
                        iter_val) + "_"
                    iter_policy_id = iter + scheduling_policy
                    if not skip_default or scheduling_policy != "lrb_default":
                        count_ax.plot(lrb_src_tp_dfs[iter_policy_id]["rel_time"],
                                      lrb_src_tp_dfs[iter_policy_id]["count"],
                                      label=iter + lrb_labels[scheduling_policy])

            # count_ax.set_ylim(bottom=0)
            count_ax.set(xlabel="Time (sec)", ylabel="Total events", title="Event count")
            count_ax.tick_params(axis="x", rotation=0)
            count_ax.legend()
            plt.savefig(
                results_dir + "/count_" + parallelism_level + "_" + experiment_date_id + ".png")
            plt.show()

        if not use_alt_metrics:
            for iter_val in range(1, num_iters + 1):
                if iter_val in iter_to_skip: continue
                iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
                iter_policy_id = iter + default_id_str
                lrb_default_df = pd.read_csv(lrb_file_names[iter_policy_id], usecols=flink_col_list)
                src_task_indexes = lrb_default_df[lrb_default_df['operator_name'].str.contains('Source:')][
                    'subtask_index'].unique()
                other_task_indexes = lrb_default_df[lrb_default_df['operator_name'].str.contains('tsw')][
                    'subtask_index'].unique()
                src_task_indexes.sort()
                other_task_indexes.sort()
                print("Source subtasks: " + str(src_task_indexes))
                print("Other subtasks: " + str(other_task_indexes))

    else:
        lrb_default_op_name_id_dict = None
        lrb_pseudo_default_op_name_id_dict = None
        lrb_fcfsp_op_name_id_dict = None
        lrb_lqf_op_name_id_dict = None
        lrb_lqsf_op_name_id_dict = None
        lrb_lsf_op_name_id_dict = None
        lrb_rr_op_name_id_dict = None
        lrb_dummy_op_name_id_dict = None
        lrb_replicating_op_name_id_dict = None
        lrb_scheduling_op_name_id_dict = None
        lrb_adaptive_op_name_id_dict = None

    if plot_latency:
        col_list = ["name", "time", "operator_id", "operator_subtask_index", "mean", "p50", "p95", "p99"]
        metric_name = "taskmanager_job_latency_source_id_operator_id_operator_subtask_index_latency"
        target_op_name = 'Sink: sink_1'
        target_stat = 'mean'
        all_latency_graph_y_top = 300
        lrb_latency_file_names = {}
        lrb_latency_dfs = {}
        lrb_latency_avgs = {}
        lrb_latency_pivoted_dfs = {}

        for scheduling_policy in lrb_scheduling_policies:
            fig_all_ops, ax_all_ops = plt.subplots(figsize=(8, 6))
            for iter_val in range(1, num_iters + 1):
                if iter_val in iter_to_skip: continue
                iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
                iter_policy_id = iter + scheduling_policy
                if not skip_default or scheduling_policy != "lrb_default":
                    lrb_latency_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                                          file_date,
                                                                          scheduling_policy, parallelism_level,
                                                                          default_sched_period if scheduling_policy == "lrb_default" else scheduling_period,
                                                                          src_parallelism, iter, exp_host)
                    print(lrb_op_name_id_dicts[iter + default_id_str])
                    target_op_id = lrb_op_name_id_dicts[iter + default_id_str][target_op_name]
                    lrb_latency_dfs[iter_policy_id], lrb_latency_avgs[
                        get_iteration_id(iter, is_global_iter, scheduling_policy)] = get_formatted_latency(
                        lrb_latency_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        lrb_offsets["lrb_default"],
                        target_op_id, target_stat)
                    lrb_latency_pivoted_dfs[iter_policy_id] = get_pivoted_latency(
                        lrb_latency_file_names[iter_policy_id],
                        col_list,
                        target_stat,
                        lrb_op_name_id_dicts[iter_policy_id],
                        upper_time_threshold,
                        lower_time_threshold)

                    print(lrb_latency_pivoted_dfs[iter_policy_id])
                    ax_all_ops.plot(lrb_latency_pivoted_dfs[iter_policy_id]["rel_time"],
                                    lrb_latency_pivoted_dfs[iter_policy_id]["Sink: sink_1"],
                                    label="Sink - Iter: " + get_iteration_id(iter, is_global_iter, scheduling_policy))

            ax_all_ops.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                           title=lrb_labels[
                                     scheduling_policy] + " Latency (" + target_stat + ") - All Operators ")
            ax_all_ops.legend()
            ax_all_ops.set_ylim(bottom=0)
            plt.savefig(
                results_dir + "/latency_flink_" + scheduling_policy + "_" + parallelism_level + "_all_" + target_stat + "_" + experiment_date_id + ".png")
            plt.show()

        print("Latency avgs: " + str(lrb_latency_avgs))
        fig_lat, ax = plt.subplots(figsize=(8, 5))
        for iter, lat_avg in lrb_latency_avgs.items():
            ax.bar(iter, lat_avg)

        ax.set(xlabel="Iteration", ylabel="Time (ms)",
               title="Flink Latency (" + target_stat + ") - Operator: " + target_op_name)
        ax.tick_params(axis="x", rotation=90)
        # ax.set_ylim(0, all_latency_graph_y_top)
        ax.set_ylim(bottom=0)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            results_dir + "/latency_bar_flink_" + parallelism_level + "_" + target_op_name + "_" + target_stat + "_" + experiment_date_id + ".png")
        plt.show()

    if plot_event_time_latency:
        col_list = ["name", "time", "subtask_index", "task_name", "mean", "p50", "p95", "p99"]
        metric_name = "taskmanager_job_task_latencyFromGenerationHistogram"
        target_op_name = 'Sink: sink_1'
        target_stat = 'mean'
        all_latency_graph_y_top = 300
        et_latency_file_names = {}
        et_latency_dfs = {}
        et_latency_avgs_iter_dict = {}
        et_latency_avgs_policy_dict = {}
        et_latency_pivoted_dfs = {}
        lrb_current_policy_labels = []

        for scheduling_policy in lrb_scheduling_policies:
            current_policy_lbl = lrb_labels[scheduling_policy]
            lrb_current_policy_labels.append(current_policy_lbl)
            et_latency_avgs_policy_dict[current_policy_lbl] = list()
            fig_all_ops, ax_all_ops = plt.subplots(figsize=(8, 6))
            for iter_val in range(1, num_iters + 1):
                if iter_val in iter_to_skip: continue
                iter_lbl = "Iter " + str(iter_val)
                if not iter_lbl in et_latency_avgs_iter_dict:
                    et_latency_avgs_iter_dict[iter_lbl] = list()
                iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
                iter_policy_id = iter + scheduling_policy
                if not skip_default or scheduling_policy != "lrb_default":
                    et_latency_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                                          file_date,
                                                                          scheduling_policy, parallelism_level,
                                                                          default_sched_period if scheduling_policy == "lrb_default" else scheduling_period,
                                                                          src_parallelism, iter, exp_host)
                    et_latency_dfs[iter_policy_id], et_lat_avg = get_formatted_alt_latency(
                        et_latency_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        lrb_offsets["lrb_default"],
                        target_op_name, target_stat)
                    et_latency_avgs_iter_dict[iter_lbl].append(scheduling_policy + ":" + str(et_lat_avg))
                    et_latency_avgs_policy_dict[current_policy_lbl].append(et_lat_avg)

                    et_latency_pivoted_dfs[iter_policy_id] = get_pivoted_alt_latency(
                        et_latency_file_names[iter_policy_id],
                        col_list,
                        target_stat,
                        upper_time_threshold,
                        lower_time_threshold)

                    print(et_latency_pivoted_dfs[iter_policy_id])
                    ax_all_ops.plot(et_latency_pivoted_dfs[iter_policy_id]["rel_time"],
                                    et_latency_pivoted_dfs[iter_policy_id]["Sink: sink_1"],
                                    label="Sink - Iter: " + get_iteration_id(iter, is_global_iter, scheduling_policy))

            ax_all_ops.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                           title=lrb_labels[
                                     scheduling_policy] + " Latency (" + target_stat + ") - All Operators ")
            ax_all_ops.legend()
            ax_all_ops.set_ylim(bottom=0)
            plt.savefig(
                results_dir + "/latency_gen_to_sink_" + scheduling_policy + "_" + parallelism_level + "_all_" + target_stat + "_" + experiment_date_id + ".png")
            plt.show()

        print("Latency avgs: " + str(et_latency_avgs_iter_dict))
        fig_lat, ax_lat = plt.subplots(figsize=(8, 5))

        x = np.arange(len(lrb_scheduling_policies))
        width = 0.8 / num_iters  # the width of the bars
        multiplier = 0
        for itr, encoded_measurements in et_latency_avgs_iter_dict.items():
            offset = width * multiplier
            derived_measurements = []
            for idx, value in enumerate(encoded_measurements):
                policy = value.split(":")[0]
                assert lrb_scheduling_policies[idx] == policy
                derived_measurements.append(float(value.split(":")[1]))
            ax_lat.bar(x + offset, derived_measurements, width, label=itr)
            # ax_bar.bar_label(rects, padding=3)
            multiplier += 1

        line_colors = ['red', 'black', 'green', 'blue']
        color_cylce = 0
        for pol, measurements in et_latency_avgs_policy_dict.items():
            avg_metric = round(fmean(measurements))
            ax_lat.axhline(avg_metric, linestyle='--', color=line_colors[color_cylce])
            ax_lat.text(color_cylce, avg_metric + 10000, "Avg (" + pol + ") = " + str(avg_metric))
            color_cylce += 1

        # for iter, lat_avg in et_latency_avgs.items():
        #     ax_lat.bar(iter, lat_avg)

        ax_lat.set(xlabel="Iteration", ylabel="Time (ms)",
               title="Gen-to-sink Latency (" + target_stat + ") - Operator: " + target_op_name)
        ax_lat.set_xticks(x + width, lrb_current_policy_labels)
        ax_lat.set_ylim(bottom=0)
        ax_lat.legend()
        plt.tight_layout()
        plt.savefig(
            results_dir + "/latency_bar_gen_to_sink_" + parallelism_level + "_" + target_op_name + "_" + target_stat + "_" + experiment_date_id + ".png")
        plt.show()

    exit(0)
    if plot_cpu:
        lrb_default_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                  file_date, default_id_str, parallelism_level,
                                                  default_sched_period, src_parallelism, iter, exp_host)
        if has_replicating_only_metrics:
            lrb_replicating_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                          file_date, "lrb_replicating", parallelism_level,
                                                          scheduling_period, src_parallelism, iter, exp_host)
        else:
            lrb_replicating_cpu_usage_file = None

        if has_adaptive_metrics:
            lrb_adaptive_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                       file_date, "lrb_adaptive", parallelism_level,
                                                       scheduling_period, src_parallelism, iter, exp_host)
        else:
            lrb_adaptive_cpu_usage_file = None

        if has_scheduling_only_metrics:
            lrb_scheduling_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                         file_date, "lrb_scheduling", parallelism_level,
                                                         scheduling_period, src_parallelism, iter, exp_host)
        else:
            lrb_scheduling_cpu_usage_file = None

        cpu_usage_col_list = ["name", "time", "value"]
        cpu_usage_df = pd.read_csv(lrb_default_cpu_usage_file, usecols=cpu_usage_col_list)
        cpu_usage_df['rel_time'] = cpu_usage_df['time'].subtract(cpu_usage_df['time'].min()).div(
            1_000_000_000).subtract(lrb_offsets["lrb_default"])
        cpu_usage_df = cpu_usage_df.loc[cpu_usage_df['rel_time'] > 0]
        cpu_usage_df.describe()

        if has_replicating_only_metrics:
            repl_cpu_usage_df = pd.read_csv(lrb_replicating_cpu_usage_file, usecols=cpu_usage_col_list)
            repl_cpu_usage_df['rel_time'] = repl_cpu_usage_df['time'].subtract(repl_cpu_usage_df['time'].min()).div(
                1_000_000_000)
            # print(repl_cpu_usage_df)
        else:
            repl_cpu_usage_df = None

        if has_adaptive_metrics:
            adapt_cpu_usage_df = pd.read_csv(lrb_adaptive_cpu_usage_file, usecols=cpu_usage_col_list)
            adapt_cpu_usage_df['rel_time'] = adapt_cpu_usage_df['time'].subtract(adapt_cpu_usage_df['time'].min()).div(
                1_000_000_000)
            # print(adapt_cpu_usage_df)
        else:
            adapt_cpu_usage_df = None

        if has_scheduling_only_metrics:
            sched_cpu_usage_df = pd.read_csv(lrb_scheduling_cpu_usage_file, usecols=cpu_usage_col_list)
            sched_cpu_usage_df['rel_time'] = sched_cpu_usage_df['time'].subtract(sched_cpu_usage_df['time'].min()).div(
                1_000_000_000)
            # print(sched_cpu_usage_df)
        else:
            sched_cpu_usage_df = None

        cpu_fig, cpu_ax = plt.subplots(figsize=(12, 8))

        cpu_ax.plot(cpu_usage_df["rel_time"], cpu_usage_df["value"], label="LRB-Default")
        if has_replicating_only_metrics:
            cpu_ax.plot(repl_cpu_usage_df["rel_time"], repl_cpu_usage_df["value"], label="LRB-Replicating")
        if has_adaptive_metrics:
            cpu_ax.plot(adapt_cpu_usage_df["rel_time"], adapt_cpu_usage_df["value"], label="LRB-Adaptive")
        if has_scheduling_only_metrics:
            cpu_ax.plot(sched_cpu_usage_df["rel_time"], sched_cpu_usage_df["value"], label="LRB-Scheduling")

        # cpu_ax.set_ylim(bottom=0)
        cpu_ax.set(xlabel="Time (sec)", ylabel="CPU Usage (%)", title="CPU Usage")
        cpu_ax.tick_params(axis="x", rotation=0)
        cpu_ax.legend()
        plt.savefig(results_dir + "/cpu_" + parallelism_level + "_" + experiment_date_id + "_iter" + iter + ".png")
        plt.show()

    if plot_mem:
        lrb_default_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                  default_id_str, parallelism_level, default_sched_period,
                                                  src_parallelism, iter, exp_host)
        if has_replicating_only_metrics:
            lrb_replicating_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                          "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                          "lrb_replicating", parallelism_level, scheduling_period,
                                                          src_parallelism, iter, exp_host)
        else:
            lrb_replicating_mem_usage_file = None

        if has_adaptive_metrics:
            lrb_adaptive_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                       "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                       "lrb_adaptive", parallelism_level, scheduling_period,
                                                       src_parallelism, iter, exp_host)
        else:
            lrb_adaptive_mem_usage_file = None

        if has_scheduling_only_metrics:
            lrb_scheduling_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                         "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                         "lrb_scheduling", parallelism_level, scheduling_period,
                                                         src_parallelism, iter, exp_host)
        else:
            lrb_scheduling_mem_usage_file = None

        mem_usage_col_list = ["name", "time", "value"]
        mem_usage_df = pd.read_csv(lrb_default_mem_usage_file, usecols=mem_usage_col_list)
        mem_usage_df['rel_time'] = mem_usage_df['time'].subtract(mem_usage_df['time'].min()).div(
            1_000_000_000).subtract(lrb_offsets["lrb_default"])
        mem_usage_df = mem_usage_df.loc[mem_usage_df['rel_time'] > 0]
        mem_usage_df['value'] = mem_usage_df['value'].div(1048576)
        # print(mem_usage_df)

        if has_replicating_only_metrics:
            repl_mem_usage_df = pd.read_csv(lrb_replicating_mem_usage_file, usecols=mem_usage_col_list)
            repl_mem_usage_df['rel_time'] = repl_mem_usage_df['time'].subtract(repl_mem_usage_df['time'].min()).div(
                1_000_000_000)
            repl_mem_usage_df['value'] = repl_mem_usage_df['value'].div(1048576)
            # print(repl_mem_usage_df)
        else:
            repl_mem_usage_df = None

        if has_adaptive_metrics:
            adapt_mem_usage_df = pd.read_csv(lrb_adaptive_mem_usage_file, usecols=mem_usage_col_list)
            adapt_mem_usage_df['rel_time'] = adapt_mem_usage_df['time'].subtract(adapt_mem_usage_df['time'].min()).div(
                1_000_000_000)
            adapt_mem_usage_df['value'] = adapt_mem_usage_df['value'].div(1048576)
            # print(adapt_mem_usage_df)
        else:
            adapt_mem_usage_df = None

        if has_scheduling_only_metrics:
            sched_mem_usage_df = pd.read_csv(lrb_scheduling_mem_usage_file, usecols=mem_usage_col_list)
            sched_mem_usage_df['rel_time'] = sched_mem_usage_df['time'].subtract(sched_mem_usage_df['time'].min()).div(
                1_000_000_000)
            sched_mem_usage_df['value'] = sched_mem_usage_df['value'].div(1048576)
            # print(sched_mem_usage_df)
        else:
            sched_mem_usage_df = None

        mem_fig, mem_ax = plt.subplots(figsize=(12, 8))

        mem_ax.plot(mem_usage_df["rel_time"], mem_usage_df["value"], label="LRB-Default")
        if has_replicating_only_metrics:
            mem_ax.plot(repl_mem_usage_df["rel_time"], repl_mem_usage_df["value"], label="LRB-Replicating")
        if has_adaptive_metrics:
            mem_ax.plot(adapt_mem_usage_df["rel_time"], adapt_mem_usage_df["value"], label="LRB-Adaptive")
        if has_scheduling_only_metrics:
            mem_ax.plot(sched_mem_usage_df["rel_time"], sched_mem_usage_df["value"], label="LRB-Scheduling")

        # mem_ax.set_ylim(bottom=0)
        mem_ax.set(xlabel="Time (sec)", ylabel="Memory Usage (MB)", title="Heap Memory")
        mem_ax.tick_params(axis="x", rotation=0)
        mem_ax.legend()
        plt.savefig(results_dir + "/mem_" + parallelism_level + "_" + experiment_date_id + "_iter" + iter + ".png")
        plt.show()

    if plot_busy:
        busy_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        x_label = "Time (sec)"
        y_label = "ms/sec"
        plot_title_base = "Busy Time (ms/sec) - "
        plot_filename_base = "busy_time_"
        plot_filename_base_for_grouped_plots = "busy_time_grouped_"
        group_by_col_name = "task_name"
        group_by_col_name_for_grouped_plots = "task"

        lrb_default_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_job_task_busyTimeMsPerSecond", file_date,
                                                  default_id_str, parallelism_level, default_sched_period,
                                                  src_parallelism, iter, exp_host)
        busy_time_df = pd.read_csv(lrb_default_busy_time_file, usecols=busy_time_col_list)
        busy_time_grouped_df = busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
        busy_time_grouped_df['rel_time'] = busy_time_grouped_df['time'].subtract(
            busy_time_grouped_df['time'].min()).div(1_000_000_000)
        plot_metric(busy_time_grouped_df, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name, plot_filename_base + "default", experiment_date_id, iter)

        if has_replicating_only_metrics:
            lrb_replicating_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                          "taskmanager_job_task_busyTimeMsPerSecond",
                                                          file_date, "lrb_replicating", parallelism_level,
                                                          scheduling_period, src_parallelism, iter, exp_host)
            repl_busy_time_df = pd.read_csv(lrb_replicating_busy_time_file, usecols=busy_time_col_list)
            repl_busy_time_grouped_df = repl_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
            repl_busy_time_grouped_df['rel_time'] = repl_busy_time_grouped_df['time'].subtract(
                repl_busy_time_grouped_df['time'].min()).div(1_000_000_000)
            plot_metric(repl_busy_time_grouped_df, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name, plot_filename_base + "replicating", experiment_date_id, iter)

        if has_adaptive_metrics:
            lrb_adaptive_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                       "taskmanager_job_task_busyTimeMsPerSecond", file_date,
                                                       "lrb_adaptive", parallelism_level, scheduling_period,
                                                       src_parallelism,
                                                       iter, exp_host)
            adapt_busy_time_df = pd.read_csv(lrb_adaptive_busy_time_file, usecols=busy_time_col_list)
            adapt_busy_time_grouped_df = adapt_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
            adapt_busy_time_grouped_df['rel_time'] = adapt_busy_time_grouped_df['time'].subtract(
                adapt_busy_time_grouped_df['time'].min()).div(1_000_000_000)
            plot_metric(adapt_busy_time_grouped_df, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name, plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            lrb_scheduling_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                         "taskmanager_job_task_busyTimeMsPerSecond", file_date,
                                                         "lrb_scheduling", parallelism_level, scheduling_period,
                                                         src_parallelism, iter, exp_host)
            sched_busy_time_df = pd.read_csv(lrb_scheduling_busy_time_file, usecols=busy_time_col_list)
            sched_busy_time_grouped_df = sched_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
            sched_busy_time_grouped_df['rel_time'] = sched_busy_time_grouped_df['time'].subtract(
                sched_busy_time_grouped_df['time'].min()).div(1_000_000_000)
            plot_metric(sched_busy_time_grouped_df, x_label, y_label, plot_title_base + "Scheduling",
                        group_by_col_name, plot_filename_base + "scheduling", experiment_date_id, iter)

        # Grouped analytics
        busy_time_grouped_df.loc[
            busy_time_grouped_df['task_name'] == 'fil_1 -> tsw_1 -> prj_1', 'task'] = 'TSW+'
        busy_time_grouped_df.loc[
            (busy_time_grouped_df['task_name'] == 'speed_win_1 -> Map') | (
                    busy_time_grouped_df['task_name'] == 'acc_win_1 -> Map') | (
                    busy_time_grouped_df['task_name'] == 'vehicle_win_1 -> Map'), 'task'] = 'Upstream Windows'
        busy_time_grouped_df.loc[
            (busy_time_grouped_df['task_name'] == 'toll_acc_win_1 -> Sink: sink_1') | (
                    busy_time_grouped_df['task_name'] == 'toll_win_1 -> Map'), 'task'] = 'Downstream Windows'
        default_busy_time_final = busy_time_grouped_df.groupby(['rel_time', 'task'])['value'].mean().reset_index()
        # print(default_busy_time_final)

        default_busy_time_final.set_index('rel_time', inplace=True)

        plot_metric(default_busy_time_final, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name_for_grouped_plots, plot_filename_base_for_grouped_plots + "default",
                    experiment_date_id, iter)

        if has_replicating_only_metrics:
            repl_busy_time_grouped_df.loc[
                repl_busy_time_grouped_df[
                    'task_name'] == 'fil_1 -> tsw_1 -> prj_1', 'task'] = 'TSW+'
            repl_busy_time_grouped_df.loc[
                (repl_busy_time_grouped_df['task_name'] == 'speed_win_1 -> Map') | (
                        repl_busy_time_grouped_df['task_name'] == 'acc_win_1 -> Map') | (
                        repl_busy_time_grouped_df['task_name'] == 'vehicle_win_1 -> Map'), 'task'] = 'Upstream Windows'
            repl_busy_time_grouped_df.loc[
                (repl_busy_time_grouped_df['task_name'] == 'toll_acc_win_1 -> Sink: sink_1') | (
                        repl_busy_time_grouped_df['task_name'] == 'toll_win_1 -> Map'), 'task'] = 'Downstream Windows'
            repl_busy_time_final = repl_busy_time_grouped_df.groupby(['rel_time', 'task'])['value'].mean().reset_index()
            repl_busy_time_final.set_index('rel_time', inplace=True)

            plot_metric(repl_busy_time_final, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name_for_grouped_plots, plot_filename_base_for_grouped_plots + "replicating",
                        experiment_date_id, iter)

        if has_adaptive_metrics:
            adapt_busy_time_grouped_df.loc[
                adapt_busy_time_grouped_df[
                    'task_name'] == 'fil_1 -> tsw_1 -> prj_1', 'task'] = 'TSW+'
            adapt_busy_time_grouped_df.loc[
                (adapt_busy_time_grouped_df['task_name'] == 'speed_win_1 -> Map') | (
                        adapt_busy_time_grouped_df['task_name'] == 'acc_win_1 -> Map') | (
                        adapt_busy_time_grouped_df['task_name'] == 'vehicle_win_1 -> Map'), 'task'] = 'Upstream Windows'
            adapt_busy_time_grouped_df.loc[
                (adapt_busy_time_grouped_df['task_name'] == 'toll_acc_win_1 -> Sink: sink_1') | (
                        adapt_busy_time_grouped_df['task_name'] == 'toll_win_1 -> Map'), 'task'] = 'Downstream Windows'
            adapt_busy_time_final = adapt_busy_time_grouped_df.groupby(['rel_time', 'task'])[
                'value'].mean().reset_index()
            adapt_busy_time_final.set_index('rel_time', inplace=True)

            plot_metric(adapt_busy_time_final, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name_for_grouped_plots, plot_filename_base_for_grouped_plots + "adaptive",
                        experiment_date_id, iter)

        if has_scheduling_only_metrics:
            sched_busy_time_grouped_df.loc[
                sched_busy_time_grouped_df[
                    'task_name'] == 'fil_1 -> tsw_1 -> prj_1', 'task'] = 'TSW+'
            sched_busy_time_grouped_df.loc[
                (sched_busy_time_grouped_df['task_name'] == 'speed_win_1 -> Map') | (
                        sched_busy_time_grouped_df['task_name'] == 'acc_win_1 -> Map') | (
                        sched_busy_time_grouped_df['task_name'] == 'vehicle_win_1 -> Map'), 'task'] = 'Upstream Windows'
            sched_busy_time_grouped_df.loc[
                (sched_busy_time_grouped_df['task_name'] == 'toll_acc_win_1 -> Sink: sink_1') | (
                        sched_busy_time_grouped_df['task_name'] == 'toll_win_1 -> Map'), 'task'] = 'Downstream Windows'
            sched_busy_time_final = sched_busy_time_grouped_df.groupby(['rel_time', 'task'])[
                'value'].mean().reset_index()
            sched_busy_time_final.set_index('rel_time', inplace=True)

            plot_metric(sched_busy_time_final, x_label, y_label, plot_title_base + "Scheduling",
                        group_by_col_name_for_grouped_plots, plot_filename_base_for_grouped_plots + "scheduling",
                        experiment_date_id, iter)

    if plot_idle:
        idle_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        x_label = "Time (sec)"
        y_label = "ms/sec"
        plot_title_base = "Idle Time (ms/sec) - "
        plot_filename_base = "idle_time_"
        group_by_col_name = "task_name"

        lrb_default_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_job_task_idleTimeMsPerSecond", file_date,
                                                  default_id_str, parallelism_level, default_sched_period,
                                                  src_parallelism,
                                                  iter, exp_host)
        idle_time_grouped_df = get_grouped_df(idle_time_col_list, lrb_default_idle_time_file)
        plot_metric(idle_time_grouped_df, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name, plot_filename_base + "default", experiment_date_id, iter)

        if has_replicating_only_metrics:
            lrb_replicating_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                          "taskmanager_job_task_idleTimeMsPerSecond",
                                                          file_date, "lrb_replicating", parallelism_level,
                                                          scheduling_period, src_parallelism, iter, exp_host)
            repl_idle_time_grouped_df = get_grouped_df(idle_time_col_list, lrb_replicating_idle_time_file)
            plot_metric(repl_idle_time_grouped_df, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name, plot_filename_base + "replicating", experiment_date_id, iter)

        if has_adaptive_metrics:
            lrb_adaptive_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                       "taskmanager_job_task_idleTimeMsPerSecond", file_date,
                                                       "lrb_adaptive", parallelism_level, scheduling_period,
                                                       src_parallelism,
                                                       iter, exp_host)
            adapt_idle_time_grouped_df = get_grouped_df(idle_time_col_list, lrb_adaptive_idle_time_file)
            plot_metric(adapt_idle_time_grouped_df, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name, plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            lrb_scheduling_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                         "taskmanager_job_task_idleTimeMsPerSecond", file_date,
                                                         "lrb_scheduling", parallelism_level, scheduling_period,
                                                         src_parallelism, iter, exp_host)
            sched_idle_time_grouped_df = get_grouped_df(idle_time_col_list, lrb_scheduling_idle_time_file)
            plot_metric(sched_idle_time_grouped_df, x_label, y_label, plot_title_base + "Scheduling",
                        group_by_col_name, plot_filename_base + "scheduling", experiment_date_id, iter)

    if plot_backpressure:
        backpressured_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        x_label = "Time (sec)"
        y_label = "ms/sec"
        plot_title_base = "BP Time (ms/sec) - "
        plot_filename_base = "backpressured_time_"
        group_by_col_name = "task_name"

        lrb_default_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                           "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                           file_date, default_id_str, parallelism_level,
                                                           default_sched_period, src_parallelism, iter, exp_host)
        backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list, lrb_default_backpressured_time_file)
        plot_metric(backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name, plot_filename_base + "default", experiment_date_id, iter, 500)

        if has_replicating_only_metrics:
            lrb_replicating_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                                   "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                                   file_date, "lrb_replicating",
                                                                   parallelism_level, scheduling_period,
                                                                   src_parallelism,
                                                                   iter, exp_host)
            repl_backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list,
                                                                lrb_replicating_backpressured_time_file)
            plot_metric(repl_backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name, plot_filename_base + "replicating", experiment_date_id, iter)

        if has_adaptive_metrics:
            lrb_adaptive_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                                "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                                file_date, "lrb_adaptive", parallelism_level,
                                                                scheduling_period, src_parallelism, iter, exp_host)
            adapt_backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list,
                                                                 lrb_adaptive_backpressured_time_file)
            plot_metric(adapt_backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name, plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            lrb_scheduling_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                                  "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                                  file_date, "lrb_scheduling",
                                                                  parallelism_level, scheduling_period, src_parallelism,
                                                                  iter, exp_host)
            sched_backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list,
                                                                 lrb_scheduling_backpressured_time_file)
            plot_metric(sched_backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Scheduling",
                        group_by_col_name, plot_filename_base + "scheduling", experiment_date_id, iter)

    if plot_iq_len:
        iq_len_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        x_label = "Time (sec)"
        y_label = "Num. buffers"
        plot_title_base = "Input Queue Length - "
        plot_filename_base = "iq_len_"
        group_by_col_name = "task_name"

        lrb_default_iq_len_file = get_filename(data_dir, experiment_date_id,
                                               "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                               file_date, default_id_str, parallelism_level,
                                               default_sched_period, src_parallelism, iter, exp_host)
        iq_len_grouped_df = get_grouped_df(iq_len_col_list, lrb_default_iq_len_file)
        plot_metric(iq_len_grouped_df, x_label, y_label, plot_title_base + "Default", group_by_col_name,
                    plot_filename_base + "default", experiment_date_id, iter)

        if has_adaptive_metrics:
            lrb_adaptive_iq_len_file = get_filename(data_dir, experiment_date_id,
                                                    "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                                    file_date, "lrb_adaptive", parallelism_level,
                                                    scheduling_period, src_parallelism, iter, exp_host)
            adapt_iq_len_grouped_df = get_grouped_df(iq_len_col_list, lrb_adaptive_iq_len_file)
            plot_metric(adapt_iq_len_grouped_df, x_label, y_label, plot_title_base + "Adaptive", group_by_col_name,
                        plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            lrb_scheduling_iq_len_file = get_filename(data_dir, experiment_date_id,
                                                      "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                                      file_date, "lrb_scheduling", parallelism_level,
                                                      scheduling_period, src_parallelism, iter, exp_host)
            sched_iq_len_grouped_df = get_grouped_df(iq_len_col_list, lrb_scheduling_iq_len_file)
            plot_metric(sched_iq_len_grouped_df, x_label, y_label, plot_title_base + "Scheduling", group_by_col_name,
                        plot_filename_base + "scheduling", experiment_date_id, iter)

    if plot_nw:
        nw_col_list = ["name", "host", "time", "value"]
        x_label = "Time (sec)"
        y_label = "Bytes/sec"
        plot_title_base = "Network Receive Rate - "
        plot_filename_base = "nw_"
        group_by_col_name = "host"
        nw_if = "enp4s0"

        lrb_default_nw_file = get_filename(data_dir, experiment_date_id,
                                           "taskmanager_System_Network_" + nw_if + "_ReceiveRate", file_date,
                                           default_id_str, parallelism_level, default_sched_period, src_parallelism,
                                           iter, exp_host)
        nw_df = get_df_without_groupby(nw_col_list, lrb_default_nw_file)
        combined_df = nw_df
        combined_df['sched_policy'] = "LRB-Default"
        plot_metric(nw_df, x_label, y_label, plot_title_base + "Default", group_by_col_name,
                    plot_filename_base + "default", experiment_date_id, iter)

        if has_adaptive_metrics:
            lrb_adaptive_nw_file = get_filename(data_dir, experiment_date_id,
                                                "taskmanager_System_Network_" + nw_if + "_ReceiveRate",
                                                file_date, "lrb_adaptive", parallelism_level,
                                                scheduling_period, src_parallelism, iter, exp_host)
            adapt_nw_df = get_df_without_groupby(nw_col_list, lrb_adaptive_nw_file)
            combined_df = combine_df_without_groupby(combined_df, nw_col_list, lrb_adaptive_nw_file, "LRB-Adaptive")
            plot_metric(adapt_nw_df, x_label, y_label, plot_title_base + "Adaptive", group_by_col_name,
                        plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            lrb_scheduling_nw_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_System_Network_" + nw_if + "_ReceiveRate",
                                                  file_date, "lrb_scheduling", parallelism_level,
                                                  scheduling_period, src_parallelism, iter, exp_host)
            sched_nw_df = get_df_without_groupby(nw_col_list, lrb_scheduling_nw_file)
            combined_df = combine_df_without_groupby(combined_df, nw_col_list, lrb_scheduling_nw_file, "LRB-Scheduling")
            plot_metric(sched_nw_df, x_label, y_label, plot_title_base + "Scheduling", group_by_col_name,
                        plot_filename_base + "scheduling", experiment_date_id, iter)

        combined_df = combined_df.loc[
            (combined_df.index > lower_time_threshold) & (combined_df.index < upper_time_threshold)]
        plot_metric(combined_df, x_label, y_label, "Network Receive Rate", "sched_policy", "nw_rcv", experiment_date_id,
                    iter)
