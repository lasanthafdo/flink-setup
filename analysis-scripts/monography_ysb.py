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


def get_formatted_tput(ysb_num_out_file, column_list, lower_threshold, upper_threshold, offset,
                       target_op_for_tp):
    print("Reading file " + ysb_num_out_file + " filtering events between " + str(lower_threshold) + " and " + str(
        upper_threshold) + " seconds")
    ysb_df = pd.read_csv(ysb_num_out_file, usecols=column_list)
    ysb_target_op_df = ysb_df[ysb_df['operator_name'].str.contains(target_op_for_tp)].drop(
        ['name'], axis=1).groupby(['time'])[['rate', 'count']].sum().reset_index()
    ysb_target_op_df['rel_time'] = ysb_target_op_df['time'].subtract(ysb_target_op_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    ysb_target_op_df = ysb_target_op_df.loc[
        (ysb_target_op_df['rel_time'] > lower_threshold) & (ysb_target_op_df['rel_time'] < upper_threshold)]
    ysb_avg = np.mean(ysb_target_op_df['rate'])
    return ysb_target_op_df, ysb_avg


def get_formatted_alt_tput(ysb_num_out_file, column_list, lower_threshold, upper_threshold, offset):
    print("Reading file : " + ysb_num_out_file)
    ysb_df = pd.read_csv(ysb_num_out_file, usecols=column_list)
    ysb_src_df = ysb_df[ysb_df['task_name'].str.contains('Source')].drop(
        ['name'], axis=1).groupby(['time'])['value'].sum().reset_index()
    ysb_src_df['rel_time'] = ysb_src_df['time'].subtract(ysb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    ysb_src_df = ysb_src_df.loc[
        (ysb_src_df['rel_time'] > lower_threshold) & (ysb_src_df['rel_time'] < upper_threshold)]
    ysb_avg = np.mean(ysb_src_df['value'])
    return ysb_src_df, ysb_avg


def get_formatted_latency(ysb_latency_file, column_list, lower_threshold, upper_threshold, offset, target_op_id,
                          target_stat):
    print("Reading file : " + ysb_latency_file)
    ysb_latency_df = pd.read_csv(ysb_latency_file, usecols=column_list)
    ysb_sink_latency_df = ysb_latency_df[ysb_latency_df['operator_id'] == target_op_id].drop(
        ['name'], axis=1).groupby(['time'])[['mean', 'p50', 'p95', 'p99']].mean().reset_index()
    ysb_sink_latency_df['rel_time'] = ysb_sink_latency_df['time'].subtract(ysb_sink_latency_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    ysb_sink_latency_df = ysb_sink_latency_df.loc[
        (ysb_sink_latency_df['rel_time'] > lower_threshold) & (ysb_sink_latency_df['rel_time'] < upper_threshold)]
    ysb_avg = np.mean(ysb_sink_latency_df[target_stat])
    return ysb_sink_latency_df, ysb_avg


def get_formatted_alt_latency(ysb_latency_file, column_list, lower_threshold, upper_threshold, offset, target_task_name,
                              target_stat):
    print("Reading file : " + ysb_latency_file)
    ysb_latency_df = pd.read_csv(ysb_latency_file, usecols=column_list)
    ysb_sink_latency_df = ysb_latency_df[ysb_latency_df['task_name'] == target_task_name].drop(
        ['name'], axis=1).groupby(['time'])[['mean', 'p50', 'p95', 'p99']].mean().reset_index()
    ysb_sink_latency_df['rel_time'] = ysb_sink_latency_df['time'].subtract(ysb_sink_latency_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    ysb_sink_latency_df = ysb_sink_latency_df.loc[
        (ysb_sink_latency_df['rel_time'] > lower_threshold) & (ysb_sink_latency_df['rel_time'] < upper_threshold)]
    ysb_avg = np.mean(ysb_sink_latency_df[target_stat])
    return ysb_sink_latency_df, ysb_avg


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


def get_op_name_id_mapping(ysb_tp_file):
    op_name_id_mapping_df = pd.read_csv(ysb_tp_file, usecols=['operator_name', 'operator_id']).drop_duplicates()
    op_name_id_dict = dict(zip(op_name_id_mapping_df['operator_name'], op_name_id_mapping_df['operator_id']))
    return op_name_id_dict


def get_pivoted_latency(ysb_latency_file, column_list, target_stat, op_to_id_dict, upper_threshold, lower_threshold):
    ysb_all_latency_for_sched_mode = pd.read_csv(ysb_latency_file, usecols=column_list)
    join_dict_df = pd.DataFrame(op_to_id_dict.items(), columns=['operator_name', 'operator_id'])
    ysb_all_latency_for_sched_mode = pd.merge(ysb_all_latency_for_sched_mode, join_dict_df, on="operator_id")
    ysb_all_latency_for_sched_mode = ysb_all_latency_for_sched_mode.groupby(['time', 'operator_name'])[
        [target_stat]].mean().reset_index()
    ysb_pivoted_latency_df = ysb_all_latency_for_sched_mode.pivot(index='time', columns='operator_name',
                                                                  values=target_stat)
    ysb_pivoted_latency_df.columns = [''.join(col).strip() for col in ysb_pivoted_latency_df.columns]
    # col_order = ['time', 'fil_1', 'tsw_1', 'prj_1', 'vehicle_win_1', 'speed_win_1', 'acc_win_1', 'toll_win_1',
    #              'toll_acc_win_1', 'Sink: sink_1']
    col_order = ['time', 'Filter', 'Map', 'JoinWithRedis (1)', "Window (1)", 'Sink: Sink(1)']
    ysb_pivoted_latency_df = ysb_pivoted_latency_df.reset_index()[col_order]
    ysb_pivoted_latency_df['rel_time'] = ysb_pivoted_latency_df['time'].subtract(
        ysb_pivoted_latency_df['time'].min()).div(1_000_000_000)
    ysb_pivoted_latency_df = ysb_pivoted_latency_df.loc[
        (ysb_pivoted_latency_df['rel_time'] > lower_threshold) & (ysb_pivoted_latency_df['rel_time'] < upper_threshold)]

    return ysb_pivoted_latency_df


def get_pivoted_alt_latency(ysb_latency_file, column_list, target_stat, upper_threshold, lower_threshold):
    ysb_all_latency_for_sched_mode = pd.read_csv(ysb_latency_file, usecols=column_list)
    ysb_all_latency_for_sched_mode = ysb_all_latency_for_sched_mode.groupby(['time', 'task_name'])[
        [target_stat]].mean().reset_index()
    ysb_pivoted_latency_df = ysb_all_latency_for_sched_mode.pivot(index='time', columns='task_name',
                                                                  values=target_stat)
    ysb_pivoted_latency_df.columns = [''.join(col).strip() for col in ysb_pivoted_latency_df.columns]
    col_order = ['time', 'Sink: sink_1']
    ysb_pivoted_latency_df = ysb_pivoted_latency_df.reset_index()[col_order]
    ysb_pivoted_latency_df['rel_time'] = ysb_pivoted_latency_df['time'].subtract(
        ysb_pivoted_latency_df['time'].min()).div(1_000_000_000)
    ysb_pivoted_latency_df = ysb_pivoted_latency_df.loc[
        (ysb_pivoted_latency_df['rel_time'] > lower_threshold) & (ysb_pivoted_latency_df['rel_time'] < upper_threshold)]

    return ysb_pivoted_latency_df


def calc_plot_graphs_for_metric(metric_name, ysb_scheduling_policies, ysb_offsets, ysb_labels, default_sched_period,
                                scheduling_period, num_iters, target_metric, user_ylabel,
                                simple_metric_name, user_xlabel="Elapsed Time (sec)", exp_host="tem104",
                                target_op='Filter'):
    ysb_file_names = {}
    ysb_metric_dfs = {}
    ysb_metric_avgs = {}
    metric_avgs_iter_dict = {}
    metric_avgs_policy_dict = {}
    ysb_current_policy_labels = []
    for scheduling_policy in ysb_scheduling_policies:
        current_policy_lbl = ysb_labels[scheduling_policy]
        ysb_current_policy_labels.append(current_policy_lbl)
        metric_avgs_policy_dict[current_policy_lbl] = list()
        fig, ax = plt.subplots(figsize=(8, 5))
        for iter_val in range(1, num_iters + 1):
            if iter_val in iter_to_skip: continue
            iter_lbl = "Iter " + str(iter_val)
            if not iter_lbl in metric_avgs_iter_dict:
                metric_avgs_iter_dict[iter_lbl] = list()
            iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
            iter_policy_id = iter + scheduling_policy
            if skip_default and scheduling_policy == "ysb_default":
                ysb_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                              file_date,
                                                              scheduling_policy,
                                                              parallelism_level,
                                                              default_sched_period if scheduling_policy == "ysb_default" else scheduling_period,
                                                              src_parallelism, iter, exp_host)
                ysb_metric_dfs[iter_policy_id], ysb_metric_avgs[iter_policy_id] = get_formatted_tput(
                    ysb_file_names[iter_policy_id], flink_col_list,
                    lower_time_threshold,
                    upper_time_threshold,
                    ysb_offsets[scheduling_policy] if ysb_offsets[scheduling_policy] >= 0 else ysb_offsets[
                        "ysb_default"], target_op)
                ysb_op_name_id_dicts[iter_policy_id] = get_op_name_id_mapping(ysb_file_names[iter_policy_id])
                ax._get_lines.get_next_color()
            else:
                ysb_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                              file_date,
                                                              scheduling_policy,
                                                              parallelism_level,
                                                              default_sched_period if scheduling_policy == "ysb_default" else scheduling_period,
                                                              src_parallelism, iter, exp_host)

            if not skip_default or scheduling_policy != "ysb_default":
                if use_alt_metrics:
                    ysb_metric_dfs[iter_policy_id], ysb_metric_avgs[iter_policy_id] = get_formatted_alt_tput(
                        ysb_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        ysb_offsets[scheduling_policy] if ysb_offsets[scheduling_policy] >= 0 else ysb_offsets[
                            "ysb_default"])
                    ax.plot(ysb_metric_dfs[iter_policy_id]["rel_time"], ysb_metric_dfs[iter_policy_id]["value"],
                            label=iter + current_policy_lbl)
                else:
                    ysb_metric_dfs[iter_policy_id], ysb_metric_avgs[iter_policy_id] = get_formatted_tput(
                        ysb_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        ysb_offsets[scheduling_policy] if ysb_offsets[scheduling_policy] >= 0 else ysb_offsets[
                            "ysb_default"], target_op)
                    # metric_avgs_iter_dict[get_iteration_id(iter, is_global_iter, ysb_labels[scheduling_policy])] = \
                    #     ysb_metric_avgs[iter_policy_id]
                    metric_avgs_policy_dict[current_policy_lbl].append(ysb_metric_avgs[iter_policy_id])
                    metric_avgs_iter_dict[iter_lbl].append(
                        scheduling_policy + ":" + str(ysb_metric_avgs[iter_policy_id]))
                    ysb_op_name_id_dicts[iter_policy_id] = get_op_name_id_mapping(ysb_file_names[iter_policy_id])
                    ax.plot(ysb_metric_dfs[iter_policy_id]["rel_time"], ysb_metric_dfs[iter_policy_id][target_metric],
                            label=iter + current_policy_lbl)

        ax.set(xlabel=user_xlabel, ylabel=user_ylabel,
               title=("Custom " if use_alt_metrics else "Flink ") + simple_metric_name)
        ax.tick_params(axis="x", rotation=0)
        ax.set_ylim(bottom=0, top=2500000)
        ax.legend()
        plt.savefig(results_dir + "/" + simple_metric_name.lower() + "_" + (
            "custom_" if use_alt_metrics else "flink_") + scheduling_policy + "_" + parallelism_level + "_" + experiment_date_id + ".png")
        plt.show()

    fig_bar, ax_bar = plt.subplots(figsize=(8, 5))
    print(metric_avgs_iter_dict)

    x = np.arange(len(ysb_scheduling_policies))
    width = 0.8 / num_iters  # the width of the bars
    multiplier = 0
    for itr, encoded_measurements in metric_avgs_iter_dict.items():
        offset = width * multiplier
        derived_measurements = []
        for idx, value in enumerate(encoded_measurements):
            policy = value.split(":")[0]
            assert ysb_scheduling_policies[idx] == policy
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
    ax_bar.set_xticks(x + width, ysb_current_policy_labels)
    ax_bar.legend(loc="lower right")
    # ax_bar.tick_params(axis="x", rotation=90)
    plt.tight_layout()
    plt.savefig(results_dir + "/" + simple_metric_name.lower() + "_bar_" + (
        "custom_" if use_alt_metrics else "flink_") + parallelism_level + "_" + experiment_date_id + ".png")
    plt.show()

    print("Metric avgs" + str(metric_avgs_iter_dict))
    return ysb_file_names, ysb_metric_dfs


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
    parser.add_argument("-def", "--defaultid", default="ysb_osdef")
    parser.add_argument("-pol", "--policies", default="ysb_osdef")
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
    lower_time_threshold = 100
    plot_tp = True
    plot_latency = True
    plot_event_time_latency = False
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
    ysb_scheduling_policies = args.policies.split(",")
    default_sched_period = scheduling_period
    ysb_offsets = {"ysb_default": 0, "ysb_def116": 0, "ysb_bpoff": -1, "ysb_pd": -1, "ysb_schedidling": -1,
                   "ysb_scheduling": -1, "ysb_bpscheduling": -1, "ysb_osdef": -1, "ysb_lqf": -1, "ysb_bposdef": -1,
                   "ysb_bplqf": -1, "ysb_bpmitigation": -1, "ysb_bplatency": -1, "ysb_dynbuffers": -1,
                   "ysb_osdefbpoff": -1, "ysb_adaptltncy": -1}
    ysb_labels = {"ysb_default": "YSB-Default", "ysb_def116": "LRB-Default (1.16.x)", "ysb_bpoff": "LRB-BP Off",
                  "ysb_pd": "LRB-PD", "ysb_scheduling": "LRB-Scheduling",
                  "ysb_schedidling": "LRB-Scheduling with blocking", "ysb_osdef": "LRB-OS default",
                  "ysb_bpscheduling": "LRB-Scheduling BP", "ysb_bposdef": "LRB-OS default BP",
                  "ysb_bplqf": "LRB-Largest Q First BP", "ysb_lqf": "LRB-Largest Q First",
                  "ysb_bpmitigation": "LRB-Backpressure Mitigation", "ysb_bplatency": "LRB-BP Latency Mitigation",
                  "ysb_dynbuffers": "LRB-Dynamic Buffers", "ysb_osdefbpoff": "LRB-OS default BP off",
                  "ysb_adaptltncy": "LRB-Adaptive Buffers"}
    ysb_op_name_id_dicts = {}
    iter_to_skip = []
    local_iter_default = "2"
    is_global_iter = True
    tgt_op_for_tp = "Source"
    tgt_op_for_sink = "Sink: Sink(1)"

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

        ysb_file_names, ysb_src_tp_dfs = calc_plot_graphs_for_metric(flink_metric_name, ysb_scheduling_policies,
                                                                     ysb_offsets, ysb_labels,
                                                                     default_sched_period,
                                                                     scheduling_period, num_iters, "rate",
                                                                     "Rate (events/sec)", "Throughput",
                                                                     exp_host=exp_host, target_op=tgt_op_for_tp)

        if not use_alt_metrics:
            count_fig, count_ax = plt.subplots(figsize=(12, 6))

            for scheduling_policy in ysb_scheduling_policies:
                for iter_val in range(1, num_iters + 1):
                    if iter_val in iter_to_skip: continue
                    iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(
                        iter_val) + "_"
                    iter_policy_id = iter + scheduling_policy
                    if not skip_default or scheduling_policy != "ysb_default":
                        count_ax.plot(ysb_src_tp_dfs[iter_policy_id]["rel_time"],
                                      ysb_src_tp_dfs[iter_policy_id]["count"],
                                      label=iter + ysb_labels[scheduling_policy])

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
                ysb_default_df = pd.read_csv(ysb_file_names[iter_policy_id], usecols=flink_col_list)
                src_task_indexes = ysb_default_df[ysb_default_df['operator_name'].str.contains('Source')][
                    'subtask_index'].unique()
                other_task_indexes = ysb_default_df[ysb_default_df['operator_name'].str.contains('tsw')][
                    'subtask_index'].unique()
                src_task_indexes.sort()
                other_task_indexes.sort()
                print("Source subtasks: " + str(src_task_indexes))
                print("Other subtasks: " + str(other_task_indexes))

    else:
        ysb_default_op_name_id_dict = None
        ysb_pseudo_default_op_name_id_dict = None
        ysb_fcfsp_op_name_id_dict = None
        ysb_lqf_op_name_id_dict = None
        ysb_lqsf_op_name_id_dict = None
        ysb_lsf_op_name_id_dict = None
        ysb_rr_op_name_id_dict = None
        ysb_dummy_op_name_id_dict = None
        ysb_replicating_op_name_id_dict = None
        ysb_scheduling_op_name_id_dict = None
        ysb_adaptive_op_name_id_dict = None

    if plot_latency:
        col_list = ["name", "time", "operator_id", "operator_subtask_index", "mean", "p50", "p95", "p99"]
        metric_name = "taskmanager_job_latency_source_id_operator_id_operator_subtask_index_latency"
        target_op_name = 'Sink: Sink(1)'
        target_stat = 'mean'
        all_latency_graph_y_top = 300
        ysb_latency_file_names = {}
        ysb_latency_dfs = {}
        ysb_latency_avgs = {}
        ysb_latency_pivoted_dfs = {}

        for scheduling_policy in ysb_scheduling_policies:
            fig_all_ops, ax_all_ops = plt.subplots(figsize=(8, 6))
            for iter_val in range(1, num_iters + 1):
                if iter_val in iter_to_skip: continue
                iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
                iter_policy_id = iter + scheduling_policy
                if not skip_default or scheduling_policy != "ysb_default":
                    ysb_latency_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                                          file_date,
                                                                          scheduling_policy, parallelism_level,
                                                                          default_sched_period if scheduling_policy == "ysb_default" else scheduling_period,
                                                                          src_parallelism, iter, exp_host)
                    print(ysb_op_name_id_dicts[iter + default_id_str])
                    target_op_id = ysb_op_name_id_dicts[iter + default_id_str][target_op_name]
                    ysb_latency_dfs[iter_policy_id], ysb_latency_avgs[
                        get_iteration_id(iter, is_global_iter, scheduling_policy)] = get_formatted_latency(
                        ysb_latency_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        ysb_offsets["ysb_default"],
                        target_op_id, target_stat)
                    ysb_latency_pivoted_dfs[iter_policy_id] = get_pivoted_latency(
                        ysb_latency_file_names[iter_policy_id],
                        col_list,
                        target_stat,
                        ysb_op_name_id_dicts[iter_policy_id],
                        upper_time_threshold,
                        lower_time_threshold)

                    print(ysb_latency_pivoted_dfs[iter_policy_id])
                    ax_all_ops.plot(ysb_latency_pivoted_dfs[iter_policy_id]["rel_time"],
                                    ysb_latency_pivoted_dfs[iter_policy_id][tgt_op_for_sink],
                                    label="Sink - Iter: " + get_iteration_id(iter, is_global_iter, scheduling_policy))

            ax_all_ops.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                           title=ysb_labels[
                                     scheduling_policy] + " Processing Latency (" + target_stat + ") - All Operators ")
            ax_all_ops.legend()
            ax_all_ops.set_ylim(bottom=0)
            plt.savefig(
                results_dir + "/latency_flink_" + scheduling_policy + "_" + parallelism_level + "_all_" + target_stat + "_" + experiment_date_id + ".png")
            plt.show()

        print("Latency avgs: " + str(ysb_latency_avgs))
        fig_lat, ax = plt.subplots(figsize=(8, 5))
        for iter, lat_avg in ysb_latency_avgs.items():
            ax.bar(iter, lat_avg)

        ax.set(xlabel="Iteration", ylabel="Time (ms)",
               title="Processing Latency (" + target_stat + ") - Operator: " + target_op_name)
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
        target_op_name = tgt_op_for_sink
        target_stat = 'mean'
        all_latency_graph_y_top = 300
        et_latency_file_names = {}
        et_latency_dfs = {}
        et_latency_avgs_iter_dict = {}
        et_latency_avgs_policy_dict = {}
        et_latency_pivoted_dfs = {}
        ysb_current_policy_labels = []

        for scheduling_policy in ysb_scheduling_policies:
            current_policy_lbl = ysb_labels[scheduling_policy]
            ysb_current_policy_labels.append(current_policy_lbl)
            et_latency_avgs_policy_dict[current_policy_lbl] = list()
            fig_all_ops, ax_all_ops = plt.subplots(figsize=(8, 6))
            for iter_val in range(1, num_iters + 1):
                if iter_val in iter_to_skip: continue
                iter_lbl = "Iter " + str(iter_val)
                if not iter_lbl in et_latency_avgs_iter_dict:
                    et_latency_avgs_iter_dict[iter_lbl] = list()
                iter = str(iter_val) + "_" + local_iter_default + "_" if is_global_iter else "0_" + str(iter_val) + "_"
                iter_policy_id = iter + scheduling_policy
                if not skip_default or scheduling_policy != "ysb_default":
                    et_latency_file_names[iter_policy_id] = get_filename(data_dir, experiment_date_id, metric_name,
                                                                         file_date,
                                                                         scheduling_policy, parallelism_level,
                                                                         default_sched_period if scheduling_policy == "ysb_default" else scheduling_period,
                                                                         src_parallelism, iter, exp_host)
                    et_latency_dfs[iter_policy_id], et_lat_avg = get_formatted_alt_latency(
                        et_latency_file_names[iter_policy_id], col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        ysb_offsets["ysb_default"],
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
                           title=ysb_labels[
                                     scheduling_policy] + " Event Time Latency (" + target_stat + ") - All Operators ")
            ax_all_ops.legend()
            ax_all_ops.set_ylim(bottom=0)
            plt.savefig(
                results_dir + "/latency_gen_to_sink_" + scheduling_policy + "_" + parallelism_level + "_all_" + target_stat + "_" + experiment_date_id + ".png")
            plt.show()

        print("Latency avgs: " + str(et_latency_avgs_iter_dict))
        fig_lat, ax_lat = plt.subplots(figsize=(8, 5))

        x = np.arange(len(ysb_scheduling_policies))
        width = 0.8 / num_iters  # the width of the bars
        multiplier = 0
        for itr, encoded_measurements in et_latency_avgs_iter_dict.items():
            offset = width * multiplier
            derived_measurements = []
            for idx, value in enumerate(encoded_measurements):
                policy = value.split(":")[0]
                assert ysb_scheduling_policies[idx] == policy
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
                   title="Event Time Latency (" + target_stat + ") - Operator: " + target_op_name)
        ax_lat.set_xticks(x + width, ysb_current_policy_labels)
        ax_lat.set_ylim(bottom=0)
        ax_lat.legend()
        plt.tight_layout()
        plt.savefig(
            results_dir + "/latency_bar_gen_to_sink_" + parallelism_level + "_" + target_op_name + "_" + target_stat + "_" + experiment_date_id + ".png")
        plt.show()

    if plot_cpu:
        iter = "0_1_" if not is_global_iter else "1_2_"
        ysb_default_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                  file_date, default_id_str, parallelism_level,
                                                  default_sched_period, src_parallelism, iter, exp_host)
        if has_replicating_only_metrics:
            ysb_replicating_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                          file_date, "ysb_replicating", parallelism_level,
                                                          scheduling_period, src_parallelism, iter, exp_host)
        else:
            ysb_replicating_cpu_usage_file = None

        if has_adaptive_metrics:
            ysb_adaptive_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                       file_date, "ysb_adaptive", parallelism_level,
                                                       scheduling_period, src_parallelism, iter, exp_host)
        else:
            ysb_adaptive_cpu_usage_file = None

        if has_scheduling_only_metrics:
            ysb_scheduling_cpu_usage_file = get_filename(data_dir, experiment_date_id, "taskmanager_System_CPU_Usage",
                                                         file_date, "ysb_scheduling", parallelism_level,
                                                         scheduling_period, src_parallelism, iter, exp_host)
        else:
            ysb_scheduling_cpu_usage_file = None

        cpu_usage_col_list = ["name", "time", "value"]
        cpu_usage_df = pd.read_csv(ysb_default_cpu_usage_file, usecols=cpu_usage_col_list)
        cpu_usage_df['rel_time'] = cpu_usage_df['time'].subtract(cpu_usage_df['time'].min()).div(
            1_000_000_000).subtract(ysb_offsets["ysb_default"])
        cpu_usage_df = cpu_usage_df.loc[cpu_usage_df['rel_time'] > 0]
        cpu_usage_df.describe()

        if has_replicating_only_metrics:
            repl_cpu_usage_df = pd.read_csv(ysb_replicating_cpu_usage_file, usecols=cpu_usage_col_list)
            repl_cpu_usage_df['rel_time'] = repl_cpu_usage_df['time'].subtract(repl_cpu_usage_df['time'].min()).div(
                1_000_000_000)
            # print(repl_cpu_usage_df)
        else:
            repl_cpu_usage_df = None

        if has_adaptive_metrics:
            adapt_cpu_usage_df = pd.read_csv(ysb_adaptive_cpu_usage_file, usecols=cpu_usage_col_list)
            adapt_cpu_usage_df['rel_time'] = adapt_cpu_usage_df['time'].subtract(adapt_cpu_usage_df['time'].min()).div(
                1_000_000_000)
            # print(adapt_cpu_usage_df)
        else:
            adapt_cpu_usage_df = None

        if has_scheduling_only_metrics:
            sched_cpu_usage_df = pd.read_csv(ysb_scheduling_cpu_usage_file, usecols=cpu_usage_col_list)
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
        ysb_default_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                  default_id_str, parallelism_level, default_sched_period,
                                                  src_parallelism, iter, exp_host)
        if has_replicating_only_metrics:
            ysb_replicating_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                          "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                          "ysb_replicating", parallelism_level, scheduling_period,
                                                          src_parallelism, iter, exp_host)
        else:
            ysb_replicating_mem_usage_file = None

        if has_adaptive_metrics:
            ysb_adaptive_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                       "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                       "ysb_adaptive", parallelism_level, scheduling_period,
                                                       src_parallelism, iter, exp_host)
        else:
            ysb_adaptive_mem_usage_file = None

        if has_scheduling_only_metrics:
            ysb_scheduling_mem_usage_file = get_filename(data_dir, experiment_date_id,
                                                         "taskmanager_Status_JVM_Memory_Heap_Used", file_date,
                                                         "ysb_scheduling", parallelism_level, scheduling_period,
                                                         src_parallelism, iter, exp_host)
        else:
            ysb_scheduling_mem_usage_file = None

        mem_usage_col_list = ["name", "time", "value"]
        mem_usage_df = pd.read_csv(ysb_default_mem_usage_file, usecols=mem_usage_col_list)
        mem_usage_df['rel_time'] = mem_usage_df['time'].subtract(mem_usage_df['time'].min()).div(
            1_000_000_000).subtract(ysb_offsets["ysb_default"])
        mem_usage_df = mem_usage_df.loc[mem_usage_df['rel_time'] > 0]
        mem_usage_df['value'] = mem_usage_df['value'].div(1048576)
        # print(mem_usage_df)

        if has_replicating_only_metrics:
            repl_mem_usage_df = pd.read_csv(ysb_replicating_mem_usage_file, usecols=mem_usage_col_list)
            repl_mem_usage_df['rel_time'] = repl_mem_usage_df['time'].subtract(repl_mem_usage_df['time'].min()).div(
                1_000_000_000)
            repl_mem_usage_df['value'] = repl_mem_usage_df['value'].div(1048576)
            # print(repl_mem_usage_df)
        else:
            repl_mem_usage_df = None

        if has_adaptive_metrics:
            adapt_mem_usage_df = pd.read_csv(ysb_adaptive_mem_usage_file, usecols=mem_usage_col_list)
            adapt_mem_usage_df['rel_time'] = adapt_mem_usage_df['time'].subtract(adapt_mem_usage_df['time'].min()).div(
                1_000_000_000)
            adapt_mem_usage_df['value'] = adapt_mem_usage_df['value'].div(1048576)
            # print(adapt_mem_usage_df)
        else:
            adapt_mem_usage_df = None

        if has_scheduling_only_metrics:
            sched_mem_usage_df = pd.read_csv(ysb_scheduling_mem_usage_file, usecols=mem_usage_col_list)
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

        ysb_default_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_job_task_busyTimeMsPerSecond", file_date,
                                                  default_id_str, parallelism_level, default_sched_period,
                                                  src_parallelism, iter, exp_host)
        busy_time_df = pd.read_csv(ysb_default_busy_time_file, usecols=busy_time_col_list)
        busy_time_grouped_df = busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
        busy_time_grouped_df['rel_time'] = busy_time_grouped_df['time'].subtract(
            busy_time_grouped_df['time'].min()).div(1_000_000_000)
        plot_metric(busy_time_grouped_df, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name, plot_filename_base + "default", experiment_date_id, iter)

        if has_replicating_only_metrics:
            ysb_replicating_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                          "taskmanager_job_task_busyTimeMsPerSecond",
                                                          file_date, "ysb_replicating", parallelism_level,
                                                          scheduling_period, src_parallelism, iter, exp_host)
            repl_busy_time_df = pd.read_csv(ysb_replicating_busy_time_file, usecols=busy_time_col_list)
            repl_busy_time_grouped_df = repl_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
            repl_busy_time_grouped_df['rel_time'] = repl_busy_time_grouped_df['time'].subtract(
                repl_busy_time_grouped_df['time'].min()).div(1_000_000_000)
            plot_metric(repl_busy_time_grouped_df, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name, plot_filename_base + "replicating", experiment_date_id, iter)

        if has_adaptive_metrics:
            ysb_adaptive_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                       "taskmanager_job_task_busyTimeMsPerSecond", file_date,
                                                       "ysb_adaptive", parallelism_level, scheduling_period,
                                                       src_parallelism,
                                                       iter, exp_host)
            adapt_busy_time_df = pd.read_csv(ysb_adaptive_busy_time_file, usecols=busy_time_col_list)
            adapt_busy_time_grouped_df = adapt_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
            adapt_busy_time_grouped_df['rel_time'] = adapt_busy_time_grouped_df['time'].subtract(
                adapt_busy_time_grouped_df['time'].min()).div(1_000_000_000)
            plot_metric(adapt_busy_time_grouped_df, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name, plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            ysb_scheduling_busy_time_file = get_filename(data_dir, experiment_date_id,
                                                         "taskmanager_job_task_busyTimeMsPerSecond", file_date,
                                                         "ysb_scheduling", parallelism_level, scheduling_period,
                                                         src_parallelism, iter, exp_host)
            sched_busy_time_df = pd.read_csv(ysb_scheduling_busy_time_file, usecols=busy_time_col_list)
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

        ysb_default_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_job_task_idleTimeMsPerSecond", file_date,
                                                  default_id_str, parallelism_level, default_sched_period,
                                                  src_parallelism,
                                                  iter, exp_host)
        idle_time_grouped_df = get_grouped_df(idle_time_col_list, ysb_default_idle_time_file)
        plot_metric(idle_time_grouped_df, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name, plot_filename_base + "default", experiment_date_id, iter)

        if has_replicating_only_metrics:
            ysb_replicating_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                          "taskmanager_job_task_idleTimeMsPerSecond",
                                                          file_date, "ysb_replicating", parallelism_level,
                                                          scheduling_period, src_parallelism, iter, exp_host)
            repl_idle_time_grouped_df = get_grouped_df(idle_time_col_list, ysb_replicating_idle_time_file)
            plot_metric(repl_idle_time_grouped_df, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name, plot_filename_base + "replicating", experiment_date_id, iter)

        if has_adaptive_metrics:
            ysb_adaptive_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                       "taskmanager_job_task_idleTimeMsPerSecond", file_date,
                                                       "ysb_adaptive", parallelism_level, scheduling_period,
                                                       src_parallelism,
                                                       iter, exp_host)
            adapt_idle_time_grouped_df = get_grouped_df(idle_time_col_list, ysb_adaptive_idle_time_file)
            plot_metric(adapt_idle_time_grouped_df, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name, plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            ysb_scheduling_idle_time_file = get_filename(data_dir, experiment_date_id,
                                                         "taskmanager_job_task_idleTimeMsPerSecond", file_date,
                                                         "ysb_scheduling", parallelism_level, scheduling_period,
                                                         src_parallelism, iter, exp_host)
            sched_idle_time_grouped_df = get_grouped_df(idle_time_col_list, ysb_scheduling_idle_time_file)
            plot_metric(sched_idle_time_grouped_df, x_label, y_label, plot_title_base + "Scheduling",
                        group_by_col_name, plot_filename_base + "scheduling", experiment_date_id, iter)

    if plot_backpressure:
        backpressured_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        x_label = "Time (sec)"
        y_label = "ms/sec"
        plot_title_base = "BP Time (ms/sec) - "
        plot_filename_base = "backpressured_time_"
        group_by_col_name = "task_name"

        ysb_default_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                           "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                           file_date, default_id_str, parallelism_level,
                                                           default_sched_period, src_parallelism, iter, exp_host)
        backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list, ysb_default_backpressured_time_file)
        plot_metric(backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Default",
                    group_by_col_name, plot_filename_base + "default", experiment_date_id, iter, 500)

        if has_replicating_only_metrics:
            ysb_replicating_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                                   "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                                   file_date, "ysb_replicating",
                                                                   parallelism_level, scheduling_period,
                                                                   src_parallelism,
                                                                   iter, exp_host)
            repl_backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list,
                                                                ysb_replicating_backpressured_time_file)
            plot_metric(repl_backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Replicating",
                        group_by_col_name, plot_filename_base + "replicating", experiment_date_id, iter)

        if has_adaptive_metrics:
            ysb_adaptive_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                                "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                                file_date, "ysb_adaptive", parallelism_level,
                                                                scheduling_period, src_parallelism, iter, exp_host)
            adapt_backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list,
                                                                 ysb_adaptive_backpressured_time_file)
            plot_metric(adapt_backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Adaptive",
                        group_by_col_name, plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            ysb_scheduling_backpressured_time_file = get_filename(data_dir, experiment_date_id,
                                                                  "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                                  file_date, "ysb_scheduling",
                                                                  parallelism_level, scheduling_period, src_parallelism,
                                                                  iter, exp_host)
            sched_backpressured_time_grouped_df = get_grouped_df(backpressured_time_col_list,
                                                                 ysb_scheduling_backpressured_time_file)
            plot_metric(sched_backpressured_time_grouped_df, x_label, y_label, plot_title_base + "Scheduling",
                        group_by_col_name, plot_filename_base + "scheduling", experiment_date_id, iter)

    if plot_iq_len:
        iq_len_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        x_label = "Time (sec)"
        y_label = "Num. buffers"
        plot_title_base = "Input Queue Length - "
        plot_filename_base = "iq_len_"
        group_by_col_name = "task_name"

        ysb_default_iq_len_file = get_filename(data_dir, experiment_date_id,
                                               "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                               file_date, default_id_str, parallelism_level,
                                               default_sched_period, src_parallelism, iter, exp_host)
        iq_len_grouped_df = get_grouped_df(iq_len_col_list, ysb_default_iq_len_file)
        plot_metric(iq_len_grouped_df, x_label, y_label, plot_title_base + "Default", group_by_col_name,
                    plot_filename_base + "default", experiment_date_id, iter)

        if has_adaptive_metrics:
            ysb_adaptive_iq_len_file = get_filename(data_dir, experiment_date_id,
                                                    "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                                    file_date, "ysb_adaptive", parallelism_level,
                                                    scheduling_period, src_parallelism, iter, exp_host)
            adapt_iq_len_grouped_df = get_grouped_df(iq_len_col_list, ysb_adaptive_iq_len_file)
            plot_metric(adapt_iq_len_grouped_df, x_label, y_label, plot_title_base + "Adaptive", group_by_col_name,
                        plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            ysb_scheduling_iq_len_file = get_filename(data_dir, experiment_date_id,
                                                      "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                                      file_date, "ysb_scheduling", parallelism_level,
                                                      scheduling_period, src_parallelism, iter, exp_host)
            sched_iq_len_grouped_df = get_grouped_df(iq_len_col_list, ysb_scheduling_iq_len_file)
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

        ysb_default_nw_file = get_filename(data_dir, experiment_date_id,
                                           "taskmanager_System_Network_" + nw_if + "_ReceiveRate", file_date,
                                           default_id_str, parallelism_level, default_sched_period, src_parallelism,
                                           iter, exp_host)
        nw_df = get_df_without_groupby(nw_col_list, ysb_default_nw_file)
        combined_df = nw_df
        combined_df['sched_policy'] = "LRB-Default"
        plot_metric(nw_df, x_label, y_label, plot_title_base + "Default", group_by_col_name,
                    plot_filename_base + "default", experiment_date_id, iter)

        if has_adaptive_metrics:
            ysb_adaptive_nw_file = get_filename(data_dir, experiment_date_id,
                                                "taskmanager_System_Network_" + nw_if + "_ReceiveRate",
                                                file_date, "ysb_adaptive", parallelism_level,
                                                scheduling_period, src_parallelism, iter, exp_host)
            adapt_nw_df = get_df_without_groupby(nw_col_list, ysb_adaptive_nw_file)
            combined_df = combine_df_without_groupby(combined_df, nw_col_list, ysb_adaptive_nw_file, "LRB-Adaptive")
            plot_metric(adapt_nw_df, x_label, y_label, plot_title_base + "Adaptive", group_by_col_name,
                        plot_filename_base + "adaptive", experiment_date_id, iter)

        if has_scheduling_only_metrics:
            ysb_scheduling_nw_file = get_filename(data_dir, experiment_date_id,
                                                  "taskmanager_System_Network_" + nw_if + "_ReceiveRate",
                                                  file_date, "ysb_scheduling", parallelism_level,
                                                  scheduling_period, src_parallelism, iter, exp_host)
            sched_nw_df = get_df_without_groupby(nw_col_list, ysb_scheduling_nw_file)
            combined_df = combine_df_without_groupby(combined_df, nw_col_list, ysb_scheduling_nw_file, "LRB-Scheduling")
            plot_metric(sched_nw_df, x_label, y_label, plot_title_base + "Scheduling", group_by_col_name,
                        plot_filename_base + "scheduling", experiment_date_id, iter)

        combined_df = combined_df.loc[
            (combined_df.index > lower_time_threshold) & (combined_df.index < upper_time_threshold)]
        plot_metric(combined_df, x_label, y_label, "Network Receive Rate", "sched_policy", "nw_rcv", experiment_date_id,
                    iter)
