# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRIC_TYPE_NW = "nw"

METRIC_TYPE_IQ_LEN = "iq_len"

METRIC_TYPE_BP_TIME = "bp"

METRIC_TYPE_IDLE_TIME = "idle"

METRIC_TYPE_BUSY_TIME = "busy"

METRIC_TYPE_MEM = "mem"
METRIC_TYPE_CPU = "cpu"
METRIC_TYPE_LATENCY = "latency"
METRIC_TYPE_TP = "tp"

LABEL_GAP_Y = 100000
LRB_DEFAULT = "lrb_default"

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)


def get_formatted_metrics(target_attribs, lrb_metric_file, column_list, lower_threshold, upper_threshold, offset,
                          target_operator, div_all_by_val, group_by_cols):
    print("Reading file : " + lrb_metric_file)
    lrb_df = pd.read_csv(lrb_metric_file, usecols=column_list)
    if target_operator:
        lrb_src_df = lrb_df[lrb_df['operator_name'].str.contains('Source:')].drop(
            ['name'], axis=1).groupby(['time'])[target_attribs].sum().reset_index()
        lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
            1_000_000_000).subtract(offset)
        lrb_src_df = lrb_src_df.loc[
            (lrb_src_df['rel_time'] > lower_threshold) & (lrb_src_df['rel_time'] < upper_threshold)]
        lrb_avg = np.mean(lrb_src_df[target_attribs[0]])
        return lrb_src_df, lrb_avg
    else:
        if group_by_cols:
            lrb_df = lrb_df.groupby(group_by_cols)[target_attribs[0]].mean().reset_index()
        lrb_df['rel_time'] = lrb_df['time'].subtract(lrb_df['time'].min()).div(
            1_000_000_000).subtract(offset)
        lrb_df = lrb_df.loc[
            (lrb_df['rel_time'] > lower_threshold) & (lrb_df['rel_time'] < upper_threshold)]
        if div_all_by_val:
            lrb_df[target_attribs[0]] = lrb_df[target_attribs[0]].div(div_all_by_val)
        lrb_avg = np.mean(lrb_df[target_attribs[0]])
        return lrb_df, lrb_avg


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
    print(lrb_pivoted_latency_df.columns)
    # col_order = ['time', 'fil_1 -> tsw_1 -> prj_1', 'vehicle_win_1 -> Map', 'speed_win_1 -> Map', 'acc_win_1 -> Map',
    #              'toll_win_1 -> Map', 'toll_acc_win_1', 'Sink: sink_1']
    col_order = ['time', 'fil_1 -> tsw_1 -> prj_1', 'Sink: sink_1']
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.reset_index()[col_order]
    lrb_pivoted_latency_df['rel_time'] = lrb_pivoted_latency_df['time'].subtract(
        lrb_pivoted_latency_df['time'].min()).div(1_000_000_000)
    lrb_pivoted_latency_df = lrb_pivoted_latency_df.loc[
        (lrb_pivoted_latency_df['rel_time'] > lower_threshold) & (lrb_pivoted_latency_df['rel_time'] < upper_threshold)]

    return lrb_pivoted_latency_df


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
    # print(combined_df)
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


def plot_graphs_for(tgt_metric_name, col_list, tgt_metric_lbls, tgt_attribs, tgt_sched_policies,
                    input_file_names, metric_dfs, metric_avgs, tgt_op=None, div_all_by=None, group_by_cols=None,
                    filter_cols=None):
    num_plots = 1
    if filter_cols:
        num_plots = len(filter_cols)

    axes = None
    ax_all = None
    if num_plots > 1:
        fig_all, axes = plt.subplots(nrows=num_plots, figsize=(8, 5 * num_plots))
    else:
        fig_all, ax_all = plt.subplots(figsize=(8, 5))

    for policy in tgt_sched_policies:
        if skip_default and policy == LRB_DEFAULT:
            input_file_names[policy] = get_filename(data_dir, experiment_date_id, tgt_metric_name,
                                                    file_date,
                                                    policy,
                                                    parallelism_level,
                                                    default_sched_period if policy == LRB_DEFAULT else scheduling_period,
                                                    num_parts, iter, exp_host)
            metric_dfs[policy], metric_avgs[policy] = get_formatted_metrics(tgt_attribs, input_file_names[policy],
                                                                            col_list, lower_time_threshold,
                                                                            upper_time_threshold,
                                                                            lrb_offsets[policy] if lrb_offsets[
                                                                                                       policy] >= 0 else
                                                                            lrb_offsets[
                                                                                LRB_DEFAULT], tgt_op, div_all_by,
                                                                            group_by_cols)
            if tgt_op:
                lrb_op_name_id_dicts[policy] = get_op_name_id_mapping(input_file_names[policy])
            ax_all._get_lines.get_next_color()
        else:
            input_file_names[policy] = get_filename(data_dir, experiment_date_id, tgt_metric_name,
                                                    file_date,
                                                    policy,
                                                    parallelism_level,
                                                    default_sched_period if policy == LRB_DEFAULT else scheduling_period,
                                                    num_parts, iter, exp_host)

        if not skip_default or policy != LRB_DEFAULT:
            if use_alt_metrics:
                metric_dfs[policy], metric_avgs[policy] = get_formatted_alt_tput(
                    input_file_names[policy], col_list,
                    lower_time_threshold,
                    upper_time_threshold,
                    lrb_offsets[policy] if lrb_offsets[policy] >= 0 else lrb_offsets[
                        LRB_DEFAULT])
                ax_all.plot(metric_dfs[policy]["rel_time"], metric_dfs[policy]["value"],
                            label=lrb_labels[policy])
            else:
                metric_dfs[policy], metric_avgs[policy] = get_formatted_metrics(tgt_attribs, input_file_names[policy],
                                                                                col_list, lower_time_threshold,
                                                                                upper_time_threshold,
                                                                                lrb_offsets[policy] if lrb_offsets[
                                                                                                           policy] >= 0 else
                                                                                lrb_offsets[LRB_DEFAULT], tgt_op,
                                                                                div_all_by, group_by_cols)
                if tgt_op:
                    lrb_op_name_id_dicts[policy] = get_op_name_id_mapping(input_file_names[policy])
                if group_by_cols:
                    if num_plots > 1:
                        current_ax = 0
                        for key, grp in metric_dfs[policy].groupby(group_by_cols[1]):
                            if key in filter_cols:
                                axes[current_ax].plot(grp['rel_time'], grp[tgt_attribs[0]],
                                                      label="{} - {}".format(lrb_labels[policy], key))
                                axes[current_ax].set_ylabel("{} {}".format(tgt_metric_lbls[0], tgt_metric_lbls[2]))

                                axes[current_ax].axhline(y=np.mean(grp[tgt_attribs[0]]), ls='--',
                                                         label="{} - {} Avg.".format(lrb_labels[policy], key))
                                label_gap_y_axis = metric_dfs[list(metric_dfs.keys())[0]][tgt_attribs[0]].max() / 10
                                text_offset = (list(metric_avgs).index(policy) + 1) * label_gap_y_axis
                                axes[current_ax].text(100, np.mean(grp[tgt_attribs[0]]) - text_offset,
                                                      "{} Avg. {} ({}) = {}".format(lrb_labels[policy],
                                                                                    tgt_metric_lbls[1], key,
                                                                                    f'{np.mean(grp[tgt_attribs[0]]):,.2f}'))
                                axes[current_ax].legend()
                                current_ax += 1
                    else:
                        for key, grp in metric_dfs[policy].groupby(group_by_cols[1]):
                            ax_all.plot(grp['rel_time'], grp[tgt_attribs[0]],
                                        label="{} - {}".format(lrb_labels[policy], key))
                else:
                    ax_all.plot(metric_dfs[policy]["rel_time"],
                                metric_dfs[policy][tgt_attribs[0]],
                                label=lrb_labels[policy])

            if not num_plots > 1:
                plt.axhline(y=metric_avgs[policy], ls='--', label=lrb_labels[policy] + "-Avg")
                label_gap_y_axis = metric_dfs[list(metric_dfs.keys())[0]][tgt_attribs[0]].max() / 10
                text_offset = (list(metric_avgs).index(policy) + 1) * label_gap_y_axis
                plt.text(100, metric_avgs[policy] - text_offset,
                         "{} Avg. {} = {}".format(lrb_labels[policy], tgt_metric_lbls[1],
                                                  f'{metric_avgs[policy]:,.2f}'))

    if num_plots > 1:
        plt.xlabel("Time (sec)")
        # plt.ylabel("{} {}".format(tgt_metric_lbls[0], tgt_metric_lbls[2]))
        plt.title("{} {}".format("Custom " if use_alt_metrics else "Flink ", tgt_metric_lbls[0]))
        plt.tick_params(axis="x", rotation=0)
    else:
        ax_all.set(xlabel="Time (sec)", ylabel="{} {}".format(tgt_metric_lbls[0], tgt_metric_lbls[2]),
                   title="{} {}".format("Custom " if use_alt_metrics else "Flink ", tgt_metric_lbls[0]))
        ax_all.tick_params(axis="x", rotation=0)
        ax_all.legend()

    plt.savefig("{}/{}_{}_{}_{}_iter{}.png".format(results_dir, tgt_metric_lbls[1].lower(),
                                                   "custom" if use_alt_metrics else "flink", parallelism_level,
                                                   experiment_date_id, iter))
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "/home/m34ferna/flink-tests/data"
    experiment_date_id = "oct-28-1"
    file_date = "2023_10_28"
    parallelism_level = "1"
    num_parts = "1"
    results_dir = "results/" + experiment_date_id + "/par_" + parallelism_level
    os.makedirs(results_dir, exist_ok=True)
    scheduling_period = "5"
    exp_host = "tem110"

    upper_time_threshold = 1200
    lower_time_threshold = 0
    plot_tp = True
    plot_latency = True
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

    default_id_str = "lrb_osdef"
    default_sched_period = "5"
    lrb_scheduling_policies = ["lrb_osdef", "lrb_bpmitigation"]
    lrb_offsets = {"lrb_default": 0, "lrb_pd": -1, "lrb_scheduling": -1, "lrb_osdef": -1, "lrb_bpmitigation": -1}
    lrb_labels = {"lrb_default": "LRB-Default", "lrb_pd": "LRB-PD", "lrb_scheduling": "LRB-Scheduling",
                  "lrb_osdef": "LRB-OS default", "lrb_bpmitigation": "LRB-BP Mitigation"}
    lrb_op_name_id_dicts = {}
    col_lists = {}
    metric_names = {}
    iter = "2_2_"

    if plot_tp:
        flink_col_list = ["name", "time", "operator_name", "operator_id", "task_name", "subtask_index", "count", "rate"]
        alt_col_list = ["name", "time", "task_name", "subtask_index", "value"]
        flink_metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
        alt_metric_name = "taskmanager_job_task_numRecordsInChannelPerSecond"
        if use_alt_metrics:
            metric_names[METRIC_TYPE_TP] = alt_metric_name
            col_lists[METRIC_TYPE_TP] = alt_col_list
        else:
            metric_names[METRIC_TYPE_TP] = flink_metric_name
            col_lists[METRIC_TYPE_TP] = flink_col_list
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["rate", "count"]
        metric_type_labels = ["Throughput", "TP", "events/sec"]
        plot_graphs_for(flink_metric_name, col_lists[METRIC_TYPE_TP], metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, "Source")

        if not use_alt_metrics:
            count_fig, count_ax = plt.subplots(figsize=(12, 6))

            for scheduling_policy in lrb_scheduling_policies:
                if not skip_default or scheduling_policy != "lrb_default":
                    count_ax.plot(lrb_metric_dfs[scheduling_policy]["rel_time"],
                                  lrb_metric_dfs[scheduling_policy]["count"],
                                  label=lrb_labels[scheduling_policy])

            # count_ax.set_ylim(bottom=0)
            count_ax.set(xlabel="Time (sec)", ylabel="Total events", title="Event count")
            count_ax.tick_params(axis="x", rotation=0)
            count_ax.legend()
            plt.savefig(
                results_dir + "/count_" + parallelism_level + "_" + experiment_date_id + "_iter" + iter + ".png")
            plt.show()

        if not use_alt_metrics:
            lrb_default_df = pd.read_csv(lrb_file_names[default_id_str], usecols=flink_col_list)
            print(lrb_metric_dfs[default_id_str].describe().apply(lambda s: s.apply('{0:.1f}'.format)))
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
        col_lists[METRIC_TYPE_LATENCY] = ["name", "time", "operator_id", "operator_subtask_index", "mean", "p50", "p95",
                                          "p99"]
        metric_names[
            METRIC_TYPE_LATENCY] = "taskmanager_job_latency_source_id_operator_id_operator_subtask_index_latency"
        alt_col_list = ["name", "time", "subtask_index", "task_name", "mean", "p50", "p95", "p99"]
        alt_metric_name = "taskmanager_job_task_latencyHistogram"
        if use_alt_metrics:
            metric_names[METRIC_TYPE_LATENCY] = alt_metric_name
            col_lists[METRIC_TYPE_LATENCY] = alt_col_list
        target_op_name = 'Sink: sink_1'
        target_stat = 'mean'
        all_latency_graph_y_top = 300
        lrb_latency_file_names = {}
        lrb_latency_dfs = {}
        lrb_latency_avgs = {}
        lrb_latency_pivoted_dfs = {}

        for scheduling_policy in lrb_scheduling_policies:
            if not skip_default or scheduling_policy != "lrb_default":
                lrb_latency_file_names[scheduling_policy] = get_filename(data_dir, experiment_date_id,
                                                                         metric_names[METRIC_TYPE_LATENCY],
                                                                         file_date,
                                                                         scheduling_policy, parallelism_level,
                                                                         default_sched_period if scheduling_policy == "lrb_default" else scheduling_period,
                                                                         num_parts, iter, exp_host)
                if use_alt_metrics:
                    lrb_latency_dfs[scheduling_policy], lrb_latency_avgs[scheduling_policy] = get_formatted_alt_latency(
                        lrb_latency_file_names[scheduling_policy], col_lists[METRIC_TYPE_LATENCY],
                        lower_time_threshold,
                        upper_time_threshold,
                        lrb_offsets["lrb_default"],
                        target_op_name, target_stat)
                    lrb_latency_pivoted_dfs[scheduling_policy] = get_pivoted_alt_latency(
                        lrb_latency_file_names[scheduling_policy],
                        col_lists[METRIC_TYPE_LATENCY],
                        target_stat,
                        upper_time_threshold,
                        lower_time_threshold)
                else:
                    print(lrb_op_name_id_dicts[default_id_str])
                    target_op_id = lrb_op_name_id_dicts[default_id_str][target_op_name]
                    lrb_latency_dfs[scheduling_policy], lrb_latency_avgs[scheduling_policy] = get_formatted_latency(
                        lrb_latency_file_names[scheduling_policy], col_lists[METRIC_TYPE_LATENCY],
                        lower_time_threshold,
                        upper_time_threshold,
                        lrb_offsets["lrb_default"],
                        target_op_id, target_stat)
                    lrb_latency_pivoted_dfs[scheduling_policy] = get_pivoted_latency(
                        lrb_latency_file_names[scheduling_policy],
                        col_lists[METRIC_TYPE_LATENCY],
                        target_stat,
                        lrb_op_name_id_dicts[scheduling_policy],
                        upper_time_threshold,
                        lower_time_threshold)

                fig_all_ops, ax_all_ops = plt.subplots(figsize=(8, 6))
                if use_alt_metrics:
                    # op_name_array = ['vehicle_win_1 -> Map', 'toll_win_1 -> Map', 'toll_acc_win_1', 'Sink: sink_1']
                    op_name_array = ['Sink: sink_1']
                else:
                    # op_name_array = ['prj_1', 'vehicle_win_1', 'toll_win_1', 'toll_acc_win_1', 'Sink: sink_1']
                    op_name_array = ['prj_1', 'Sink: sink_1']

                lrb_latency_pivoted_dfs[scheduling_policy].plot(x="rel_time", y=op_name_array, ax=ax_all_ops)
                ax_all_ops.set(xlabel="Time (sec)", ylabel="Latency (ms)",
                               title=lrb_labels[scheduling_policy] + " Latency (" + target_stat + ") - All Operators ")
                ax_all_ops.set_ylim(bottom=0)
                plt.savefig(
                    results_dir + "/latency_" + (
                        "custom_" if use_alt_metrics else "flink_") + scheduling_policy + "_" + parallelism_level + "_all_" + target_stat + "_" + experiment_date_id + "_iter" + iter + ".png")
                plt.show()

        fig_lat, ax = plt.subplots(figsize=(8, 5))

        for scheduling_policy in lrb_scheduling_policies:
            if skip_default and scheduling_policy == "lrb_default":
                ax._get_lines.get_next_color()
            if not skip_default or scheduling_policy != "lrb_default":
                ax.plot(lrb_latency_dfs[scheduling_policy]["rel_time"], lrb_latency_dfs[scheduling_policy][target_stat],
                        label=lrb_labels[scheduling_policy])
                plt.axhline(y=lrb_latency_avgs[scheduling_policy], ls='--', color='c',
                            label=lrb_labels[scheduling_policy] + "-Avg")
                offset = (list(lrb_latency_avgs).index(scheduling_policy) + 1) * 5
                plt.text(160, lrb_latency_avgs[scheduling_policy] - offset, target_stat + ' - ' + lrb_labels[
                    scheduling_policy] + ' = ' + f'{lrb_latency_avgs[scheduling_policy]:,.2f}')

        # ax.set_ylim(bottom=0)
        ax.set(xlabel="Time (sec)", ylabel="Latency (ms)",
               title=(
                         "Custom " if use_alt_metrics else "Flink ") + "Latency (" + target_stat + ") - Operator: " + target_op_name)
        ax.tick_params(axis="x", rotation=0)
        # ax.set_ylim(0, all_latency_graph_y_top)
        ax.set_ylim(bottom=0)
        ax.legend()
        # plt.tight_layout()
        plt.savefig(
            results_dir + "/latency_" + (
                "custom_" if use_alt_metrics else "flink_") + parallelism_level + "_" + target_op_name + "_" + target_stat + "_" + experiment_date_id + "_iter" + iter + ".png")
        plt.show()

    if plot_cpu:
        print("Plotting CPU usage...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["CPU Usage", "CPU", "(%)"]
        col_lists[METRIC_TYPE_CPU] = ["name", "time", "value"]
        plot_graphs_for("taskmanager_System_CPU_Usage", col_lists[METRIC_TYPE_CPU], metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies,
                        lrb_file_names, lrb_metric_dfs, lrb_metric_avgs)

    if plot_mem:
        print("Plotting memory usage...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["Heap Memory", "mem", "MB"]
        col_lists[METRIC_TYPE_MEM] = ["name", "time", "value"]
        plot_graphs_for("taskmanager_Status_JVM_Memory_Heap_Used", col_lists[METRIC_TYPE_MEM], metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, div_all_by=1048576)

    if plot_busy:
        print("Plotting busy time...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["Busy Time", "busy_time", "(ms/sec)"]
        col_lists[METRIC_TYPE_BUSY_TIME] = ["name", "task_name", "subtask_index", "time", "value"]
        group_by_col_names = ["time", "task_name"]
        plot_graphs_for("taskmanager_job_task_busyTimeMsPerSecond", col_lists[METRIC_TYPE_BUSY_TIME],
                        metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, group_by_cols=group_by_col_names,
                        filter_cols=['fil_1', 'tsw_1', 'prj_1', 'Sink: sink_1'])

    if plot_idle:
        print("Plotting idle time...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["Idle Time", "idle_time", "(ms/sec)"]
        col_lists[METRIC_TYPE_IDLE_TIME] = ["name", "task_name", "subtask_index", "time", "value"]
        group_by_col_names = ["time", "task_name"]
        plot_graphs_for("taskmanager_job_task_idleTimeMsPerSecond", col_lists[METRIC_TYPE_IDLE_TIME],
                        metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, group_by_cols=group_by_col_names,
                        filter_cols=['fil_1', 'tsw_1', 'prj_1', 'Sink: sink_1'])

    if plot_backpressure:
        print("Plotting back-pressured time...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["BP Time", "bp_time", "(ms/sec)"]
        col_lists[METRIC_TYPE_BP_TIME] = ["name", "task_name", "subtask_index", "time", "value"]
        group_by_col_names = ["time", "task_name"]
        plot_graphs_for("taskmanager_job_task_backPressuredTimeMsPerSecond", col_lists[METRIC_TYPE_BP_TIME],
                        metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, group_by_cols=group_by_col_names,
                        filter_cols=['fil_1', 'tsw_1', 'prj_1', 'Sink: sink_1'])

    if plot_iq_len:
        print("Plotting input queue length...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["Input Queue Length", "iq_len", "(# buffers)"]
        col_lists[METRIC_TYPE_IQ_LEN] = ["name", "task_name", "subtask_index", "time", "value"]
        group_by_col_names = ["time", "task_name"]
        plot_graphs_for("taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                        col_lists[METRIC_TYPE_IQ_LEN],
                        metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, group_by_cols=group_by_col_names,
                        filter_cols=['fil_1', 'tsw_1', 'prj_1', 'Sink: sink_1'])

    if plot_nw:
        nw_if = "enp4s0"
        print("Plotting network rcv rate...")
        lrb_file_names = {}
        lrb_metric_dfs = {}
        lrb_metric_avgs = {}

        target_attributes_for_metric = ["value"]
        metric_type_labels = ["Network Receive Rate", "nw", "(Bytes/sec)"]
        col_lists[METRIC_TYPE_NW] = ["name", "host", "time", "value"]
        group_by_col_names = ["time", "host"]
        metric_name = "taskmanager_System_Network_{}_ReceiveRate".format(nw_if)
        plot_graphs_for(metric_name,
                        col_lists[METRIC_TYPE_NW],
                        metric_type_labels,
                        target_attributes_for_metric, lrb_scheduling_policies, lrb_file_names, lrb_metric_dfs,
                        lrb_metric_avgs, group_by_cols=group_by_col_names)
