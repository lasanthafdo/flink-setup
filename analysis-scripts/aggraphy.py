# This is a sample Python script.
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format


def get_formatted_tput(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset):
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


def get_sum_value_for_task(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset,
                           task='Source:'):
    print("Reading file : " + lrb_num_out_file)
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    # print(lrb_df[lrb_df['task_name'].str.contains(task)])
    lrb_src_df = lrb_df[lrb_df['task_name'].str.contains(task)].drop(
        ['name'], axis=1).groupby(['time'])[['value']].sum().reset_index()
    # print(lrb_src_df)
    lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_src_df = lrb_src_df.loc[
        (lrb_src_df['rel_time'] > lower_threshold) & (
                lrb_src_df['rel_time'] < upper_threshold)]
    lrb_avg = np.mean(lrb_src_df['value'])
    return lrb_src_df, lrb_avg


def get_avg_value_for_task(lrb_num_out_file, column_list, lower_threshold, upper_threshold, offset,
                           task='Source:'):
    print("Reading file : " + lrb_num_out_file)
    lrb_df = pd.read_csv(lrb_num_out_file, usecols=column_list)
    print(lrb_df[lrb_df['task_name'].str.contains(task)])
    lrb_src_df = lrb_df[lrb_df['task_name'].str.contains(task)].drop(
        ['name'], axis=1).groupby(['time'])[['value']].mean().reset_index()
    print(lrb_src_df)
    lrb_src_df['rel_time'] = lrb_src_df['time'].subtract(lrb_src_df['time'].min()).div(
        1_000_000_000).subtract(offset)
    lrb_src_df = lrb_src_df.loc[
        (lrb_src_df['rel_time'] > lower_threshold) & (
                lrb_src_df['rel_time'] < upper_threshold)]
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


def get_op_name_id_mapping(lrb_tp_file):
    op_name_id_mapping_df = pd.read_csv(lrb_tp_file, usecols=['operator_name', 'operator_id']).drop_duplicates()
    op_name_id_dict = dict(zip(op_name_id_mapping_df['operator_name'], op_name_id_mapping_df['operator_id']))
    return op_name_id_dict


def get_filename(data_directory, exp_id, metric_name, file_date, sched_policy, par_level='12', sched_period='0',
                 num_parts='1', iter='1', global_iter='0'):
    return data_directory + "/" + exp_id + \
        "/" + metric_name + "_" + sched_policy + "_" + sched_period + "ms_" + par_level + "_" + num_parts + "parts_iter" + global_iter + "_" + iter + "_" + file_date + ".csv"


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
    experiment_date_id = "aug-15-1"
    file_date = "2023_08_15"
    results_dir = "results/" + experiment_date_id + "/agg"
    os.makedirs(results_dir, exist_ok=True)

    upper_time_threshold = 1200
    lower_time_threshold = 480
    plot_tp = True
    plot_latency = True
    plot_src_cpu_time = True
    plot_cpu_time_comparison = False

    has_replicating_only_metrics = False
    has_scheduling_only_metrics = True
    has_bpscheduling_only_metrics = False
    has_pseudo_default_metrics = False
    has_adaptive_metrics = False

    default_offset = 600
    scheduling_offset = 600

    num_repeats = 5
    sched_period = 5
    default_sched_period = str(sched_period)
    pd_sched_period = str(sched_period)
    default_id_str = "lrb_osdef"
    default_label_str = "1-OS_Default-LQ"
    pd_id_str = "lrb_osdef"
    pd_label_str = "2-OS_Default-LQ"
    sched_label_str = "2-Scheduling-LQ"
    num_parts = '1'
    iter_to_skip = [3]

    sched_periods = [sched_period]
    # parallelism_levels = [6, 12, 18, 24]
    parallelism_levels = [1]

    metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
    lrb_default_tp_file = get_filename(data_dir, experiment_date_id, metric_name, file_date, default_id_str,
                                       str(parallelism_levels[0]), default_sched_period, num_parts)
    lrb_default_op_name_id_dict = get_op_name_id_mapping(lrb_default_tp_file)

    if plot_tp:
        col_list = ["name", "time", "operator_name", "task_name", "subtask_index", "count", "rate"]
        lrb_avg_all_df = pd.DataFrame(columns=['Scheduling Policy', 'iter', 'Parallelism', 'tp'])
        for parallelism_level in parallelism_levels:
            for iter in range(1, num_repeats + 1):
                if iter in iter_to_skip: continue
                lrb_default_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                        default_id_str, str(parallelism_level), default_sched_period,
                                                        num_parts, str(iter))
                lrb_default_src_df, lrb_default_avg = get_formatted_tput(lrb_default_num_out_file, col_list,
                                                                         lower_time_threshold, upper_time_threshold,
                                                                         default_offset)
                lrb_avg_all_df.loc[len(lrb_avg_all_df)] = [default_label_str, iter, parallelism_level, lrb_default_avg]
                if has_replicating_only_metrics:
                    replicating_offset = 0
                    lrb_replicating_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                                "lrb_replicating", str(parallelism_level),
                                                                num_parts=num_parts, iter=str(iter))
                    lrb_replicating_src_df, lrb_replicating_avg = get_formatted_tput(lrb_replicating_num_out_file,
                                                                                     col_list,
                                                                                     lower_time_threshold,
                                                                                     upper_time_threshold,
                                                                                     replicating_offset)
                    lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Replicating", iter, parallelism_level,
                                                               lrb_replicating_avg]

                if has_pseudo_default_metrics:
                    pd_offset = 100
                    lrb_pd_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date, pd_id_str,
                                                       str(parallelism_level), pd_sched_period, num_parts, str(iter))
                    lrb_pd_src_df, lrb_pd_avg = get_formatted_tput(lrb_pd_num_out_file,
                                                                   col_list,
                                                                   lower_time_threshold,
                                                                   upper_time_threshold,
                                                                   pd_offset)
                    lrb_avg_all_df.loc[len(lrb_avg_all_df)] = [pd_label_str, iter,
                                                               parallelism_level,
                                                               lrb_pd_avg]

                if has_scheduling_only_metrics:
                    for sched_period in sched_periods:
                        lrb_scheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                                   "lrb_scheduling", str(parallelism_level),
                                                                   str(sched_period), num_parts, str(iter))
                        lrb_scheduling_src_df, lrb_scheduling_avg = get_formatted_tput(lrb_scheduling_num_out_file,
                                                                                       col_list,
                                                                                       lower_time_threshold,
                                                                                       upper_time_threshold,
                                                                                       scheduling_offset)
                        lrb_avg_all_df.loc[len(lrb_avg_all_df)] = [sched_label_str + f"{sched_period:03d}", iter,
                                                                   parallelism_level,
                                                                   lrb_scheduling_avg]

                if has_bpscheduling_only_metrics:
                    for sched_period in sched_periods:
                        lrb_bpscheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name,
                                                                     file_date, "lrb_bpscheduling",
                                                                     str(parallelism_level), str(sched_period),
                                                                     num_parts, str(iter))
                        lrb_bpscheduling_src_df, lrb_bpscheduling_avg = get_formatted_tput(
                            lrb_bpscheduling_num_out_file,
                            col_list,
                            lower_time_threshold,
                            upper_time_threshold,
                            scheduling_offset)
                        lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["4 - Scheduling-" + f"{sched_period:03d} - Small Q",
                                                                   iter,
                                                                   parallelism_level,
                                                                   lrb_bpscheduling_avg]

                if has_adaptive_metrics:
                    adaptive_offset = 0
                    lrb_adaptive_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                             "lrb_adaptive", str(parallelism_level),
                                                             num_parts=num_parts, iter=str(iter))
                    lrb_adaptive_src_df, lrb_adaptive_avg = get_formatted_tput(lrb_adaptive_num_out_file, col_list,
                                                                               lower_time_threshold,
                                                                               upper_time_threshold,
                                                                               adaptive_offset)
                    lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Adaptive", iter, parallelism_level, lrb_adaptive_avg]

        lrb_avg_all_df = lrb_avg_all_df.groupby(['Scheduling Policy', 'Parallelism']).agg(
            {'tp': ['mean', 'std', 'count']}).reset_index()
        lrb_avg_all_df.columns = ['_'.join(tup).rstrip('_') for tup in lrb_avg_all_df.columns]
        lrb_avg_all_df['conf_interval_95'] = lrb_avg_all_df['tp_std'] * 1.96 / (np.sqrt(lrb_avg_all_df['tp_count']))
        pivoted_lrb_avg_all_df = lrb_avg_all_df.pivot(index='Parallelism', columns='Scheduling Policy',
                                                      values=['tp_mean', 'conf_interval_95']).reset_index()
        print(pivoted_lrb_avg_all_df)

        ax = pivoted_lrb_avg_all_df.plot.bar(x='Parallelism', y='tp_mean', rot=0, yerr='conf_interval_95', capsize=10)
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_ylabel('Events/sec')
        ax.legend(loc="lower left")
        plt.title('LRB Throughput - ' + num_parts + ' source replicas')
        plt.tight_layout()
        plt.savefig(results_dir + "/throughput_all_" + experiment_date_id + ".png")
        plt.show()

    if plot_latency:
        col_list = ["name", "time", "operator_id", "operator_subtask_index", "mean", "p50", "p95", "p99"]
        metric_name = "taskmanager_job_latency_source_id_operator_id_operator_subtask_index_latency"
        target_op_name = 'Sink: sink_1'
        target_stat = 'mean'

        print(lrb_default_op_name_id_dict)

        lrb_avg_latency_all_df = pd.DataFrame(columns=['Scheduling Policy', 'iter', 'Parallelism', 'latency'])
        for parallelism_level in parallelism_levels:
            for iter in range(1, num_repeats + 1):
                if iter in iter_to_skip: continue
                lrb_default_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                        default_id_str, str(parallelism_level), default_sched_period,
                                                        num_parts, str(iter))
                target_op_id = lrb_default_op_name_id_dict[target_op_name]
                lrb_default_src_df, lrb_default_avg = get_formatted_latency(lrb_default_num_out_file, col_list,
                                                                            lower_time_threshold, upper_time_threshold,
                                                                            default_offset, target_op_id, target_stat)
                lrb_avg_latency_all_df.loc[len(lrb_avg_latency_all_df)] = [default_label_str, iter, parallelism_level,
                                                                           lrb_default_avg]
                if has_replicating_only_metrics:
                    replicating_offset = 0
                    lrb_replicating_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                                "lrb_replicating", str(parallelism_level),
                                                                num_parts=num_parts, iter=str(iter))
                    lrb_replicating_src_df, lrb_replicating_avg = get_formatted_latency(lrb_replicating_num_out_file,
                                                                                        col_list, lower_time_threshold,
                                                                                        upper_time_threshold,
                                                                                        replicating_offset,
                                                                                        target_op_id,
                                                                                        target_stat)
                    lrb_avg_latency_all_df.loc[len(lrb_avg_latency_all_df)] = ["Replicating", iter, parallelism_level,
                                                                               lrb_replicating_avg]

                if has_pseudo_default_metrics:
                    pd_offset = 100
                    lrb_pd_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date, pd_id_str,
                                                       str(parallelism_level), pd_sched_period, num_parts, str(iter))
                    target_op_id = lrb_default_op_name_id_dict[target_op_name]
                    lrb_pd_src_df, lrb_pd_avg = get_formatted_latency(lrb_pd_num_out_file,
                                                                      col_list, lower_time_threshold,
                                                                      upper_time_threshold,
                                                                      pd_offset, target_op_id,
                                                                      target_stat)
                    lrb_avg_latency_all_df.loc[len(lrb_avg_latency_all_df)] = [pd_label_str, iter, parallelism_level,
                                                                               lrb_pd_avg]

                if has_scheduling_only_metrics:
                    for sched_period in sched_periods:
                        lrb_scheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                                   "lrb_scheduling", str(parallelism_level),
                                                                   str(sched_period), num_parts, str(iter))
                        target_op_id = lrb_default_op_name_id_dict[target_op_name]
                        lrb_scheduling_src_df, lrb_scheduling_avg = get_formatted_latency(lrb_scheduling_num_out_file,
                                                                                          col_list,
                                                                                          lower_time_threshold,
                                                                                          upper_time_threshold,
                                                                                          scheduling_offset,
                                                                                          target_op_id,
                                                                                          target_stat)
                        lrb_avg_latency_all_df.loc[len(lrb_avg_latency_all_df)] = [
                            sched_label_str + f"{sched_period:03d}", iter,
                            parallelism_level, lrb_scheduling_avg]

                if has_bpscheduling_only_metrics:
                    for sched_period in sched_periods:
                        lrb_bpscheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name,
                                                                     file_date,
                                                                     "lrb_bpscheduling", str(parallelism_level),
                                                                     str(sched_period), num_parts, str(iter))
                        target_op_id = lrb_default_op_name_id_dict[target_op_name]
                        lrb_bpscheduling_src_df, lrb_bpscheduling_avg = get_formatted_latency(
                            lrb_bpscheduling_num_out_file,
                            col_list,
                            lower_time_threshold,
                            upper_time_threshold,
                            scheduling_offset,
                            target_op_id,
                            target_stat)
                        lrb_avg_latency_all_df.loc[len(lrb_avg_latency_all_df)] = [
                            "3 - Scheduling-" + f"{sched_period:03d} - Small Q", iter,
                            parallelism_level, lrb_bpscheduling_avg]

                if has_adaptive_metrics:
                    adaptive_offset = 0
                    lrb_adaptive_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                             "lrb_adaptive", str(parallelism_level),
                                                             num_parts=num_parts, iter=str(iter))
                    lrb_adaptive_src_df, lrb_adaptive_avg = get_formatted_tput(lrb_adaptive_num_out_file, col_list,
                                                                               lower_time_threshold,
                                                                               upper_time_threshold,
                                                                               adaptive_offset)
                    lrb_avg_latency_all_df.loc[len(lrb_avg_latency_all_df)] = ["Adaptive", iter, parallelism_level,
                                                                               lrb_adaptive_avg]

        lrb_avg_latency_all_df = lrb_avg_latency_all_df.groupby(['Scheduling Policy', 'Parallelism']).agg(
            {'latency': ['mean', 'std', 'count']}).reset_index()
        lrb_avg_latency_all_df.columns = ['_'.join(tup).rstrip('_') for tup in lrb_avg_latency_all_df.columns]
        lrb_avg_latency_all_df['conf_interval_95'] = lrb_avg_latency_all_df['latency_std'] * 1.96 / (
            np.sqrt(lrb_avg_latency_all_df['latency_count']))
        pivoted_lrb_avg_latency_all_df = lrb_avg_latency_all_df.pivot(index='Parallelism', columns='Scheduling Policy',
                                                                      values=['latency_mean',
                                                                              'conf_interval_95']).reset_index()
        print(pivoted_lrb_avg_latency_all_df)

        ax = pivoted_lrb_avg_latency_all_df.plot.bar(x='Parallelism', y='latency_mean', rot=0, yerr='conf_interval_95',
                                                     capsize=10)
        ax.ticklabel_format(style='plain', axis='y')
        ax.set_ylabel('Time (ms)')
        # ax.set_ylim(top=250)
        ax.legend()
        plt.title('LRB ' + target_stat + ' latency - ' + num_parts + ' source replicas - ' + target_op_name)
        plt.tight_layout()
        plt.savefig(
            results_dir + "/latency_all_" + target_op_name + "_" + target_stat + "_" + experiment_date_id + ".png")
        plt.show()

    if plot_src_cpu_time:
        col_list = ["name", "time", "task_name", "subtask_index", "value"]
        metric_name = "taskmanager_job_task_cpuTime"
        lrb_avg_all_df = pd.DataFrame(columns=['Scheduling Policy', 'iter', 'Parallelism', 'cpu_time'])
        for parallelism_level in parallelism_levels:
            for iter in range(1, num_repeats + 1):
                if iter in iter_to_skip: continue
                if has_pseudo_default_metrics:
                    lrb_default_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                            pd_id_str,
                                                            str(parallelism_level), pd_sched_period, num_parts)
                    lrb_default_src_df, lrb_default_avg = get_sum_value_for_task(lrb_default_num_out_file, col_list,
                                                                                 lower_time_threshold,
                                                                                 upper_time_threshold,
                                                                                 default_offset)
                    lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["PseudoDefault", iter, parallelism_level,
                                                               lrb_default_avg]

                if has_replicating_only_metrics:
                    replicating_offset = 0
                    lrb_replicating_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                                "lrb_replicating", str(parallelism_level),
                                                                num_parts=num_parts)
                    lrb_replicating_src_df, lrb_replicating_avg = get_sum_value_for_task(lrb_replicating_num_out_file,
                                                                                         col_list,
                                                                                         lower_time_threshold,
                                                                                         upper_time_threshold,
                                                                                         replicating_offset)
                    lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Replicating", iter, parallelism_level,
                                                               lrb_replicating_avg]

                if has_scheduling_only_metrics:
                    for sched_period in sched_periods:
                        lrb_scheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                                   "lrb_scheduling", str(parallelism_level),
                                                                   str(sched_period), num_parts)
                        lrb_scheduling_src_df, lrb_scheduling_avg = get_sum_value_for_task(
                            lrb_scheduling_num_out_file,
                            col_list,
                            lower_time_threshold,
                            upper_time_threshold,
                            scheduling_offset)
                        lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Scheduling-" + f"{sched_period:03d}",
                                                                   parallelism_level,
                                                                   iter, lrb_scheduling_avg]

                if has_adaptive_metrics:
                    adaptive_offset = 0
                    lrb_adaptive_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                             "lrb_adaptive", str(parallelism_level),
                                                             num_parts=num_parts)
                    lrb_adaptive_src_df, lrb_adaptive_avg = get_sum_value_for_task(lrb_adaptive_num_out_file,
                                                                                   col_list,
                                                                                   lower_time_threshold,
                                                                                   upper_time_threshold,
                                                                                   adaptive_offset)
                    lrb_avg_all_df.loc[len(lrb_avg_all_df)] = ["Adaptive", iter, parallelism_level, lrb_adaptive_avg]

        lrb_avg_all_df = lrb_avg_all_df.groupby(['Scheduling Policy', 'Parallelism']).mean().reset_index()
        pivoted_lrb_avg_all_df = lrb_avg_all_df.pivot(index='Parallelism', columns='Scheduling Policy',
                                                      values='cpu_time')
        # print(pivoted_lrb_avg_all_df)

        ax = pivoted_lrb_avg_all_df.plot.bar(rot=0)
        # ax.ticklabel_format(style='plain', axis='y')
        ax.set_ylabel('ms')
        ax.legend(loc="lower left")
        plt.title('LRB CPU time - 4 source replicas')
        plt.tight_layout()
        plt.savefig(results_dir + "/cpu_time_all_" + experiment_date_id + ".png")
        plt.show()

    if plot_cpu_time_comparison:
        col_list = ["name", "time", "task_name", "subtask_index", "value"]
        metric_name = "taskmanager_job_task_cpuTime"
        lrb_avg_default_df = pd.DataFrame(columns=['Scheduling Policy', 'Parallelism', 'cpu_time'])
        op_order = ["Source", "fil_1", "vehicle_win_1", "speed_win_1", "acc_win_1", "toll_win_1", "Sink"]

        if has_pseudo_default_metrics:
            for parallelism_level in parallelism_levels:
                lrb_default_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                        default_id_str, str(parallelism_level), default_sched_period,
                                                        num_parts)
                for task_name in op_order:
                    lrb_default_src_df, lrb_default_avg = get_sum_value_for_task(lrb_default_num_out_file, col_list,
                                                                                 lower_time_threshold,
                                                                                 upper_time_threshold,
                                                                                 default_offset, task_name)
                    lrb_avg_default_df.loc[len(lrb_avg_default_df)] = [task_name, parallelism_level,
                                                                       lrb_default_avg]

            print(lrb_avg_default_df.dtypes)
            pivoted_lrb_avg_default_df = lrb_avg_default_df.pivot(index='Parallelism', columns='Scheduling Policy',
                                                                  values='cpu_time')[op_order]
            print(pivoted_lrb_avg_default_df)

            ax = pivoted_lrb_avg_default_df.plot.bar(rot=0)
            # ax.ticklabel_format(style='plain', axis='y')
            ax.set_ylim(0, 800000000000)
            ax.set_ylabel('ms')
            ax.legend()
            plt.title('LRB CPU time - Default')
            plt.tight_layout()
            plt.savefig(results_dir + "/cpu_time_comp_default_" + experiment_date_id + ".png")
            plt.show()

        lrb_avg_sched_df = pd.DataFrame(columns=['Scheduling Policy', 'Parallelism', 'cpu_time'])
        for parallelism_level in parallelism_levels:
            scheduling_offset = 0
            for sched_period in sched_periods:
                lrb_scheduling_num_out_file = get_filename(data_dir, experiment_date_id, metric_name, file_date,
                                                           "lrb_scheduling", str(parallelism_level), str(sched_period),
                                                           num_parts)
                for task_name in op_order:
                    lrb_scheduling_src_df, lrb_scheduling_avg = get_sum_value_for_task(
                        lrb_scheduling_num_out_file,
                        col_list,
                        lower_time_threshold,
                        upper_time_threshold,
                        scheduling_offset, task_name)
                    lrb_avg_sched_df.loc[len(lrb_avg_sched_df)] = [task_name, parallelism_level,
                                                                   lrb_scheduling_avg]

        pivoted_lrb_avg_sched_df = lrb_avg_sched_df.pivot(index='Parallelism', columns='Scheduling Policy',
                                                          values='cpu_time')[op_order]
        print(pivoted_lrb_avg_sched_df)

        ax = pivoted_lrb_avg_sched_df.plot.bar(rot=0)
        # ax.ticklabel_format(style='plain', axis='y')
        ax.set_ylim(0, 800000000000)
        ax.set_ylabel('ms')
        ax.legend()
        plt.title('LRB CPU time - Scheduling')
        plt.tight_layout()
        plt.savefig(results_dir + "/cpu_time_comp_sched_" + experiment_date_id + ".png")
        plt.show()
