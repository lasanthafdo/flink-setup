# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import os

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
    print("Printing values for LRB-" + policy_name)
    print(lrb_src_df)
    return lrb_src_df, lrb_avg


def get_filename(data_directory, exp_id, metric_name, file_date, sched_policy):
    return data_directory + "/" + exp_id + \
           "/" + metric_name + "_" + sched_policy + "_" + file_date + ".csv"


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_dir = "/home/m34ferna/flink-tests/data"
    exp_date_id = "jun-28"
    file_date_default = "2022_06_28"
    file_date_adaptive = "2022_06_28"
    results_dir = "results/" + exp_date_id
    os.makedirs(results_dir, exist_ok=True)
    metric_name = "taskmanager_job_task_operator_numRecordsOutPerSecond"
    lrb_default_num_out_file = data_dir + "/" + exp_date_id + \
                               "/" + metric_name + "_lrb_default_" + file_date_default + ".csv"
    lrb_adaptive_num_out_file = data_dir + "/" + exp_date_id + \
                                "/" + metric_name + "_lrb_adaptive_" + file_date_adaptive + ".csv"
    lrb_replicating_num_out_file = data_dir + "/" + exp_date_id + \
                                   "/" + metric_name + "_lrb_replicating_" + file_date_adaptive + ".csv"

    upper_time_threshold = 500
    lower_time_threshold = 100
    plot_tp = True
    plot_cpu = True
    plot_mem = True
    plot_busy = True
    plot_idle = True
    plot_backpressure = True
    plot_iq_len = True
    has_replicating_only_metrics = True

    if plot_tp:
        col_list = ["name", "time", "operator_name", "task_name", "subtask_index", "count", "rate"]
        # col_list = ["name", "time", "operator_name", "subtask_index", "rate"]
        lrb_default_src_df, lrb_default_avg = get_formatted_tput(lrb_default_num_out_file, col_list,
                                                                 lower_time_threshold, upper_time_threshold, 0,
                                                                 "Default")

        if has_replicating_only_metrics:
            replicating_offset = 0
            lrb_replicating_src_df, lrb_replicating_avg = get_formatted_tput(lrb_replicating_num_out_file, col_list,
                                                                             lower_time_threshold, upper_time_threshold,
                                                                             replicating_offset, "Replicating")
        else:
            lrb_replicating_src_df = None
            lrb_replicating_avg = None

        adaptive_offset = 0
        lrb_adaptive_src_df, lrb_adaptive_avg = get_formatted_tput(lrb_adaptive_num_out_file, col_list,
                                                                   lower_time_threshold, upper_time_threshold,
                                                                   adaptive_offset, "Adaptive")

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.plot(lrb_default_src_df["rel_time"], lrb_default_src_df["rate"], label="LRB-Default")
        if has_replicating_only_metrics:
            ax.plot(lrb_replicating_src_df["rel_time"], lrb_replicating_src_df["rate"], label="LRB-Replicating")
        ax.plot(lrb_adaptive_src_df["rel_time"], lrb_adaptive_src_df["rate"], label="LRB-Adaptive")

        plt.axhline(y=lrb_default_avg, ls='--', color='c', label="LRB-Default-Avg")
        plt.text(100, lrb_default_avg + 5000, 'Default Avg. TP = ' + f'{lrb_default_avg:,.2f}')

        if has_replicating_only_metrics:
            plt.axhline(y=lrb_replicating_avg, ls='--', color='m', label="LRB-Replicating-Avg")
            plt.text(200, lrb_replicating_avg + 5000, 'Replicating Avg. TP = ' + f'{lrb_replicating_avg:,.2f}')

        plt.axhline(y=lrb_adaptive_avg, ls='--', color='r', label="LRB-Adaptive-Avg")
        plt.text(160, lrb_adaptive_avg + 5000, 'Adaptive Avg. TP = ' + f'{lrb_adaptive_avg:,.2f}')

        # ax.set_ylim(bottom=0)
        ax.set(xlabel="Time (sec)", ylabel="Throughput (event/sec)", title="Throughput")
        ax.tick_params(axis="x", rotation=0)
        ax.legend()
        plt.savefig(results_dir + "/throughput_" + exp_date_id + ".png")
        plt.show()

        count_fig, count_ax = plt.subplots(figsize=(12, 6))

        count_ax.plot(lrb_default_src_df["rel_time"], lrb_default_src_df["count"], label="LRB-Default")
        if has_replicating_only_metrics:
            count_ax.plot(lrb_replicating_src_df["rel_time"], lrb_replicating_src_df["count"], label="LRB-Replicating")
        count_ax.plot(lrb_adaptive_src_df["rel_time"], lrb_adaptive_src_df["count"], label="LRB-Adaptive")
        # plt.axhline(y=lrb_default_avg, ls='--', color='c', label="LRB-Default-Avg")
        # plt.axhline(y=lrb_replicating_avg, ls='--', color='m', label="LRB-Replicating-Avg")
        # plt.axhline(y=lrb_adaptive_avg, ls='--', color='r', label="LRB-Adaptive-Avg")
        # plt.text(100, lrb_default_avg - 8000, 'Default Avg. TP = ' + f'{lrb_default_avg:,.2f}')
        # plt.text(20, lrb_replicating_avg + 5000, 'Replicating Avg. TP = ' + f'{lrb_replicating_avg:,.2f}')
        # plt.text(300, lrb_adaptive_avg + 5000, 'Adaptive Avg. TP = ' + f'{lrb_adaptive_avg:,.2f}')

        # count_ax.set_ylim(bottom=0)
        count_ax.set(xlabel="Time (sec)", ylabel="Total events", title="Event count")
        count_ax.tick_params(axis="x", rotation=0)
        count_ax.legend()
        plt.savefig(results_dir + "/count_" + exp_date_id + ".png")
        plt.show()

    if plot_cpu:
        lrb_default_cpu_usage_file = get_filename(data_dir, exp_date_id, "taskmanager_System_CPU_Usage",
                                                  file_date_default,
                                                  "lrb_default")
        if has_replicating_only_metrics:
            lrb_replicating_cpu_usage_file = get_filename(data_dir, exp_date_id, "taskmanager_System_CPU_Usage",
                                                          file_date_default,
                                                          "lrb_replicating")
        else:
            lrb_replicating_cpu_usage_file = None

        lrb_adaptive_cpu_usage_file = get_filename(data_dir, exp_date_id, "taskmanager_System_CPU_Usage",
                                                   file_date_default,
                                                   "lrb_adaptive")
        cpu_usage_col_list = ["name", "time", "value"]
        cpu_usage_df = pd.read_csv(lrb_default_cpu_usage_file, usecols=cpu_usage_col_list)
        cpu_usage_df['rel_time'] = cpu_usage_df['time'].subtract(cpu_usage_df['time'].min()).div(
            1_000_000_000)
        print(cpu_usage_df)

        if has_replicating_only_metrics:
            repl_cpu_usage_df = pd.read_csv(lrb_replicating_cpu_usage_file, usecols=cpu_usage_col_list)
            # last_before_start = adapt_cpu_usage_df.loc[adapt_cpu_usage_df['value'] < 1].iloc[-1]['time']
            # adapt_cpu_usage_df = adapt_cpu_usage_df.loc[adapt_cpu_usage_df['time'] > last_before_start]
            repl_cpu_usage_df['rel_time'] = repl_cpu_usage_df['time'].subtract(repl_cpu_usage_df['time'].min()).div(
                1_000_000_000)
            print(repl_cpu_usage_df)
        else:
            repl_cpu_usage_df = None

        adapt_cpu_usage_df = pd.read_csv(lrb_adaptive_cpu_usage_file, usecols=cpu_usage_col_list)
        # last_before_start = adapt_cpu_usage_df.loc[adapt_cpu_usage_df['value'] < 1].iloc[-1]['time']
        # adapt_cpu_usage_df = adapt_cpu_usage_df.loc[adapt_cpu_usage_df['time'] > last_before_start]
        adapt_cpu_usage_df['rel_time'] = adapt_cpu_usage_df['time'].subtract(adapt_cpu_usage_df['time'].min()).div(
            1_000_000_000)
        print(adapt_cpu_usage_df)

        cpu_fig, cpu_ax = plt.subplots(figsize=(12, 8))

        cpu_ax.plot(cpu_usage_df["rel_time"], cpu_usage_df["value"], label="LRB-Default")
        if has_replicating_only_metrics:
            cpu_ax.plot(repl_cpu_usage_df["rel_time"], repl_cpu_usage_df["value"], label="LRB-Replicating")
        cpu_ax.plot(adapt_cpu_usage_df["rel_time"], adapt_cpu_usage_df["value"], label="LRB-Adaptive")

        # cpu_ax.set_ylim(bottom=0)
        cpu_ax.set(xlabel="Time (sec)", ylabel="CPU Usage (%)", title="CPU Usage")
        cpu_ax.tick_params(axis="x", rotation=0)
        cpu_ax.legend()
        plt.savefig(results_dir + "/cpu_" + exp_date_id + ".png")
        plt.show()

    if plot_mem:
        lrb_default_mem_usage_file = get_filename(data_dir, exp_date_id, "taskmanager_Status_JVM_Memory_Heap_Used",
                                                  file_date_default,
                                                  "lrb_default")
        if has_replicating_only_metrics:
            lrb_replicating_mem_usage_file = get_filename(data_dir, exp_date_id, "taskmanager_Status_JVM_Memory_Heap_Used",
                                                       file_date_default,
                                                       "lrb_replicating")
        else:
            lrb_replicating_mem_usage_file = None

        lrb_adaptive_mem_usage_file = get_filename(data_dir, exp_date_id, "taskmanager_Status_JVM_Memory_Heap_Used",
                                                   file_date_default,
                                                   "lrb_adaptive")
        mem_usage_col_list = ["name", "time", "value"]
        mem_usage_df = pd.read_csv(lrb_default_mem_usage_file, usecols=mem_usage_col_list)
        mem_usage_df['rel_time'] = mem_usage_df['time'].subtract(mem_usage_df['time'].min()).div(
            1_000_000_000)
        mem_usage_df['value'] = mem_usage_df['value'].div(1048576)
        print(mem_usage_df)

        if has_replicating_only_metrics:
            repl_mem_usage_df = pd.read_csv(lrb_replicating_mem_usage_file, usecols=mem_usage_col_list)
            repl_mem_usage_df['rel_time'] = repl_mem_usage_df['time'].subtract(repl_mem_usage_df['time'].min()).div(
                1_000_000_000)
            repl_mem_usage_df['value'] = repl_mem_usage_df['value'].div(1048576)
            print(repl_mem_usage_df)
        else:
            repl_mem_usage_df = None

        adapt_mem_usage_df = pd.read_csv(lrb_adaptive_mem_usage_file, usecols=mem_usage_col_list)
        adapt_mem_usage_df['rel_time'] = adapt_mem_usage_df['time'].subtract(adapt_mem_usage_df['time'].min()).div(
            1_000_000_000)
        adapt_mem_usage_df['value'] = adapt_mem_usage_df['value'].div(1048576)
        print(adapt_mem_usage_df)

        mem_fig, mem_ax = plt.subplots(figsize=(12, 8))

        mem_ax.plot(mem_usage_df["rel_time"], mem_usage_df["value"], label="LRB-Default")
        if has_replicating_only_metrics:
            mem_ax.plot(repl_mem_usage_df["rel_time"], repl_mem_usage_df["value"], label="LRB-Replicating")
        mem_ax.plot(adapt_mem_usage_df["rel_time"], adapt_mem_usage_df["value"], label="LRB-Adaptive")

        # mem_ax.set_ylim(bottom=0)
        mem_ax.set(xlabel="Time (sec)", ylabel="Memory Usage (MB)", title="Heap Memory")
        mem_ax.tick_params(axis="x", rotation=0)
        mem_ax.legend()
        plt.savefig(results_dir + "/mem_" + exp_date_id + ".png")
        plt.show()

    if plot_busy:
        lrb_default_busy_time_file = get_filename(data_dir, exp_date_id, "taskmanager_job_task_busyTimeMsPerSecond",
                                                  file_date_default,
                                                  "lrb_default")
        if has_replicating_only_metrics:
            lrb_replicating_busy_time_file = get_filename(data_dir, exp_date_id,
                                                       "taskmanager_job_task_busyTimeMsPerSecond",
                                                       file_date_default,
                                                       "lrb_replicating")
        else:
            lrb_replicating_busy_time_file = None

        lrb_adaptive_busy_time_file = get_filename(data_dir, exp_date_id, "taskmanager_job_task_busyTimeMsPerSecond",
                                                   file_date_default,
                                                   "lrb_adaptive")
        busy_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        busy_time_df = pd.read_csv(lrb_default_busy_time_file, usecols=busy_time_col_list)
        busy_time_grouped_df = busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
        # print(busy_time_grouped_df['task_name'].unique())
        busy_time_grouped_df['rel_time'] = busy_time_grouped_df['time'].subtract(
            busy_time_grouped_df['time'].min()).div(
            1_000_000_000)
        print(busy_time_grouped_df)

        if has_replicating_only_metrics:
            repl_busy_time_df = pd.read_csv(lrb_replicating_busy_time_file, usecols=busy_time_col_list)
            # last_before_start = repl_busy_time_df.loc[repl_busy_time_df['value'] < 1].iloc[-1]['time']
            # repl_busy_time_df = repl_busy_time_df.loc[repl_busy_time_df['time'] > last_before_start]
            repl_busy_time_grouped_df = repl_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
            repl_busy_time_grouped_df['rel_time'] = repl_busy_time_grouped_df['time'].subtract(
                repl_busy_time_grouped_df['time'].min()).div(
                1_000_000_000)
            print(repl_busy_time_grouped_df)
        else:
            repl_busy_time_grouped_df = None

        adapt_busy_time_df = pd.read_csv(lrb_adaptive_busy_time_file, usecols=busy_time_col_list)
        # last_before_start = adapt_busy_time_df.loc[adapt_busy_time_df['value'] < 1].iloc[-1]['time']
        # adapt_busy_time_df = adapt_busy_time_df.loc[adapt_busy_time_df['time'] > last_before_start]
        adapt_busy_time_grouped_df = adapt_busy_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
        adapt_busy_time_grouped_df['rel_time'] = adapt_busy_time_grouped_df['time'].subtract(
            adapt_busy_time_grouped_df['time'].min()).div(
            1_000_000_000)
        print(adapt_busy_time_grouped_df)

        ax_busy_time_default = busy_time_grouped_df.groupby('task_name')['value'].plot(legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("ms/sec")
        plt.title("Busy Time (ms/sec) - Default")
        plt.savefig(results_dir + "/busy_time_default_" + exp_date_id + ".png")
        plt.show()

        if has_replicating_only_metrics:
            ax_busy_time_repl = repl_busy_time_grouped_df.groupby('task_name')['value'].plot(legend=True)
            plt.xlabel("Time (sec)")
            plt.ylabel("ms/sec")
            plt.title("Busy Time (ms/sec) - Replicating")
            plt.savefig(results_dir + "/busy_time_replicating_" + exp_date_id + ".png")
            plt.show()

        ax_busy_time_adapt = adapt_busy_time_grouped_df.groupby('task_name')['value'].plot(legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("ms/sec")
        plt.title("Busy Time (ms/sec) - Adaptive")
        plt.savefig(results_dir + "/busy_time_adaptive_" + exp_date_id + ".png")
        plt.show()

    if plot_idle:
        lrb_default_idle_time_file = get_filename(data_dir, exp_date_id, "taskmanager_job_task_idleTimeMsPerSecond",
                                                  file_date_default,
                                                  "lrb_default")
        lrb_adaptive_idle_time_file = get_filename(data_dir, exp_date_id, "taskmanager_job_task_idleTimeMsPerSecond",
                                                   file_date_default,
                                                   "lrb_adaptive")
        idle_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        idle_time_df = pd.read_csv(lrb_default_idle_time_file, usecols=idle_time_col_list)
        idle_time_grouped_df = idle_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
        # print(idle_time_grouped_df['task_name'].unique())
        idle_time_grouped_df['rel_time'] = idle_time_grouped_df['time'].subtract(
            idle_time_grouped_df['time'].min()).div(
            1_000_000_000)
        print(idle_time_grouped_df)

        adapt_idle_time_df = pd.read_csv(lrb_adaptive_idle_time_file, usecols=idle_time_col_list)
        # last_before_start = adapt_idle_time_df.loc[adapt_idle_time_df['value'] < 1].iloc[-1]['time']
        # adapt_idle_time_df = adapt_idle_time_df.loc[adapt_idle_time_df['time'] > last_before_start]
        adapt_idle_time_grouped_df = adapt_idle_time_df.groupby(['time', 'task_name'])['value'].mean().reset_index()
        adapt_idle_time_grouped_df['rel_time'] = adapt_idle_time_grouped_df['time'].subtract(
            adapt_idle_time_grouped_df['time'].min()).div(
            1_000_000_000)
        print(adapt_idle_time_grouped_df)

        ax_idle_time_default = idle_time_grouped_df.groupby('task_name')['value'].plot(legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("ms/sec")
        plt.title("Idle Time (ms/sec) - Default")
        plt.savefig(results_dir + "/idle_time_default_" + exp_date_id + ".png")
        plt.show()

        ax_idle_time_adapt = adapt_idle_time_grouped_df.groupby('task_name')['value'].plot(legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("ms/sec")
        plt.title("Idle Time (ms/sec) - Adaptive")
        plt.savefig(results_dir + "/idle_time_adaptive_" + exp_date_id + ".png")
        plt.show()

    if plot_backpressure:
        lrb_default_backpressured_time_file = get_filename(data_dir, exp_date_id,
                                                           "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                           file_date_default,
                                                           "lrb_default")
        lrb_adaptive_backpressured_time_file = get_filename(data_dir, exp_date_id,
                                                            "taskmanager_job_task_backPressuredTimeMsPerSecond",
                                                            file_date_default,
                                                            "lrb_adaptive")
        backpressured_time_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        backpressured_time_df = pd.read_csv(lrb_default_backpressured_time_file, usecols=backpressured_time_col_list)
        backpressured_time_grouped_df = backpressured_time_df.groupby(['time', 'task_name'])[
            'value'].mean().reset_index()
        # print(backpressured_time_grouped_df['task_name'].unique())
        backpressured_time_grouped_df['rel_time'] = backpressured_time_grouped_df['time'].subtract(
            backpressured_time_grouped_df['time'].min()).div(
            1_000_000_000)
        print(backpressured_time_grouped_df)

        adapt_backpressured_time_df = pd.read_csv(lrb_adaptive_backpressured_time_file,
                                                  usecols=backpressured_time_col_list)
        # last_before_start = adapt_backpressured_time_df.loc[adapt_backpressured_time_df['value'] < 1].iloc[-1]['time']
        # adapt_backpressured_time_df = adapt_backpressured_time_df.loc[adapt_backpressured_time_df['time'] > last_before_start]
        adapt_backpressured_time_grouped_df = adapt_backpressured_time_df.groupby(['time', 'task_name'])[
            'value'].mean().reset_index()
        adapt_backpressured_time_grouped_df['rel_time'] = adapt_backpressured_time_grouped_df['time'].subtract(
            adapt_backpressured_time_grouped_df['time'].min()).div(
            1_000_000_000)
        print(adapt_backpressured_time_grouped_df)

        ax_backpressured_time_default = backpressured_time_grouped_df.groupby('task_name')['value'].plot(legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("ms/sec")
        plt.title("BP Time (ms/sec) - Default")
        plt.savefig(results_dir + "/backpressured_time_default_" + exp_date_id + ".png")
        plt.show()

        ax_backpressured_time_adapt = adapt_backpressured_time_grouped_df.groupby('task_name')['value'].plot(
            legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("ms/sec")
        plt.title("BP Time (ms/sec) - Adaptive")
        plt.savefig(results_dir + "/backpressured_time_adaptive_" + exp_date_id + ".png")
        plt.show()

    if plot_iq_len:
        lrb_default_iq_len_file = get_filename(data_dir, exp_date_id,
                                               "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                               file_date_default,
                                               "lrb_default")
        lrb_adaptive_iq_len_file = get_filename(data_dir, exp_date_id,
                                                "taskmanager_job_task_Shuffle_Netty_Input_Buffers_inputQueueLength",
                                                file_date_default,
                                                "lrb_adaptive")
        iq_len_col_list = ["name", "task_name", "subtask_index", "time", "value"]
        iq_len_df = pd.read_csv(lrb_default_iq_len_file, usecols=iq_len_col_list)
        iq_len_grouped_df = iq_len_df.groupby(['time', 'task_name'])['value'].sum().reset_index()
        # print(iq_len_grouped_df['task_name'].unique())
        iq_len_grouped_df['rel_time'] = iq_len_grouped_df['time'].subtract(iq_len_grouped_df['time'].min()).div(
            1_000_000_000)
        print(iq_len_grouped_df)

        adapt_iq_len_df = pd.read_csv(lrb_adaptive_iq_len_file,
                                      usecols=iq_len_col_list)
        # last_before_start = adapt_iq_len_df.loc[adapt_iq_len_df['value'] < 1].iloc[-1]['time']
        # adapt_iq_len_df = adapt_iq_len_df.loc[adapt_iq_len_df['time'] > last_before_start]
        adapt_iq_len_grouped_df = adapt_iq_len_df.groupby(['time', 'task_name'])['value'].sum().reset_index()
        adapt_iq_len_grouped_df['rel_time'] = adapt_iq_len_grouped_df['time'].subtract(
            adapt_iq_len_grouped_df['time'].min()).div(1_000_000_000)
        print(adapt_iq_len_grouped_df)

        ax_iq_len_default = iq_len_grouped_df.groupby('task_name')['value'].plot(legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("Num. buffers")
        plt.title("Input Queue Length - Default")
        plt.savefig(results_dir + "/iq_len_default_" + exp_date_id + ".png")
        plt.show()

        ax_iq_len_adapt = adapt_iq_len_grouped_df.groupby('task_name')['value'].plot(
            legend=True)
        plt.xlabel("Time (sec)")
        plt.ylabel("Num. buffers")
        plt.title("Input Queue Length - Adaptive")
        plt.savefig(results_dir + "/iq_len_adaptive_" + exp_date_id + ".png")
        plt.show()
