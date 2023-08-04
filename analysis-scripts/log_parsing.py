import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

exp_date_id = "aug-2-1"
results_dir = "results/" + exp_date_id + "/misc/"
os.makedirs(results_dir, exist_ok=True)
dir_path = "/home/m34ferna/flink-tests/data/misc/" + exp_date_id
file_paths = glob.glob(dir_path + '/*.csv')  # Assumes the files are named data_1.txt, data_2.txt, etc.
print(file_paths)
data_frames = []
offset = 30

policy_index = 0
for file_path in file_paths:
    log_data = pd.read_csv(file_path, sep='|', header=None,
                           names=['timestamp', 'task', 'input_q', 'emitted', 'cost', 'min_cost', 'busy_time', 'bp_time',
                                  'idle_time', 'tp', 'latency'])
    log_data['timestamp'].fillna(method='ffill', inplace=True)
    log_data['timestamp'] = pd.to_datetime(log_data['timestamp'])
    log_data['tp'] = pd.to_numeric(log_data['tp'].str.strip().str.replace(',', ''))
    log_data['timestamp'] = log_data['timestamp'].values.astype(np.int64) // 10 ** 6
    log_data['time_sec'] = (log_data['timestamp'] - log_data['timestamp'].min()).div(1000)
    log_data = log_data[log_data['time_sec'] > offset]
    log_data['run_type'] = "Better" if "good" in file_path else "Worse"
    log_data['latency'] = pd.to_numeric(log_data['latency'].str.strip(), errors='coerce')
    data_frames.append(log_data)

raw_data = pd.concat(data_frames).reset_index()
raw_data['time_slice'] = raw_data['time_sec'] // 30
semi_agg_data = raw_data.groupby(['run_type', 'task', 'time_slice']).agg({
    'input_q': 'mean', 'emitted': 'mean', 'cost': 'mean', 'min_cost': 'mean', 'busy_time': 'mean', 'bp_time': 'mean',
    'idle_time': 'mean', 'tp': 'mean', 'latency': 'mean', 'time_sec': 'mean', 'timestamp': 'last'
}).reset_index()
print(semi_agg_data)
agg_data = raw_data.groupby(['run_type', 'task']).agg(
    mean_cost=('cost', 'mean'),
    min_mean_cost=('min_cost', 'mean'),
    mean_tp=('tp', 'mean'),
    mean_bp_time=('bp_time', 'mean'),
    mean_busy_time=('busy_time', 'mean'),
    mean_input_q=('input_q', 'mean'),
    mean_emitted=('emitted', 'mean')
).reset_index()
print(agg_data)

unique_run_types = raw_data.run_type.unique()
unique_tasks = raw_data.task.unique()

print(unique_run_types)
print(unique_tasks)

fig, axes = plt.subplots(2, 2, sharex="col", figsize=(16, 12))

metric_of_interest = 'latency'
for run_type in unique_run_types:
    for idx, task in enumerate(unique_tasks):
        axes[idx // 2, idx % 2].plot('time_sec', metric_of_interest, label=run_type,
                                     data=semi_agg_data[
                                         semi_agg_data['run_type'].str.contains(run_type) &
                                         semi_agg_data['task'].str.contains(task)])
        axes[idx // 2, idx % 2].set(xlabel="Elapsed time (sec)", ylabel="events per sec",
                                    title=metric_of_interest + " : " + task)
        axes[idx // 2, idx % 2].tick_params(axis="x", rotation=0)
        axes[idx // 2, idx % 2].legend()

plt.savefig(f'{results_dir}{metric_of_interest}.png')
plt.show()
