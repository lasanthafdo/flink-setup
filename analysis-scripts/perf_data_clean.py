import math

import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

# Read data from multiple files
exp_date_id = "may-20-1"
dir_path = "/home/m34ferna/flink-tests/data/perf/" + exp_date_id
results_dir = "results/" + exp_date_id + "/perf/"
os.makedirs(results_dir, exist_ok=True)
file_paths = glob.glob(dir_path + '/perf_text_*.txt')  # Assumes the files are named data_1.txt, data_2.txt, etc.
print(file_paths)
data_frames = []

for file_path in file_paths:
    data = pd.read_csv(file_path, skiprows=3, sep='(?<=\d{3})\s+|(?<=\])\s+|\[|\]|(?<=\d{1})/', engine='python',
                       names=['time', 'u1', 'cpu', 'u2', 'task_name', 'tid', 'pid', 'u3', 'wait_time', 'sch_delay',
                              'run_time', 'desc'])
    # Clean the data
    data = data.drop(columns=['u1', 'u2', 'u3', 'desc', 'tid', 'pid'])
    data = data.dropna()  # Remove any rows with missing values

    # Clean and format specific columns
    data['task_name'] = data['task_name'].str.strip()  # Remove leading/trailing spaces in 'task_name' column
    data['task_name'] = data['task_name'].str[:12]
    data['wait_time'] = data['wait_time'].astype(float)
    data['sch_delay'] = data['sch_delay'].astype(float)
    data['run_time'] = data['run_time'].astype(float)

    data = data.groupby('task_name').agg(
        switches=('wait_time', 'count'),
        avg_wait_time=('wait_time', 'mean'),
        avg_sch_delay=('sch_delay', 'mean'),
        avg_run_time=('run_time', 'mean'),
        tot_wait_time=('wait_time', 'sum'),
        tot_sch_delay=('sch_delay', 'sum'),
        tot_run_time=('run_time', 'sum')
    ).reset_index()

    # Filter rows based on 'task_name' column values
    filter_keywords = ['Sink:', 'fil_1', 'Kafka', 'Legacy', 'Source', 'scheduler']
    data = data[data['task_name'].str.startswith(tuple(filter_keywords))]

    policy_name = (file_path.split('_')[2] + "-" + file_path.split('_')[3])  # Extract policy name from file path
    data['policy'] = policy_name  # Add a new 'Policy' column with the policy name

    print(data)
    data_frames.append(data)

# Concatenate all data frames into a single data frame
result = pd.concat(data_frames).reset_index()

# Replace the text in the 'task_name' column
result['task_name'] = result['task_name'].replace({
    'Sink: sink_1': '5 - Sink',
    'fil_1 -> tsw': '4 - Transform',
    'Kafka Fetche': '2 - Kafka Fetcher',
    'Legacy Sourc': '3 - Source Thread',
    'Source: src_': '1 - Source (Dummy)',
    'scheduler-th': '8 - Scheduler'
})

# Display the result
print(result)

# Extract the unique policies and tasks
policies = sorted(result['policy'].unique())
tasks = sorted(result['task_name'].unique())

# Set the width of each bar
bar_width = 0.8 / len(policies)  # Adjust the width based on the number of policies

# Define the columns to generate plots for
columns = ['switches', 'avg_run_time', 'avg_sch_delay', 'avg_wait_time', 'tot_run_time', 'tot_sch_delay',
           'tot_wait_time']
labels = {
    'switches': "Num context switches",
    'avg_run_time': "Avg run time (ms)",
    'avg_sch_delay': "Avg scheduling delay (ms)",
    'avg_wait_time': "Avg wait time (ms)",
    'tot_run_time': "Total run time (ms)",
    'tot_sch_delay': "Total scheduling delay (ms)",
    'tot_wait_time': "Total wait time (ms)"
}

# Generate bar plots for each column
for column in columns:
    plt.figure(figsize=(10, 6))
    for i, policy in enumerate(policies):
        policy_data = result[result['policy'] == policy].sort_values('task_name')
        adjustment_factor = bar_width * math.ceil(len(policy_data) / 2) - bar_width / 2
        x = [j - adjustment_factor + i * bar_width for j in range(len(policy_data))]
        plt.bar(x, policy_data[column], width=bar_width, label=policy)

    # Customize the plot
    plt.xlabel('Task')
    plt.ylabel(labels[column])
    plt.title(f'{labels[column]} for different policies')
    plt.xticks(range(len(tasks)), tasks)
    plt.legend()

    # Save the plot as PNG
    plt.tight_layout()
    plt.savefig(f'{results_dir}{column}.png')

    # Clear the current plot
    plt.clf()
