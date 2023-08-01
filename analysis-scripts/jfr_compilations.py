import pandas as pd
import os
import glob

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', 1000)

exp_date_id = "jul-26-1"
results_dir = "results/" + exp_date_id + "/perf/"
os.makedirs(results_dir, exist_ok=True)
dir_path = "/home/m34ferna/flink-tests/data/jfr/compilations"
file_paths = glob.glob(dir_path + '/*.csv')  # Assumes the files are named data_1.txt, data_2.txt, etc.
print(file_paths)
data_frames = []

policy_index = 0
for file_path in file_paths:
    jfr_compile_data = pd.read_csv(file_path)
    jfr_compile_data['Inlined Code Size'] = jfr_compile_data['Inlined Code Size'].str.replace(r'\D', '',
                                                                                              regex=True).astype(int)
    policy_name = file_path.split(".")[0].split("/")[-1]
    jfr_compile_data['policy'] = policy_name
    jfr_compile_data['policy_index'] = int(policy_name.split("_")[0])
    policy_index += 1
    data_frames.append(jfr_compile_data)

raw_data = pd.concat(data_frames).reset_index()
print(raw_data)
policy_info = raw_data[['policy','policy_index']].drop_duplicates()
print(policy_info)
grpdf = raw_data.groupby(["Java Method", "policy"], as_index=False).agg(
    per_count=('Java Method', 'count'),
    code_size_bytes=('Inlined Code Size', 'last'), policy_index=('policy_index', 'last'))
intermediate_grouping = grpdf
print(intermediate_grouping)
grouped_info = intermediate_grouping.groupby("Java Method").agg(method_count=('Java Method', 'count'),
                                                                tot_count=('per_count', 'sum'),
                                                                avg_code_size_bytes=('code_size_bytes', 'mean'),
                                                                std_code_size_bytes=('code_size_bytes', 'std'),
                                                                policy_mask=('policy_index', sum),
                                                                policies=('policy', lambda x: list(x))).reset_index()
# filtered_methods = grouped_info[
#    ((("OS_DEF_Good" in grouped_info.policies) | ("Default_Good" in grouped_info.policies)) & (
#                "OS_DEF_Bad" not in grouped_info.policies))]
filtered_methods = grouped_info[grouped_info.policy_mask == 7].sort_values(by=['policy_mask'])
print(filtered_methods)
