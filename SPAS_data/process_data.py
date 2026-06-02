import sys
import os
import numpy as np
import pandas as pd

'''
Read a time-series CSV file.
The first row is treated as the header, the last column is treated as the
label, and all other columns are returned as feature data.
'''
def read_time_series(file_path):
    # Read the data; keep the header and split the last column as labels.
    df_raw = pd.read_csv(file_path, header=0)
    # Return feature headers, feature values, and labels separately.
    train_header = df_raw.columns[:-1]
    train_data = df_raw.iloc[:, :-1].values
    train_label = df_raw.iloc[:, -1].values
    return train_header, train_data, train_label


'''
Generate noised data through SPAS_workflow.
epsilon, raw_stream, and window_size vary by call; warm-up and update-window
settings are fixed here.
'''
from SPAS import SPAS_workflow
def generate_noised_data(epsilon, window_size, raw_stream, seed, sensitivity_s, sensitivity_p):

    windownum_warm = 1
    windownum_updateE = 2

    dim = len(raw_stream[0])

    published_stream = SPAS_workflow(epsilon, sensitivity_s, sensitivity_p, raw_stream, window_size, windownum_warm, windownum_updateE, dim, seed)
    return published_stream


'''
Generate noised training data for each epsilon value.
For each epsilon, generate files in batches under noised_data_original and
store them as train_noised_<i>.csv in the matching epsilon directory.
'''
def generate_noised_data_for_epsilon(file_path, train_header, train_data, train_label, epsilon_list, w, start_i=0, end_i=30):
    # Create the noised data directory.
    noised_data_dir = os.path.join(os.path.dirname(file_path), 'noised_data_original')
    os.makedirs(noised_data_dir, exist_ok=True)

    # sensitivity_p is the maximum per-dimension range in train_data.
    sensitivity_p = np.max(np.ptp(train_data, axis=0))
    # sensitivity_s is the maximum absolute difference between adjacent rows.
    sensitivity_s = np.max(np.abs(np.diff(train_data, axis=0)), axis=0).max()
    # sensitivity_p = 2
    # sensitivity_s = 2
    for epsilon in epsilon_list:
        # Create the directory for this epsilon value.
        epsilon_dir = os.path.join(noised_data_dir, f'epsilon_{epsilon}')
        if os.path.exists(epsilon_dir):
            print(f"Warning: Directory {epsilon_dir} already exists. Files may be overwritten.")
        else:
            os.makedirs(epsilon_dir, exist_ok=True)
        window_size = int(w * len(train_data))
        print(f"Generating noised data for epsilon={epsilon}, window_size={window_size}, i=[{start_i},{end_i})...")
        for i in range(start_i, end_i):
            seed = 42 + i
            noised_train_data = generate_noised_data(epsilon, window_size, train_data, seed, sensitivity_s, sensitivity_p)
            # Build a DataFrame with feature headers and append labels.
            noised_df = pd.DataFrame(noised_train_data, columns=train_header)
            noised_df['label'] = train_label
            # print(noised_df.shape)
            index = i
            noised_train_file = os.path.join(epsilon_dir, f'train_noised_{index}.csv')
            noised_df.to_csv(noised_train_file, index=False)
            print(f"  Saved train_noised_{index}.csv (epsilon={epsilon})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--epsilon', type=float, default=None)
    parser.add_argument('--start_i', type=int, default=0, help='Start file index, inclusive')
    parser.add_argument('--end_i', type=int, default=30, help='End file index, exclusive')
    args = parser.parse_args()

    file_paths = ['/data/yulian/traffic/traffic.csv', '/data/yulian/DynamicMTS/dynamic_data.csv']
    epsilon_list = [5.5, 5, 4.5, 4, 3.5, 3, 2.5, 2, 1.5, 1]
    w = 0.005

    # # Single-run mode: values passed from the shell.
    # args.file_path = '/data/yulian/classifier_dataset/training_data.csv'
    # args.epsilon = 3
    # args.start_i = 0
    # args.end_i = 1 # Script inputs: file_path, epsilon, and batch range.

    if args.file_path is not None and args.epsilon is not None:
        train_header, train_data, train_label = read_time_series(args.file_path)
        generate_noised_data_for_epsilon(
            args.file_path, train_header, train_data, train_label,
            [args.epsilon], w, start_i=args.start_i, end_i=args.end_i)
    else:
        # Original sequential mode fallback.
        for file_th in file_paths:
            train_header, train_data, train_label = read_time_series(file_th)
            generate_noised_data_for_epsilon(
                file_th, train_header, train_data, train_label,
                epsilon_list, w)




    # epsilon = [6]
    # w = 0.005
    # file_path = '/data/yulian/traffic/traffic.csv'
    # train_data, val_data, test_data, train_date, val_date, test_date = read_time_series(file_path)
    # # val_file = os.path.join(os.path.dirname(file_path), 'val_data.csv')
    # # test_file = os.path.join(os.path.dirname(file_path), 'test_data.csv')
    # # _val = np.hstack((val_date.to_numpy().reshape(-1, 1), val_data)) if val_date is not None else val_data
    # # _test = np.hstack((test_date.to_numpy().reshape(-1, 1), test_data)) if test_date is not None else test_data
    # # pd.DataFrame(_val).to_csv(val_file, index=False, header=False)
    # # pd.DataFrame(_test).to_csv(test_file, index=False, header=False)
    # sensitivity_p = np.max(np.ptp(train_data, axis=0))
    # # sensitivity_s is the maximum absolute difference between adjacent rows.
    # sensitivity_s = np.max(np.abs(np.diff(train_data, axis=0)), axis=0).max()
    # print(sensitivity_p, sensitivity_s)
    # # Print per-dimension ranges, then report the top 5 dimensions by range.
    # ptp_values = np.ptp(train_data, axis=0)
    # top5_indices = np.argsort(ptp_values)[::-1][:5]
    # print("Top 5 dimensions by range (max-min):")
    # for idx in top5_indices:
    #     print(f"Dimension {idx}: Range={ptp_values[idx]}, Max={np.max(train_data[:, idx])}, Min={np.min(train_data[:, idx])}")
    # # Check whether train_data was normalized by inspecting mean and std.
    # print("Train data mean (should be close to 0):", np.mean(train_data, axis=0))
    # print("Train data std (should be close to 1):", np.std(train_data, axis=0))
