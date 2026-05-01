"""
USC-HAD dataset processing script
================================
Features:
1. Read .mat files from the USC-HAD dataset
2. Apply per-subject z-score normalization on feature dimensions
3. Split data into training, validation, and test sets
4. Save outputs in CSV format

Dataset split:
- Training set (train): Subject 1, 2, 5, 8, 9, 10, 11, 12, 13
- Validation set (validation): Subject 3, 6
- Test set (test): Subject 4, 7

Column description:
- acc_x, acc_y, acc_z: 3-axis acceleration (unit: g)
- gyro_x, gyro_y, gyro_z: 3-axis angular velocity (unit: dps)
- activity: activity label (1-12)
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import zscore
from collections import defaultdict


# ============== Configuration ==============
BASE_PATH = ""
OUTPUT_PATH = ""

# Dataset split
TRAIN_SUBJECTS = [1, 2,   6, 8, 11, 12, 13]      # Subject 1-9
VAL_SUBJECTS = [4, 5, 9]        # Subject 3, 6
TEST_SUBJECTS = [3, 7, 10]      # Subject 4, 7

# Column definitions
FEATURE_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABEL_COLUMN = 'activity'
ALL_COLUMNS = FEATURE_COLUMNS + [LABEL_COLUMN]

# Activity name mapping
ACTIVITY_NAMES = {
    1: "Walking Forward",
    2: "Walking Left",
    3: "Walking Right",
    4: "Walking Upstairs",
    5: "Walking Downstairs",
    6: "Running Forward",
    7: "Jumping Up",
    8: "Sitting",
    9: "Standing",
    10: "Sleeping",
    11: "Elevator Up",
    12: "Elevator Down"
}


def load_single_mat_file(mat_file_path):
    """
    Load a single .mat file and extract sensor_readings and activity label.
    
    Args:
        mat_file_path: Full path to the .mat file
    
    Returns:
        sensor_data: numpy array, shape (N, 6)
        activity_num: int, activity ID
    """
    # Parse activity ID from filename: a{m}t{n}.mat
    filename = os.path.basename(mat_file_path)
    activity_num = int(filename.split('t')[0].replace('a', ''))
    
    # Load .mat file
    data = loadmat(mat_file_path)
    sensor_readings = data['sensor_readings']  # shape: (N, 6)
    
    return sensor_readings, activity_num


def load_subject_data(subject_num, apply_zscore=True):
    """
    Load all data for a single subject and optionally apply z-score normalization.
    
    Args:
        subject_num: Subject ID
        apply_zscore: Whether to apply z-score normalization for this subject
    
    Returns:
        data_with_labels: numpy array, shape (N_total, 7), where the last column is activity label
    """
    subject_folder = f"Subject{subject_num}"
    subject_path = os.path.join(BASE_PATH, subject_folder)
    
    if not os.path.exists(subject_path):
        print(f"Warning: {subject_folder} does not exist, skipping")
        return None
    
    # Collect all data for this subject
    all_sensor_data = []
    all_labels = []
    
    mat_files = sorted(glob.glob(os.path.join(subject_path, "*.mat")))
    
    for mat_file in mat_files:
        sensor_data, activity_num = load_single_mat_file(mat_file)
        all_sensor_data.append(sensor_data)
        all_labels.extend([activity_num] * len(sensor_data))
    
    # Merge all data for this subject
    sensor_data_combined = np.vstack(all_sensor_data)  # shape: (N_total, 6)
    labels = np.array(all_labels).reshape(-1, 1)       # shape: (N_total, 1)
    
    # Apply z-score normalization per feature dimension for this subject
    if apply_zscore:
        sensor_data_normalized = zscore(sensor_data_combined, axis=0)
        # Handle possible NaN values (when standard deviation is 0)
        sensor_data_normalized = np.nan_to_num(sensor_data_normalized, nan=0.0)
    else:
        sensor_data_normalized = sensor_data_combined
    
    # Combine features and labels
    data_with_labels = np.hstack([sensor_data_normalized, labels])
    
    print(f"  {subject_folder}: {len(data_with_labels)} samples, "
          f"feature mean range: [{sensor_data_normalized.mean(axis=0).min():.4f}, {sensor_data_normalized.mean(axis=0).max():.4f}], "
          f"feature std range: [{sensor_data_normalized.std(axis=0).min():.4f}, {sensor_data_normalized.std(axis=0).max():.4f}]")
    
    return data_with_labels


def load_dataset(subject_nums, dataset_name, apply_zscore=True):
    """
    Load and merge data from multiple subjects.
    
    Args:
        subject_nums: List of subject IDs
        dataset_name: Dataset name (for printing)
        apply_zscore: Whether to apply z-score normalization for each subject
    
    Returns:
        df: pandas DataFrame
    """
    print(f"\nBuilding {dataset_name} (Subject {subject_nums[0]}-{subject_nums[-1]})...")
    
    all_data = []
    for subject_num in subject_nums:
        subject_data = load_subject_data(subject_num, apply_zscore=apply_zscore)
        if subject_data is not None:
            all_data.append(subject_data)
    
    if all_data:
        combined_data = np.vstack(all_data)
        df = pd.DataFrame(combined_data, columns=ALL_COLUMNS)
        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)
        return df
    
    return pd.DataFrame(columns=ALL_COLUMNS)


def count_activity_files():
    """Count data files for each activity."""
    activity_count = defaultdict(int)
    subject_count = 0
    
    for item in os.listdir(BASE_PATH):
        if item.startswith("Subject") and os.path.isdir(os.path.join(BASE_PATH, item)):
            subject_count += 1
            subject_path = os.path.join(BASE_PATH, item)
            mat_files = glob.glob(os.path.join(subject_path, "*.mat"))
            
            for mat_file in mat_files:
                filename = os.path.basename(mat_file)
                activity_num = int(filename.split('t')[0].replace('a', ''))
                activity_count[activity_num] += 1
    
    return activity_count, subject_count


def print_activity_statistics(activity_count, subject_count):
    """Print activity statistics."""
    print("\n" + "=" * 60)
    print("Activity data statistics report")
    print("=" * 60)
    print(f"Total subjects found: {subject_count}")
    print(f"\n{'Activity':<8} {'Name':<20} {'File Count':<10}")
    print("-" * 60)
    
    total = 0
    for activity_num in sorted(activity_count.keys()):
        count = activity_count[activity_num]
        name = ACTIVITY_NAMES.get(activity_num, "Unknown")
        print(f"a{activity_num:<7} {name:<20} {count:<10}")
        total += count
    
    print("-" * 60)
    print(f"{'Total':<29} {total:<10}")


def print_dataset_statistics(df, dataset_name):
    """Print dataset statistics."""
    print(f"\n{dataset_name}:")
    print(f"  Shape: {df.shape}")
    print(f"  Samples: {len(df)}")
    print(f"  Activity distribution:")
    for activity_num in sorted(df[LABEL_COLUMN].unique()):
        count = (df[LABEL_COLUMN] == activity_num).sum()
        name = ACTIVITY_NAMES.get(activity_num, "Unknown")
        print(f"    {activity_num}: {name} - {count} samples")


def save_dataset_summary(train_df, val_df, test_df):
    """Save dataset summary information."""
    summary = f"""
USC-HAD Dataset Processing Summary
==================================

Data source: USC-HAD human activity recognition dataset
Sampling rate: 100Hz
Sensors: Accelerometer (±6g) + Gyroscope (±500dps)

Preprocessing: Per-subject z-score normalization by feature dimension, then concatenation

Column description:
- acc_x, acc_y, acc_z: 3-axis acceleration (after z-score normalization)
- gyro_x, gyro_y, gyro_z: 3-axis angular velocity (after z-score normalization)
- activity: activity label (1-12)

Activity label mapping:
1 - Walking Forward
2 - Walking Left
3 - Walking Right
4 - Walking Upstairs
5 - Walking Downstairs
6 - Running Forward
7 - Jumping Up
8 - Sitting
9 - Standing
10 - Sleeping
11 - Elevator Up
12 - Elevator Down

Dataset split:
- Training set (training_data.csv): Subject 1-9, total {len(train_df)} samples
- Validation set (validation_data.csv): Subject 10-11, total {len(val_df)} samples
- Test set (testing_data.csv): Subject 12-13, total {len(test_df)} samples
"""
    
    summary_path = os.path.join(OUTPUT_PATH, "dataset_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"\nDataset summary saved to: {summary_path}")


def main():
    """Main function."""
    print("=" * 60)
    print("USC-HAD Dataset Processing Script")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Task 1 & 2: Count activity data
    print("\n[Task 1 & 2] Counting activity data...")
    activity_count, subject_count = count_activity_files()
    print_activity_statistics(activity_count, subject_count)
    
    # Task 3: Build datasets
    print("\n[Task 3] Building datasets...")
    print("Note: Data is z-score normalized per subject before concatenation")
    
    # Load datasets
    train_df = load_dataset(TRAIN_SUBJECTS, "training set", apply_zscore=True)
    val_df = load_dataset(VAL_SUBJECTS, "validation set", apply_zscore=True)
    test_df = load_dataset(TEST_SUBJECTS, "test set", apply_zscore=True)
    
    # Save datasets
    train_path = os.path.join(OUTPUT_PATH, "training_data.csv")
    val_path = os.path.join(OUTPUT_PATH, "validation_data.csv")
    test_path = os.path.join(OUTPUT_PATH, "testing_data.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    # Print statistics
    print("\n" + "=" * 60)
    print("Dataset build completed")
    print("=" * 60)
    
    print_dataset_statistics(train_df, "Training set (training_data.csv)")
    print_dataset_statistics(val_df, "Validation set (validation_data.csv)")
    print_dataset_statistics(test_df, "Test set (testing_data.csv)")
    
    # Save summary
    save_dataset_summary(train_df, val_df, test_df)
    
    print(f"\nFiles saved to: {OUTPUT_PATH}")
    print("  - training_data.csv")
    print("  - validation_data.csv")
    print("  - testing_data.csv")
    print("  - dataset_summary.txt")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
