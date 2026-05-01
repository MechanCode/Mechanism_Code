#!/usr/bin/env python3
"""
Sensor data alignment and labeling pipeline.

Features:
- Read acc and Gyroscope CSV files from ./data/proband1/data/
- For each action, align sensor data from different body positions to a
    common timeline (50Hz, 20ms interval)
- Resample with linear interpolation
- Append action label as the last column
- Export merged CSV output

Alignment method: common timeline + linear interpolation
Sampling rate: 50Hz (20ms interval)
"""

import os
import re
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import interpolate

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
SAMPLING_INTERVAL_MS = 20  # 50Hz = 20ms interval
VALID_SENSORS = {'acc', 'Gyroscope'}
POSITIONS = ['chest', 'forearm', 'head', 'shin', 'thigh', 'upperarm', 'waist']
TIME_COL_CANDIDATES = ['timestamp', 'time', 't', 'ts', 'unix_time', 'attr_time']


def parse_filename(filename: str) -> Optional[Tuple[str, str, str]]:
    """
    Parse filename and extract sensor type, action, and position.
    Format: {sensor}_{action}_{position}.csv
    
    Returns:
        (sensor, action, position) or None
    """
    match = re.match(r'^(acc|Gyroscope)_([a-z]+)_([a-z]+)\.csv$', filename, re.IGNORECASE)
    if match:
        sensor = match.group(1)
        action = match.group(2).lower()
        position = match.group(3).lower()
        return sensor, action, position
    return None


def detect_time_column(df: pd.DataFrame) -> str:
    """
    Automatically detect the timestamp column.
    
    Returns:
        Time column name
    
    Raises:
        ValueError: Unable to find a time column
    """
    for col in TIME_COL_CANDIDATES:
        if col in df.columns:
            return col
    
    # Try to find a column containing 'time'
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if time_cols:
        return time_cols[0]
    
    raise ValueError(f"Failed to auto-detect time column. Candidate columns: {list(df.columns)}")


def detect_time_unit(sample_value: float) -> str:
    """
    Infer timestamp unit by numeric magnitude.
    
    Returns:
        's', 'ms', 'us', 'ns'
    """
    if sample_value > 1e15:
        return 'us'  # microseconds
    elif sample_value > 1e12:
        return 'ms'  # milliseconds
    elif sample_value > 1e9:
        return 's'   # seconds
    else:
        return 'ms'  # default to milliseconds


def standardize_timestamp(df: pd.DataFrame, time_col: str, target_unit: str = 'ms') -> pd.DataFrame:
    """
    Normalize timestamps to target unit (milliseconds by default).
    
    Returns:
        DataFrame with a standardized time_ts column
    """
    df = df.copy()
    sample_value = df[time_col].iloc[0]
    detected_unit = detect_time_unit(sample_value)
    
    # Convert to milliseconds
    if detected_unit == 's':
        df['time_ts'] = (df[time_col] * 1000).astype(np.int64)
    elif detected_unit == 'ms':
        df['time_ts'] = df[time_col].astype(np.int64)
    elif detected_unit == 'us':
        df['time_ts'] = (df[time_col] / 1000).astype(np.int64)
    elif detected_unit == 'ns':
        df['time_ts'] = (df[time_col] / 1e6).astype(np.int64)
    
    return df


def load_sensor_data(filepath: str, time_col: str = 'auto') -> pd.DataFrame:
    """
    Load a sensor CSV and standardize timestamps.
    
    Returns:
        DataFrame including time_ts and sensor data columns
    """
    df = pd.read_csv(filepath)
    
    # Detect time column
    if time_col == 'auto':
        time_col = detect_time_column(df)
    
    # Standardize timestamps
    df = standardize_timestamp(df, time_col)
    
    # Keep only time_ts and sensor data columns (attr_x, attr_y, attr_z)
    data_cols = [c for c in df.columns if c.startswith('attr_') and c != 'attr_time']
    keep_cols = ['time_ts'] + data_cols
    
    return df[keep_cols].sort_values('time_ts').reset_index(drop=True)


def interpolate_to_common_time(df: pd.DataFrame, common_time: np.ndarray, 
                                data_cols: List[str]) -> pd.DataFrame:
    """
    Interpolate data onto a shared timeline.
    
    Args:
        df: Original data containing time_ts
        common_time: Shared timeline (milliseconds)
        data_cols: Names of columns to interpolate
    
    Returns:
        Interpolated DataFrame
    """
    result = pd.DataFrame({'time_ts': common_time})
    
    original_time = df['time_ts'].values
    
    for col in data_cols:
        # Create interpolation function (linear interpolation)
        f = interpolate.interp1d(
            original_time, 
            df[col].values, 
            kind='linear',
            bounds_error=False,
            fill_value='extrapolate'
        )
        result[col] = f(common_time)
    
    return result


def group_files_by_action(input_dir: str) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Group files by action.
    
    Returns:
        {action: {position: {sensor: filepath}}}
    """
    groups = defaultdict(lambda: defaultdict(dict))
    
    for filename in os.listdir(input_dir):
        parsed = parse_filename(filename)
        if parsed:
            sensor, action, position = parsed
            if sensor in VALID_SENSORS:
                filepath = os.path.join(input_dir, filename)
                groups[action][position][sensor] = filepath
    
    return groups


def process_action(action: str, position_files: Dict[str, Dict[str, str]], 
                   time_col: str = 'auto') -> Tuple[Optional[pd.DataFrame], dict]:
    """
    Process all position data for one action.
    
    Returns:
        (Aligned DataFrame, statistics)
    """
    stats = {
        'action': action,
        'positions_used': [],
        'positions_skipped': [],
        'per_position_stats': [],
        'output_rows': 0,
        't_start': None,
        't_end': None
    }
    
    # Collect data from all valid positions
    all_data = {}  # {(sensor, position): df}
    time_ranges = []  # [(t_min, t_max), ...]
    
    for position in POSITIONS:
        if position not in position_files:
            stats['positions_skipped'].append(f"{position} (file not found)")
            continue
        
        sensor_files = position_files[position]
        
        # Check whether both acc and Gyroscope are present
        if 'acc' not in sensor_files or 'Gyroscope' not in sensor_files:
            missing = 'acc' if 'acc' not in sensor_files else 'Gyroscope'
            stats['positions_skipped'].append(f"{position} (missing {missing})")
            continue
        
        try:
            # Load acc data
            acc_df = load_sensor_data(sensor_files['acc'], time_col)
            # Load Gyroscope data
            gyro_df = load_sensor_data(sensor_files['Gyroscope'], time_col)
            
            pos_stats = {
                'position': position,
                'acc_rows': len(acc_df),
                'gyro_rows': len(gyro_df),
                'acc_time_range': [int(acc_df['time_ts'].min()), int(acc_df['time_ts'].max())],
                'gyro_time_range': [int(gyro_df['time_ts'].min()), int(gyro_df['time_ts'].max())]
            }
            stats['per_position_stats'].append(pos_stats)
            
            # Record time ranges
            time_ranges.append((acc_df['time_ts'].min(), acc_df['time_ts'].max()))
            time_ranges.append((gyro_df['time_ts'].min(), gyro_df['time_ts'].max()))
            
            all_data[('acc', position)] = acc_df
            all_data[('Gyroscope', position)] = gyro_df
            
        except Exception as e:
            stats['positions_skipped'].append(f"{position} (load error: {e})")
            continue
    
    if not all_data:
        logger.warning(f"  - Action {action} has no valid data")
        return None, stats
    
    # Compute shared time range
    # t_start = max(all minimum timestamps)
    # t_end = min(all maximum timestamps)
    t_start = max(tr[0] for tr in time_ranges)
    t_end = min(tr[1] for tr in time_ranges)
    
    if t_end <= t_start:
        logger.warning(f"  - Action {action} has empty shared time range (t_start={t_start}, t_end={t_end})")
        stats['positions_skipped'].append("all positions (empty shared time range)")
        return None, stats
    
    stats['t_start'] = int(t_start)
    stats['t_end'] = int(t_end)
    
    # Generate shared timeline (50Hz, 20ms interval)
    common_time = np.arange(t_start, t_end + 1, SAMPLING_INTERVAL_MS)
    logger.info(f"  - Shared time range: [{t_start}, {t_end}], length: {t_end - t_start} ms")
    logger.info(f"  - Shared timeline points: {len(common_time)} (50Hz, 20ms interval)")
    
    # Interpolate for each sensor/position
    aligned_dfs = []
    used_positions = set()
    
    for (sensor, position), df in all_data.items():
        # Rename data columns
        sensor_prefix = 'acc' if sensor == 'acc' else 'gyro'
        col_mapping = {
            'attr_x': f'{sensor_prefix}_{position}_x',
            'attr_y': f'{sensor_prefix}_{position}_y',
            'attr_z': f'{sensor_prefix}_{position}_z'
        }
        
        data_cols = [c for c in df.columns if c.startswith('attr_')]
        
        # Interpolate to the shared timeline
        interpolated = interpolate_to_common_time(df, common_time, data_cols)
        
        # Rename columns
        for old_col, new_col in col_mapping.items():
            if old_col in interpolated.columns:
                interpolated = interpolated.rename(columns={old_col: new_col})
        
        aligned_dfs.append(interpolated)
        used_positions.add(position)
    
    stats['positions_used'] = sorted(list(used_positions))
    
    # Merge all data (aligned by time_ts)
    result = aligned_dfs[0]
    for df in aligned_dfs[1:]:
        # Remove duplicated time_ts column
        df_to_merge = df.drop(columns=['time_ts'], errors='ignore')
        result = pd.concat([result, df_to_merge], axis=1)
    
    # Ensure time_ts is integer
    result['time_ts'] = result['time_ts'].astype(np.int64)
    
    stats['output_rows'] = len(result)
    
    logger.info(f"  - Output rows: {len(result)}")
    logger.info(f"  - Positions used: {stats['positions_used']}")
    
    return result, stats


def validate_output(df: pd.DataFrame) -> List[str]:
    """
    Validate output data quality.
    
    Returns:
        List of error/warning messages
    """
    issues = []
    
    # Check whether time_ts is the first column
    if df.columns[0] != 'time_ts':
        issues.append(f"First column should be time_ts, got {df.columns[0]}")
    
    # Check whether label is the last column
    if df.columns[-1] != 'label':
        issues.append(f"Last column should be label, got {df.columns[-1]}")
    
    # Check for NaN values
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        issues.append(f"Columns containing NaN values: {nan_cols}")
    
    # Validate timeline per action segment
    for action in df['label'].unique():
        action_df = df[df['label'] == action]
        
        # Check whether time_ts is strictly increasing
        time_diff = action_df['time_ts'].diff().dropna()
        if not (time_diff > 0).all():
            issues.append(f"Action {action}: time_ts is not strictly increasing")
        
        # Check whether interval is close to 20ms
        median_interval = time_diff.median()
        if abs(median_interval - SAMPLING_INTERVAL_MS) > 1:
            issues.append(
                f"Action {action}: median interval is {median_interval}ms, expected {SAMPLING_INTERVAL_MS}ms"
            )
    
    return issues


def main():
    parser = argparse.ArgumentParser(description='Sensor data alignment and labeling pipeline')
    parser.add_argument('--input_dir', type=str, default='data/proband1/data',
                        help='Input data directory')
    parser.add_argument('--output_dir', type=str, default='data/proband1/processed',
                        help='Output directory')
    parser.add_argument('--time_col', type=str, default='auto',
                        help='Time column name (auto for automatic detection)')
    parser.add_argument('--time_unit', type=str, default='auto',
                        help='Time unit (auto/s/ms/us/ns)')
    parser.add_argument('--write_aligned_per_action', type=str, default='true',
                        help='Whether to write per-action aligned output')
    
    args = parser.parse_args()
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'aligned'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    
    # Configure file logging
    file_handler = logging.FileHandler(
        os.path.join(args.output_dir, 'logs', 'processing.log'),
        mode='w',
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 60)
    logger.info("Sensor data alignment and labeling pipeline started")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("Alignment method: common timeline + linear interpolation")
    logger.info(f"Sampling rate: 50Hz (interval {SAMPLING_INTERVAL_MS}ms)")
    logger.info(f"Time column: {args.time_col}")
    logger.info(f"Time unit: {args.time_unit}")
    logger.info("=" * 60)
    
    # Group files by action
    action_groups = group_files_by_action(args.input_dir)
    logger.info(f"Discovered {len(action_groups)} actions")
    
    # Process each action
    all_results = []
    all_stats = []
    
    for action in sorted(action_groups.keys()):
        position_files = action_groups[action]
        logger.info(f"\nProcessing action: {action}")
        
        result_df, stats = process_action(action, position_files, args.time_col)
        all_stats.append(stats)
        
        if result_df is not None and len(result_df) > 0:
            # Add label column
            result_df['label'] = action
            all_results.append(result_df)
            
            # Save aligned output for each action
            if args.write_aligned_per_action.lower() == 'true':
                aligned_path = os.path.join(args.output_dir, 'aligned', f'aligned_{action}.csv')
                result_df.to_csv(aligned_path, index=False)
                logger.info(f"  - Saved: {aligned_path}")
    
    if not all_results:
        logger.error("No valid output data!")
        return 1
    
    # Sort action segments by time order (using each segment's t_start)
    action_order = []
    for df in all_results:
        action = df['label'].iloc[0]
        t_start = df['time_ts'].iloc[0]
        action_order.append((t_start, action, df))
    
    action_order.sort(key=lambda x: x[0])
    
    # Merge all action segments
    sorted_results = [item[2] for item in action_order]
    final_df = pd.concat(sorted_results, ignore_index=True)
    
    # Ensure column order: time_ts first, label last
    data_cols = [c for c in final_df.columns if c not in ['time_ts', 'label']]
    final_cols = ['time_ts'] + sorted(data_cols) + ['label']
    final_df = final_df[final_cols]
    
    # Save final result
    output_path = os.path.join(args.output_dir, 'merged_aligned_labeled.csv')
    final_df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\nFinal output file: {output_path}")
    logger.info(f"Total rows: {len(final_df)}")
    logger.info(f"Total columns: {len(final_df.columns)}")
    logger.info(f"Action order: {[item[1] for item in action_order]}")
    
    # Validate output
    logger.info("\n" + "=" * 60)
    logger.info("Validating output...")
    issues = validate_output(final_df)
    if issues:
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("  - All checks passed ✓")
    
    # Save statistics
    stats_path = os.path.join(args.output_dir, 'logs', 'alignment_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    logger.info(f"\nStatistics saved: {stats_path}")
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Processing summary:")
    logger.info(f"  - Actions processed: {len(action_groups)}")
    logger.info(f"  - Actions with valid output: {len(all_results)}")
    logger.info(f"  - Total output rows: {len(final_df)}")
    logger.info(f"  - Output columns: {list(final_df.columns)}")
    logger.info(f"  - Label set: {sorted(final_df['label'].unique())}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
