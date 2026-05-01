#!/usr/bin/env python3
"""
处理所有用户数据的流水线 (proband1 - proband15)

功能：
1. 调用 process_user_data.py 处理每个用户的原始数据
2. 检查每个用户是否有8个类别，如果没有则舍弃该用户
3. 对每个用户的特征列进行归一化 (z-score 或 min-max)
4. 将所有用户数据按 1-15 顺序拼接
5. 去掉时间列，保存最终合并数据集
6. 支持分割数据集为训练集、验证集和测试集

输出：
- 每用户归一化后的 CSV: {output_root}/proband{N}/processed/merged_normalized.csv
- 每用户归一化参数: {output_root}/proband{N}/processed/norm_params.json
- 最终合并文件: ./merged_users_1_15_normalized.csv
- 统计信息: ./merged_users_1_15_normalized_stats.json
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# 定义8个标准类别及其编码 (1-8)
STANDARD_LABELS = ['climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking']
LABEL_ENCODING = {label: idx + 1 for idx, label in enumerate(STANDARD_LABELS)}
# {'climbingdown': 1, 'climbingup': 2, 'jumping': 3, 'lying': 4, 'running': 5, 'sitting': 6, 'standing': 7, 'walking': 8}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_user_range(user_range_str: str) -> List[int]:
    """
    解析用户编号范围字符串，支持逗号分隔和连字符区间
    例如: '1,3-5,7-15' -> [1, 3, 4, 5, 7, 8, ..., 15]
    """
    user_ids = set()
    parts = user_range_str.split(',')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        if '-' in part:
            try:
                start, end = part.split('-')
                user_ids.update(range(int(start), int(end) + 1))
            except ValueError:
                logger.warning(f"无法解析区间: {part}")
        else:
            try:
                user_ids.add(int(part))
            except ValueError:
                logger.warning(f"无法解析用户ID: {part}")
                
    return sorted(list(user_ids))


def check_user_has_all_labels(user_id: int, output_root: str) -> Tuple[bool, List[str]]:
    """
    检查用户是否有全部8个类别
    
    Returns:
        (has_all_labels, existing_labels)
    """
    input_file = os.path.join(output_root, f'proband{user_id}', 'processed', 'merged_aligned_labeled.csv')
    
    if not os.path.exists(input_file):
        logger.warning(f"用户 {user_id} 的对齐数据不存在: {input_file}")
        return False, []
    
    # 读取数据
    df = pd.read_csv(input_file)
    
    if 'label' not in df.columns:
        logger.warning(f"用户 {user_id} 数据中没有 label 列")
        return False, []
    
    existing_labels = df['label'].unique().tolist()
    existing_labels = [str(label) for label in existing_labels]
    
    # 检查是否有全部8个标准类别
    missing_labels = [label for label in STANDARD_LABELS if label not in existing_labels]
    
    if missing_labels:
        logger.warning(f"用户 {user_id} 缺少类别: {missing_labels}")
        return False, existing_labels
    
    logger.info(f"用户 {user_id} 有全部 8 个类别 ✓")
    return True, existing_labels


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    将字符串类别编码为数字 1-8
    
    Returns:
        编码后的 DataFrame
    """
    if 'label' not in df.columns:
        return df
    
    # 将标签编码为 1-8
    df['label'] = df['label'].map(LABEL_ENCODING)
    
    # 检查是否有未映射的标签
    if df['label'].isna().any():
        unmapped_count = df['label'].isna().sum()
        logger.warning(f"有 {unmapped_count} 行标签未能映射")
        # 删除未映射的行
        df = df.dropna(subset=['label'])
    
    df['label'] = df['label'].astype(int)
    
    return df


def run_process_user_data(user_id: int, input_root: str, output_root: str, 
                          continue_on_error: bool = False) -> bool:
    """
    调用 process_user_data.py 处理单个用户的数据
    
    Returns:
        True if successful, False otherwise
    """
    input_dir = os.path.join(input_root, f'proband{user_id}', 'data')
    output_dir = os.path.join(output_root, f'proband{user_id}', 'processed')
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        logger.warning(f"用户 {user_id} 的数据目录不存在: {input_dir}")
        return False
    
    # 检查是否有数据文件
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        logger.warning(f"用户 {user_id} 的数据目录为空: {input_dir}")
        return False
    
    cmd = [
        sys.executable, 'process_user_data.py',
        '--input_dir', input_dir,
        '--output_dir', output_dir,
        '--time_col', 'auto',
        '--time_unit', 'auto',
        '--write_aligned_per_action', 'true'
    ]
    
    logger.info(f"处理用户 {user_id}: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5分钟超时
        )
        
        if result.returncode != 0:
            logger.error(f"用户 {user_id} 处理失败:\n{result.stderr}")
            if not continue_on_error:
                raise RuntimeError(f"用户 {user_id} 处理失败")
            return False
        
        # 检查输出文件是否存在
        output_file = os.path.join(output_dir, 'merged_aligned_labeled.csv')
        if not os.path.exists(output_file):
            logger.error(f"用户 {user_id} 未生成输出文件: {output_file}")
            return False
        
        logger.info(f"用户 {user_id} 处理成功")
        return True
        
    except subprocess.TimeoutExpired:
        logger.error(f"用户 {user_id} 处理超时")
        return False
    except Exception as e:
        logger.error(f"用户 {user_id} 处理异常: {e}")
        return False


def normalize_user_data(user_id: int, output_root: str, norm_method: str = 'zscore',
                        save_stats: bool = True, encode_labels_flag: bool = True) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    """
    读取用户的对齐数据，进行归一化处理，并可选地编码标签为1-8
    
    Returns:
        (normalized_df, norm_params) or (None, None) if failed
    """
    input_file = os.path.join(output_root, f'proband{user_id}', 'processed', 'merged_aligned_labeled.csv')
    
    if not os.path.exists(input_file):
        logger.error(f"用户 {user_id} 的对齐数据不存在: {input_file}")
        return None, None
    
    # 读取数据
    df = pd.read_csv(input_file)
    logger.info(f"用户 {user_id}: 读取 {len(df)} 行, {len(df.columns)} 列")
    
    # 验证结构：首列 time_ts，末列 label
    if df.columns[0] != 'time_ts':
        logger.warning(f"用户 {user_id}: 首列不是 time_ts，而是 {df.columns[0]}")
    if df.columns[-1] != 'label':
        logger.warning(f"用户 {user_id}: 末列不是 label，而是 {df.columns[-1]}")
    
    # 去掉时间列
    if 'time_ts' in df.columns:
        df = df.drop(columns=['time_ts'])
    
    # 识别特征列（以 acc_ 或 gyro_ 开头的列）
    feature_cols = [c for c in df.columns if c.startswith('acc_') or c.startswith('gyro_')]
    feature_cols = sorted(feature_cols)  # 稳定排序
    
    logger.info(f"用户 {user_id}: {len(feature_cols)} 个特征列")
    
    # 归一化
    norm_params = {
        'method': norm_method,
        'user_id': user_id,
        'columns': {}
    }
    
    for col in feature_cols:
        values = df[col].values.astype(np.float64)
        
        # 处理 NaN 值：先记录，归一化后填充为 0
        nan_mask = np.isnan(values)
        valid_values = values[~nan_mask]
        
        if len(valid_values) == 0:
            # 整列都是 NaN
            df[col] = 0.0
            norm_params['columns'][col] = {'mu': 0.0, 'sigma': 0.0} if norm_method == 'zscore' else {'min': 0.0, 'max': 0.0}
            continue
        
        if norm_method == 'zscore':
            mu = np.mean(valid_values)
            sigma = np.std(valid_values, ddof=0)  # 总体标准差
            
            if sigma == 0:
                normalized = np.zeros_like(values)
            else:
                normalized = (values - mu) / sigma
            
            # 将 NaN 填充为 0
            normalized = np.where(nan_mask, 0.0, normalized)
            df[col] = normalized
            
            norm_params['columns'][col] = {'mu': float(mu), 'sigma': float(sigma)}
            
        elif norm_method == 'minmax':
            min_val = np.min(valid_values)
            max_val = np.max(valid_values)
            
            if max_val == min_val:
                normalized = np.zeros_like(values)
            else:
                normalized = (values - min_val) / (max_val - min_val)
            
            # 将 NaN 填充为 0
            normalized = np.where(nan_mask, 0.0, normalized)
            df[col] = normalized
            
            norm_params['columns'][col] = {'min': float(min_val), 'max': float(max_val)}
    
    # 保存归一化后的数据
    output_dir = os.path.join(output_root, f'proband{user_id}', 'processed')
    
    # 确保列顺序：特征列 + label
    final_cols = feature_cols + ['label']
    df = df[final_cols]
    
    # 编码标签为 1-8
    if encode_labels_flag:
        original_labels = df['label'].unique().tolist()
        df = encode_labels(df)
        logger.info(f"用户 {user_id}: 标签已编码为 1-8")
    
    normalized_file = os.path.join(output_dir, 'merged_normalized.csv')
    df.to_csv(normalized_file, index=False, encoding='utf-8')
    logger.info(f"用户 {user_id}: 保存归一化数据到 {normalized_file}")
    
    # 保存归一化参数
    if save_stats:
        params_file = os.path.join(output_dir, 'norm_params.json')
        with open(params_file, 'w', encoding='utf-8') as f:
            json.dump(norm_params, f, indent=2, ensure_ascii=False)
        logger.info(f"用户 {user_id}: 保存归一化参数到 {params_file}")
    
    return df, norm_params


def align_columns(dfs: List[pd.DataFrame], fill_value: float = 0.0) -> List[pd.DataFrame]:
    """
    对齐所有用户的列，确保列集合一致
    
    Returns:
        列对齐后的 DataFrame 列表
    """
    # 收集所有特征列
    all_feature_cols = set()
    for df in dfs:
        feature_cols = [c for c in df.columns if c.startswith('acc_') or c.startswith('gyro_')]
        all_feature_cols.update(feature_cols)
    
    # 稳定排序
    all_feature_cols = sorted(all_feature_cols)
    final_cols = all_feature_cols + ['label']
    
    logger.info(f"合并后共 {len(all_feature_cols)} 个特征列")
    
    # 对齐每个 DataFrame
    aligned_dfs = []
    for df in dfs:
        for col in all_feature_cols:
            if col not in df.columns:
                df[col] = fill_value
        
        # 确保列顺序一致
        df = df[final_cols]
        aligned_dfs.append(df)
    
    return aligned_dfs


def main():
    parser = argparse.ArgumentParser(description='处理所有用户数据并合并')
    parser.add_argument('--users', type=str, default='1-15',
                        help='用户编号列表，支持离散区间 (如 "1,3-5,7-15")')
    parser.add_argument('--input_root', type=str, default='./data',
                        help='输入数据根目录')
    parser.add_argument('--output_root', type=str, default='./data',
                        help='输出根目录')
    parser.add_argument('--norm_method', type=str, default='zscore',
                        choices=['zscore', 'minmax'],
                        help='归一化方法')
    parser.add_argument('--fill_missing', type=str, default='0',
                        help='缺失列填充值（0 或 nan）')
    parser.add_argument('--final_output', type=str, default='./merged_users_1_15_normalized.csv',
                        help='最终合并文件路径')
    parser.add_argument('--save_user_stats', type=str, default='true',
                        help='是否保存每用户归一化参数')
    parser.add_argument('--continue_on_error', action='store_true',
                        help='遇到错误时是否继续处理其他用户')
    parser.add_argument('--skip_processing', action='store_true',
                        help='跳过 process_user_data.py 处理步骤（假设已处理）')
    parser.add_argument('--require_all_labels', action='store_true',
                        help='要求用户必须有全部8个类别，否则舍弃')
    parser.add_argument('--split_mode', action='store_true',
                        help='启用分割模式：选择4个有8类的用户，分为训练/验证/测试集')
    parser.add_argument('--train_users', type=str, default='',
                        help='用于训练集的用户编号（如 "1,3"）')
    parser.add_argument('--val_users', type=str, default='',
                        help='用于验证集的用户编号（如 "5"）')
    parser.add_argument('--test_users', type=str, default='',
                        help='用于测试集的用户编号（如 "7"）')
    parser.add_argument('--train_output', type=str, default='./training_data.csv',
                        help='训练集输出路径')
    parser.add_argument('--val_output', type=str, default='./validation_data.csv',
                        help='验证集输出路径')
    parser.add_argument('--test_output', type=str, default='./testing_data.csv',
                        help='测试集输出路径')
    parser.add_argument('--encode_labels', type=str, default='true',
                        help='是否将标签编码为 1-8 数字')
    
    args = parser.parse_args()
    
    # 解析参数
    user_ids = parse_user_range(args.users)
    fill_value = np.nan if args.fill_missing.lower() == 'nan' else 0.0
    save_stats = args.save_user_stats.lower() == 'true'
    encode_labels_flag = args.encode_labels.lower() == 'true'
    
    logger.info("=" * 60)
    logger.info("开始处理所有用户数据")
    logger.info(f"用户列表: {user_ids}")
    logger.info(f"输入根目录: {args.input_root}")
    logger.info(f"输出根目录: {args.output_root}")
    logger.info(f"归一化方法: {args.norm_method}")
    logger.info(f"缺失列填充: {args.fill_missing}")
    logger.info(f"要求8类: {args.require_all_labels}")
    logger.info(f"分割模式: {args.split_mode}")
    logger.info(f"标签编码: {encode_labels_flag}")
    logger.info(f"标签映射: {LABEL_ENCODING}")
    logger.info("=" * 60)
    
    # Step 1: 处理每个用户的原始数据
    successful_users = []
    
    if not args.skip_processing:
        for user_id in user_ids:
            logger.info(f"\n{'='*40}")
            logger.info(f"Step 1: 处理用户 {user_id} 的原始数据")
            logger.info(f"{'='*40}")
            
            success = run_process_user_data(
                user_id, 
                args.input_root, 
                args.output_root,
                args.continue_on_error
            )
            
            if success:
                successful_users.append(user_id)
    else:
        logger.info("跳过原始数据处理步骤，直接使用已有的 merged_aligned_labeled.csv")
        for user_id in user_ids:
            input_file = os.path.join(args.output_root, f'proband{user_id}', 'processed', 'merged_aligned_labeled.csv')
            if os.path.exists(input_file):
                successful_users.append(user_id)
            else:
                logger.warning(f"用户 {user_id} 的对齐数据不存在")
    
    if not successful_users:
        logger.error("没有用户数据处理成功，退出")
        return 1
    
    logger.info(f"\n成功处理的用户: {successful_users}")
    
    # Step 1.5: 检查用户是否有8个类别
    if args.require_all_labels or args.split_mode:
        logger.info(f"\n{'='*40}")
        logger.info("Step 1.5: 检查用户是否有全部8个类别")
        logger.info(f"{'='*40}")
        
        users_with_all_labels = []
        users_without_all_labels = []
        
        for user_id in successful_users:
            has_all, existing = check_user_has_all_labels(user_id, args.output_root)
            if has_all:
                users_with_all_labels.append(user_id)
            else:
                users_without_all_labels.append((user_id, existing))
        
        logger.info(f"\n有全部8类的用户: {users_with_all_labels}")
        if users_without_all_labels:
            logger.info(f"缺少类别的用户:")
            for uid, labels in users_without_all_labels:
                missing = [l for l in STANDARD_LABELS if l not in labels]
                logger.info(f"  - 用户 {uid}: 缺少 {missing}")
        
        if args.require_all_labels:
            successful_users = users_with_all_labels
            logger.info(f"\n仅保留有8类的用户: {successful_users}")
        
        if not successful_users:
            logger.error("没有用户有全部8个类别，退出")
            return 1
    
    # 分割模式
    if args.split_mode:
        logger.info(f"\n{'='*40}")
        logger.info("分割模式: 准备训练/验证/测试集")
        logger.info(f"{'='*40}")
        
        # 解析用户分配
        train_ids = parse_user_range(args.train_users) if args.train_users else []
        val_ids = parse_user_range(args.val_users) if args.val_users else []
        test_ids = parse_user_range(args.test_users) if args.test_users else []
        
        # 验证用户都有8类
        all_specified = train_ids + val_ids + test_ids
        for uid in all_specified:
            if uid not in users_with_all_labels:
                logger.error(f"用户 {uid} 不在有8类的用户列表中: {users_with_all_labels}")
                return 1
        
        logger.info(f"训练集用户: {train_ids}")
        logger.info(f"验证集用户: {val_ids}")
        logger.info(f"测试集用户: {test_ids}")
        
        # 处理每个数据集
        datasets = [
            ('训练集', train_ids, args.train_output),
            ('验证集', val_ids, args.val_output),
            ('测试集', test_ids, args.test_output)
        ]
        
        for dataset_name, dataset_users, output_path in datasets:
            if not dataset_users:
                logger.info(f"\n跳过{dataset_name}（无指定用户）")
                continue
            
            logger.info(f"\n{'='*40}")
            logger.info(f"处理{dataset_name}: 用户 {dataset_users}")
            logger.info(f"{'='*40}")
            
            # 归一化每个用户
            user_dfs = []
            for user_id in dataset_users:
                df, params = normalize_user_data(
                    user_id, 
                    args.output_root, 
                    args.norm_method,
                    save_stats,
                    encode_labels_flag
                )
                if df is not None:
                    user_dfs.append((user_id, df))
            
            if not user_dfs:
                logger.warning(f"{dataset_name}没有有效数据")
                continue
            
            # 对齐列
            dfs_only = [df for _, df in user_dfs]
            aligned_dfs = align_columns(dfs_only, fill_value)
            user_dfs = [(user_id, aligned_df) for (user_id, _), aligned_df in zip(user_dfs, aligned_dfs)]
            
            # 合并
            user_dfs.sort(key=lambda x: x[0])
            final_dfs = [df for _, df in user_dfs]
            final_df = pd.concat(final_dfs, ignore_index=True)
            
            # 保存
            final_df.to_csv(output_path, index=False, encoding='utf-8')
            logger.info(f"{dataset_name}已保存: {output_path}")
            logger.info(f"  - 行数: {len(final_df)}")
            logger.info(f"  - 列数: {len(final_df.columns)}")
            logger.info(f"  - 标签: {sorted(final_df['label'].unique().tolist())}")
            
            # 保存统计信息
            stats_file = output_path.replace('.csv', '_stats.json')
            stats = {
                'generated_at': datetime.now().isoformat(),
                'dataset_type': dataset_name,
                'norm_method': args.norm_method,
                'users': dataset_users,
                'user_rows': {str(uid): len(df) for uid, df in user_dfs},
                'total_rows': len(final_df),
                'columns': list(final_df.columns),
                'labels': sorted(final_df['label'].unique().tolist()),
                'label_encoding': LABEL_ENCODING
            }
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n{'='*60}")
        logger.info("分割模式处理完成!")
        logger.info("=" * 60)
        return 0
    
    # 普通模式（非分割）
    # Step 2: 对每个用户进行归一化
    logger.info(f"\n{'='*40}")
    logger.info("Step 2: 归一化每个用户的数据")
    logger.info(f"{'='*40}")
    
    user_dfs = []
    user_stats = {}
    
    for user_id in successful_users:
        df, params = normalize_user_data(
            user_id, 
            args.output_root, 
            args.norm_method,
            save_stats,
            encode_labels_flag
        )
        
        if df is not None:
            user_dfs.append((user_id, df))
            user_stats[user_id] = {
                'rows': len(df),
                'columns': list(df.columns)
            }
    
    if not user_dfs:
        logger.error("没有用户数据归一化成功，退出")
        return 1
    
    # Step 3: 对齐所有用户的列
    logger.info(f"\n{'='*40}")
    logger.info("Step 3: 对齐列并合并")
    logger.info(f"{'='*40}")
    
    dfs_only = [df for _, df in user_dfs]
    aligned_dfs = align_columns(dfs_only, fill_value)
    
    # 更新 user_dfs 为对齐后的版本
    user_dfs = [(user_id, aligned_df) for (user_id, _), aligned_df in zip(user_dfs, aligned_dfs)]
    
    # Step 4: 按用户编号顺序拼接
    logger.info(f"\n{'='*40}")
    logger.info("Step 4: 按用户顺序拼接数据")
    logger.info(f"{'='*40}")
    
    # 按用户 ID 排序
    user_dfs.sort(key=lambda x: x[0])
    
    final_dfs = []
    for user_id, df in user_dfs:
        logger.info(f"用户 {user_id}: {len(df)} 行")
        final_dfs.append(df)
    
    final_df = pd.concat(final_dfs, ignore_index=True)
    
    logger.info(f"\n合并后总行数: {len(final_df)}")
    logger.info(f"合并后总列数: {len(final_df.columns)}")
    
    # Step 5: 保存最终数据
    logger.info(f"\n{'='*40}")
    logger.info("Step 5: 保存最终数据")
    logger.info(f"{'='*40}")
    
    final_df.to_csv(args.final_output, index=False, encoding='utf-8')
    logger.info(f"最终数据已保存: {args.final_output}")
    
    # 保存统计信息
    stats_file = args.final_output.replace('.csv', '_stats.json')
    stats = {
        'generated_at': datetime.now().isoformat(),
        'norm_method': args.norm_method,
        'users': [uid for uid, _ in user_dfs],
        'user_rows': {str(uid): len(df) for uid, df in user_dfs},
        'total_rows': len(final_df),
        'columns': list(final_df.columns),
        'labels': sorted(final_df['label'].unique().tolist()),
        'label_encoding': LABEL_ENCODING if encode_labels_flag else None
    }
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info(f"统计信息已保存: {stats_file}")
    
    # Step 6: 验证
    logger.info(f"\n{'='*40}")
    logger.info("Step 6: 验证输出")
    logger.info(f"{'='*40}")
    
    issues = []
    
    # 检查文件存在
    if not os.path.exists(args.final_output):
        issues.append("最终输出文件不存在")
    
    # 检查 NaN
    if args.fill_missing != 'nan':
        nan_cols = final_df.columns[final_df.isna().any()].tolist()
        if nan_cols:
            issues.append(f"存在 NaN 的列: {nan_cols}")
    
    # 检查 label 列
    if 'label' not in final_df.columns:
        issues.append("缺少 label 列")
    elif final_df.columns[-1] != 'label':
        issues.append(f"label 不是最后一列，实际位置: {list(final_df.columns).index('label')}")
    
    # 检查总行数
    expected_rows = sum(len(df) for _, df in user_dfs)
    if len(final_df) != expected_rows:
        issues.append(f"总行数不匹配: 期望 {expected_rows}, 实际 {len(final_df)}")
    
    if issues:
        for issue in issues:
            logger.warning(f"验证问题: {issue}")
    else:
        logger.info("所有验证通过 ✓")
    
    # 打印摘要
    logger.info(f"\n{'='*60}")
    logger.info("处理完成摘要:")
    logger.info(f"  - 处理用户数: {len(user_dfs)}")
    logger.info(f"  - 总行数: {len(final_df)}")
    logger.info(f"  - 特征列数: {len(final_df.columns) - 1}")
    logger.info(f"  - 动作标签: {sorted(final_df['label'].unique())}")
    logger.info(f"  - 输出文件: {args.final_output}")
    logger.info("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
