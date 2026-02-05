#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
日期格式统一脚本

统一 data/raw/structured 下所有 Parquet 文件中的日期字段格式为 "YYYY-MM-DD"。

Usage:
    # 处理全部数据
    python scripts/standardize_date_format.py
    
    # 只处理指定数据域
    python scripts/standardize_date_format.py --domains market_data fundamental
    
    # 只处理指定数据域的指定数据
    python scripts/standardize_date_format.py --domains derivatives --tasks fut_daily repo_daily
    
    # 预览模式（不实际修改文件）
    python scripts/standardize_date_format.py --dry-run
    
    # 显示详细日志
    python scripts/standardize_date_format.py --verbose
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Set
import re

import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 常见的日期列名
DATE_COLUMN_PATTERNS = [
    'trade_date', 'ann_date', 'end_date', 'start_date', 'list_date',
    'delist_date', 'found_date', 'due_date', 'issue_date', 'nav_date',
    'publish_date', 'report_date', 'cal_date', 'pretrade_date',
    'suspend_date', 'resume_date', 'holder_date', 'ex_date', 'pay_date',
    'record_date', 'div_listdate', 'imp_ann_date', 'base_date',
    'f_ann_date', 's_ann_date', 'actual_ann_date', 'update_flag',
]

# 正则匹配可能的日期格式
DATE_REGEX_PATTERNS = [
    (r'^\d{8}$', '%Y%m%d'),           # YYYYMMDD
    (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),  # YYYY-MM-DD (already correct)
    (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),  # YYYY/MM/DD
    (r'^\d{2}-\d{2}-\d{4}$', '%d-%m-%Y'),  # DD-MM-YYYY
    (r'^\d{2}/\d{2}/\d{4}$', '%d/%m/%Y'),  # DD/MM/YYYY
]

TARGET_FORMAT = '%Y-%m-%d'


def detect_date_columns(df: pd.DataFrame) -> List[str]:
    """检测 DataFrame 中的日期列"""
    date_cols = []
    for col in df.columns:
        col_lower = col.lower()
        # 匹配已知的日期列名
        if col_lower in DATE_COLUMN_PATTERNS or col_lower.endswith('_date'):
            date_cols.append(col)
            continue
        # 尝试从数据内容推断
        if df[col].dtype == 'object' and len(df) > 0:
            sample = df[col].dropna().head(5).astype(str)
            for val in sample:
                for pattern, _ in DATE_REGEX_PATTERNS:
                    if re.match(pattern, str(val)):
                        date_cols.append(col)
                        break
                else:
                    continue
                break
    return list(set(date_cols))


def convert_date_to_standard(value) -> Optional[str]:
    """将单个日期值转换为标准格式 YYYY-MM-DD"""
    if pd.isna(value):
        return None
    
    val_str = str(value).strip()
    if not val_str or val_str.lower() in ('none', 'nan', 'nat', ''):
        return None
    
    # 已经是目标格式
    if re.match(r'^\d{4}-\d{2}-\d{2}$', val_str):
        return val_str
    
    # 尝试各种格式解析
    for pattern, fmt in DATE_REGEX_PATTERNS:
        if re.match(pattern, val_str):
            try:
                dt = datetime.strptime(val_str, fmt)
                return dt.strftime(TARGET_FORMAT)
            except ValueError:
                continue
    
    # 尝试 pandas 自动解析
    try:
        dt = pd.to_datetime(val_str)
        if pd.notna(dt):
            return dt.strftime(TARGET_FORMAT)
    except:
        pass
    
    return val_str  # 无法转换，保持原样


def standardize_date_column(series: pd.Series) -> pd.Series:
    """标准化整列日期数据"""
    return series.apply(convert_date_to_standard)


def process_parquet_file(
    file_path: Path,
    dry_run: bool = False,
    verbose: bool = False
) -> dict:
    """
    处理单个 Parquet 文件
    
    Returns:
        dict: 处理结果统计
    """
    result = {
        'file': str(file_path),
        'status': 'skipped',
        'date_columns': [],
        'rows': 0,
        'converted': 0,
        'error': None
    }
    
    try:
        df = pd.read_parquet(file_path)
        result['rows'] = len(df)
        
        if df.empty:
            result['status'] = 'empty'
            return result
        
        # 检测日期列
        date_cols = detect_date_columns(df)
        result['date_columns'] = date_cols
        
        if not date_cols:
            result['status'] = 'no_date_cols'
            return result
        
        # 转换日期列
        modified = False
        for col in date_cols:
            if col not in df.columns:
                continue
            
            original = df[col].copy()
            df[col] = standardize_date_column(df[col])
            
            # 检查是否有变化
            changes = (original.astype(str) != df[col].astype(str)).sum()
            if changes > 0:
                modified = True
                result['converted'] += changes
                if verbose:
                    logger.info(f"  列 '{col}': 转换了 {changes} 个值")
        
        if modified:
            if not dry_run:
                df.to_parquet(file_path, index=False, compression='snappy')
                result['status'] = 'converted'
            else:
                result['status'] = 'would_convert'
        else:
            result['status'] = 'already_standard'
            
    except Exception as e:
        result['status'] = 'error'
        result['error'] = str(e)
        logger.error(f"处理文件失败 {file_path}: {e}")
    
    return result


def get_all_domains(base_dir: Path) -> List[str]:
    """获取所有数据域目录"""
    domains = []
    for item in base_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            domains.append(item.name)
    return sorted(domains)


def get_tasks_in_domain(domain_dir: Path) -> List[str]:
    """获取数据域下的所有任务（子目录或直接的parquet文件）"""
    tasks = []
    for item in domain_dir.iterdir():
        if item.is_dir():
            tasks.append(item.name)
        elif item.suffix == '.parquet':
            tasks.append(item.stem)
    return sorted(set(tasks))


def collect_parquet_files(
    base_dir: Path,
    domains: Optional[List[str]] = None,
    tasks: Optional[List[str]] = None
) -> List[Path]:
    """收集需要处理的 Parquet 文件"""
    files = []
    
    # 确定要处理的域
    all_domains = get_all_domains(base_dir)
    target_domains = domains if domains else all_domains
    
    for domain in target_domains:
        domain_path = base_dir / domain
        if not domain_path.exists():
            logger.warning(f"数据域不存在: {domain}")
            continue
        
        # 如果指定了任务，只处理这些任务
        if tasks:
            for task in tasks:
                task_path = domain_path / task
                if task_path.is_dir():
                    # 任务是一个目录，包含多个 parquet 文件
                    files.extend(task_path.glob('*.parquet'))
                elif (domain_path / f"{task}.parquet").exists():
                    # 任务是单个 parquet 文件
                    files.append(domain_path / f"{task}.parquet")
                else:
                    logger.warning(f"任务不存在: {domain}/{task}")
        else:
            # 处理域下所有文件
            files.extend(domain_path.rglob('*.parquet'))
    
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description='统一 Parquet 文件中的日期格式为 YYYY-MM-DD',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='data/raw/structured',
        help='数据根目录 (默认: data/raw/structured)'
    )
    parser.add_argument(
        '--domains', '-d',
        nargs='+',
        help='指定要处理的数据域 (如: market_data fundamental derivatives)'
    )
    parser.add_argument(
        '--tasks', '-t',
        nargs='+',
        help='指定要处理的任务 (如: fut_daily repo_daily stock_daily)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，不实际修改文件'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细日志'
    )
    parser.add_argument(
        '--list-domains',
        action='store_true',
        help='列出所有可用的数据域'
    )
    parser.add_argument(
        '--list-tasks',
        type=str,
        metavar='DOMAIN',
        help='列出指定数据域下的所有任务'
    )
    
    args = parser.parse_args()
    
    base_dir = PROJECT_ROOT / args.base_dir
    
    if not base_dir.exists():
        logger.error(f"数据目录不存在: {base_dir}")
        sys.exit(1)
    
    # 列出数据域
    if args.list_domains:
        domains = get_all_domains(base_dir)
        print("可用的数据域:")
        for d in domains:
            print(f"  - {d}")
        sys.exit(0)
    
    # 列出任务
    if args.list_tasks:
        domain_dir = base_dir / args.list_tasks
        if not domain_dir.exists():
            logger.error(f"数据域不存在: {args.list_tasks}")
            sys.exit(1)
        tasks = get_tasks_in_domain(domain_dir)
        print(f"数据域 '{args.list_tasks}' 下的任务:")
        for t in tasks:
            print(f"  - {t}")
        sys.exit(0)
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # 收集文件
    logger.info("="*60)
    logger.info("日期格式统一脚本")
    logger.info("="*60)
    logger.info(f"数据目录: {base_dir}")
    logger.info(f"目标格式: YYYY-MM-DD")
    if args.domains:
        logger.info(f"指定数据域: {args.domains}")
    if args.tasks:
        logger.info(f"指定任务: {args.tasks}")
    if args.dry_run:
        logger.info("*** 预览模式 - 不会修改任何文件 ***")
    logger.info("-"*60)
    
    files = collect_parquet_files(base_dir, args.domains, args.tasks)
    
    if not files:
        logger.warning("未找到任何 Parquet 文件")
        sys.exit(0)
    
    logger.info(f"共找到 {len(files)} 个 Parquet 文件")
    
    # 处理文件
    stats = {
        'total': len(files),
        'converted': 0,
        'already_standard': 0,
        'no_date_cols': 0,
        'empty': 0,
        'error': 0,
        'total_rows': 0,
        'total_converted_values': 0
    }
    
    for i, file_path in enumerate(files, 1):
        if args.verbose or (i % 100 == 0):
            logger.info(f"处理进度: {i}/{len(files)} - {file_path.relative_to(base_dir)}")
        
        result = process_parquet_file(file_path, args.dry_run, args.verbose)
        
        stats['total_rows'] += result['rows']
        stats['total_converted_values'] += result['converted']
        
        if result['status'] in ('converted', 'would_convert'):
            stats['converted'] += 1
        elif result['status'] == 'already_standard':
            stats['already_standard'] += 1
        elif result['status'] == 'no_date_cols':
            stats['no_date_cols'] += 1
        elif result['status'] == 'empty':
            stats['empty'] += 1
        elif result['status'] == 'error':
            stats['error'] += 1
    
    # 输出统计
    logger.info("="*60)
    logger.info("处理完成")
    logger.info("="*60)
    logger.info(f"总文件数: {stats['total']}")
    logger.info(f"已转换/需转换: {stats['converted']}")
    logger.info(f"已是标准格式: {stats['already_standard']}")
    logger.info(f"无日期列: {stats['no_date_cols']}")
    logger.info(f"空文件: {stats['empty']}")
    logger.info(f"处理失败: {stats['error']}")
    logger.info(f"总行数: {stats['total_rows']:,}")
    logger.info(f"转换的值数: {stats['total_converted_values']:,}")
    
    if args.dry_run:
        logger.info("\n*** 这是预览模式，未进行实际修改 ***")
        logger.info("移除 --dry-run 参数以执行实际转换")


if __name__ == '__main__':
    main()
