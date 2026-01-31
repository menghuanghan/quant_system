"""
过滤器调度便捷函数

提供简单的函数式接口，方便快速调用过滤器
"""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from .announcement_filter import (
    AnnouncementFilter,
    FilterConfig,
    FilterResult,
)

logger = logging.getLogger(__name__)


def filter_month(
    year: int,
    month: int,
    use_gpu: bool = True,
    dry_run: bool = False,
    raw_data_dir: str = "data/raw/unstructured",
) -> FilterResult:
    """
    过滤指定年月的公告数据
    
    Args:
        year: 年份
        month: 月份
        use_gpu: 是否使用GPU加速
        dry_run: 如果为True，只统计不写入
        raw_data_dir: 原始数据目录
        
    Returns:
        FilterResult: 过滤结果
    """
    config = FilterConfig(
        use_gpu=use_gpu,
        raw_data_dir=raw_data_dir,
    )
    
    filter_instance = AnnouncementFilter(config=config)
    return filter_instance.filter_month(year, month, dry_run=dry_run)


def filter_year(
    year: int,
    start_month: int = 1,
    end_month: int = 12,
    use_gpu: bool = True,
    dry_run: bool = False,
    raw_data_dir: str = "data/raw/unstructured",
) -> List[FilterResult]:
    """
    过滤指定年份的公告数据
    
    Args:
        year: 年份
        start_month: 起始月份
        end_month: 结束月份
        use_gpu: 是否使用GPU加速
        dry_run: 如果为True，只统计不写入
        raw_data_dir: 原始数据目录
        
    Returns:
        List[FilterResult]: 各月份的过滤结果
    """
    config = FilterConfig(
        use_gpu=use_gpu,
        raw_data_dir=raw_data_dir,
    )
    
    filter_instance = AnnouncementFilter(config=config)
    return filter_instance.filter_year(
        year, 
        start_month=start_month, 
        end_month=end_month, 
        dry_run=dry_run
    )


def filter_all(
    years: Optional[List[int]] = None,
    use_gpu: bool = True,
    dry_run: bool = False,
    raw_data_dir: str = "data/raw/unstructured",
) -> Dict[int, List[FilterResult]]:
    """
    过滤所有年份的公告数据
    
    Args:
        years: 要过滤的年份列表，如果为None则自动发现
        use_gpu: 是否使用GPU加速
        dry_run: 如果为True，只统计不写入
        raw_data_dir: 原始数据目录
        
    Returns:
        Dict[int, List[FilterResult]]: 按年份分组的过滤结果
    """
    config = FilterConfig(
        use_gpu=use_gpu,
        raw_data_dir=raw_data_dir,
    )
    
    filter_instance = AnnouncementFilter(config=config)
    return filter_instance.filter_all(years=years, dry_run=dry_run)


def get_filter_statistics(
    years: Optional[List[int]] = None,
    raw_data_dir: str = "data/raw/unstructured",
) -> Dict[str, Any]:
    """
    获取过滤统计信息（不修改数据）
    
    Args:
        years: 要统计的年份列表，如果为None则自动发现
        raw_data_dir: 原始数据目录
        
    Returns:
        统计信息字典
    """
    config = FilterConfig(
        use_gpu=False,  # 统计不需要GPU
        raw_data_dir=raw_data_dir,
    )
    
    filter_instance = AnnouncementFilter(config=config)
    
    # 自动发现年份
    if years is None:
        announcements_dir = Path(raw_data_dir) / "announcements"
        years = []
        for year_dir in announcements_dir.glob("*"):
            if year_dir.is_dir() and year_dir.name.isdigit():
                years.append(int(year_dir.name))
        years.sort()
    
    total_original = 0
    total_final = 0
    total_event_filtered = 0
    total_title_filtered = 0
    year_stats = {}
    
    for year in years:
        year_original = 0
        year_final = 0
        year_event_filtered = 0
        year_title_filtered = 0
        month_stats = []
        
        for month in range(1, 13):
            stats = filter_instance.get_statistics(year, month)
            if stats['original_count'] > 0:
                year_original += stats['original_count']
                year_final += stats['final_count']
                year_event_filtered += stats['event_filtered']
                year_title_filtered += stats['title_filtered']
                month_stats.append(stats)
        
        if year_original > 0:
            year_stats[year] = {
                'original_count': year_original,
                'final_count': year_final,
                'event_filtered': year_event_filtered,
                'title_filtered': year_title_filtered,
                'filter_rate': (year_original - year_final) / year_original,
                'months': month_stats,
            }
            
            total_original += year_original
            total_final += year_final
            total_event_filtered += year_event_filtered
            total_title_filtered += year_title_filtered
    
    return {
        'total': {
            'original_count': total_original,
            'final_count': total_final,
            'event_filtered': total_event_filtered,
            'title_filtered': total_title_filtered,
            'total_filtered': total_original - total_final,
            'filter_rate': (total_original - total_final) / total_original if total_original > 0 else 0,
            'estimated_processing_hours': total_final / 3600,  # 假设1秒处理1条
            'estimated_processing_days': total_final / 3600 / 24,
        },
        'by_year': year_stats,
    }


def print_filter_statistics(stats: Dict[str, Any]):
    """
    打印过滤统计信息
    
    Args:
        stats: get_filter_statistics返回的统计信息
    """
    print("=" * 80)
    print("公告数据过滤统计")
    print("=" * 80)
    
    total = stats['total']
    print(f"\n总计:")
    print(f"  原始记录数: {total['original_count']:,}")
    print(f"  事件过滤: {total['event_filtered']:,}")
    print(f"  标题过滤: {total['title_filtered']:,}")
    print(f"  总过滤数: {total['total_filtered']:,}")
    print(f"  最终记录数: {total['final_count']:,}")
    print(f"  过滤率: {total['filter_rate']:.1%}")
    print(f"  预估处理时间: {total['estimated_processing_hours']:.1f} 小时 "
          f"({total['estimated_processing_days']:.1f} 天)")
    
    print(f"\n按年份:")
    for year, year_stats in stats['by_year'].items():
        print(f"\n  {year}年:")
        print(f"    原始: {year_stats['original_count']:,}")
        print(f"    最终: {year_stats['final_count']:,}")
        print(f"    过滤率: {year_stats['filter_rate']:.1%}")
    
    print("=" * 80)
