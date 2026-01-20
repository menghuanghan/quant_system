"""
非结构化数据路径管理工具

提供统一的路径获取接口，避免路径硬编码
支持：分层（metadata/files）+ 分源（source）+ 分区（year/month）
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Optional, Union


# 全局根路径
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "unstructured"


class UnstructuredDataPaths:
    """
    非结构化数据路径管理器
    
    设计原则：
    1. 存算分离：元数据（metadata）与原始文件（files/pdf）分离
    2. 高效检索：按时间（year/month）或实体（stock_code）分区
    3. 分源管理：不同数据源物理隔离
    """
    
    # ==================== 1. 公告模块 ====================
    
    @staticmethod
    def get_announcement_metadata_path(
        source: str = 'cninfo',
        year: Optional[str] = None,
        month: Optional[str] = None,
        format: str = 'parquet'
    ) -> Path:
        """
        获取公告元数据路径
        
        Args:
            source: 数据源 (cninfo/eastmoney)
            year: 年份 (可选)
            month: 月份 (可选，需要year)
            format: 文件格式 (parquet/jsonl/csv)
            
        Returns:
            data/raw/unstructured/announcements/metadata/{source}/{year}{month}.{format}
            
        Examples:
            >>> get_announcement_metadata_path('cninfo', '2025', '01')
            data/raw/unstructured/announcements/metadata/cninfo/202501.parquet
        """
        base = DATA_ROOT / "announcements" / "metadata" / source
        base.mkdir(parents=True, exist_ok=True)
        
        if year and month:
            filename = f"{year}{month}.{format}"
        elif year:
            filename = f"{year}.{format}"
        else:
            filename = f"all.{format}"
        
        return base / filename
    
    @staticmethod
    def get_announcement_file_path(
        ts_code: str,
        ann_date: str,
        filename: str,
        use_stock_partition: bool = False
    ) -> Path:
        """
        获取公告PDF存储路径
        
        Args:
            ts_code: 股票代码 (000001.SZ)
            ann_date: 公告日期 (YYYYMMDD 或 YYYY-MM-DD)
            filename: 文件名（含扩展名）
            use_stock_partition: 是否按股票代码分区（避免单目录文件过多）
            
        Returns:
            data/raw/unstructured/announcements/files/{year}/{ts_code}/{filename}
            或
            data/raw/unstructured/announcements/files/{year}/{filename}
            
        Examples:
            >>> get_announcement_file_path('000001.SZ', '20250115', '年度报告.pdf', True)
            data/raw/unstructured/announcements/files/2025/000001.SZ/年度报告.pdf
        """
        # 提取年份
        date_str = ann_date.replace('-', '')
        year = date_str[:4]
        
        if use_stock_partition:
            base = DATA_ROOT / "announcements" / "files" / year / ts_code
        else:
            base = DATA_ROOT / "announcements" / "files" / year
        
        base.mkdir(parents=True, exist_ok=True)
        return base / filename
    
    # ==================== 2. 新闻模块 ====================
    
    @staticmethod
    def get_news_path(
        source: str,
        date: str,
        format: str = 'jsonl'
    ) -> Path:
        """
        获取新闻存储路径（按天归档）
        
        Args:
            source: 数据源 (sina/eastmoney/cctv/stcn)
            date: 日期 (YYYYMMDD 或 YYYY-MM-DD)
            format: 文件格式 (jsonl/parquet)
            
        Returns:
            data/raw/unstructured/news/{source}/{year}/{YYYYMMDD}.{format}
            
        Examples:
            >>> get_news_path('sina', '20250115')
            data/raw/unstructured/news/sina/2025/20250115.jsonl
        """
        date_str = date.replace('-', '')
        year = date_str[:4]
        
        base = DATA_ROOT / "news" / source / year
        base.mkdir(parents=True, exist_ok=True)
        
        return base / f"{date_str}.{format}"
    
    # ==================== 3. 研报模块 ====================
    
    @staticmethod
    def get_report_metadata_path(
        year: Optional[str] = None,
        month: Optional[str] = None,
        format: str = 'parquet'
    ) -> Path:
        """
        获取研报元数据路径
        
        Args:
            year: 年份 (可选)
            month: 月份 (可选)
            format: 文件格式 (parquet/jsonl/csv)
            
        Returns:
            data/raw/unstructured/reports/metadata/{year}{month}.{format}
            
        Examples:
            >>> get_report_metadata_path('2025', '01')
            data/raw/unstructured/reports/metadata/202501.parquet
        """
        base = DATA_ROOT / "reports" / "metadata"
        base.mkdir(parents=True, exist_ok=True)
        
        if year and month:
            filename = f"{year}{month}.{format}"
        elif year:
            filename = f"{year}.{format}"
        else:
            filename = f"all.{format}"
        
        return base / filename
    
    @staticmethod
    def get_report_pdf_path(
        ts_code: str,
        report_date: str,
        org_name: str,
        rating: str = ''
    ) -> Path:
        """
        获取研报PDF存储路径
        
        Args:
            ts_code: 股票代码
            report_date: 研报日期 (YYYYMMDD)
            org_name: 机构名称 (如：中信证券)
            rating: 评级 (可选，如：买入)
            
        Returns:
            data/raw/unstructured/reports/pdf/{year}/{ts_code}_{date}_{org}_{rating}.pdf
            
        Examples:
            >>> get_report_pdf_path('600519.SH', '20250115', '中信证券', '买入')
            data/raw/unstructured/reports/pdf/2025/600519.SH_20250115_中信证券_买入.pdf
        """
        year = report_date[:4]
        base = DATA_ROOT / "reports" / "pdf" / year
        base.mkdir(parents=True, exist_ok=True)
        
        # 构造文件名（移除特殊字符）
        safe_org = org_name.replace('/', '_').replace('\\', '_')
        safe_rating = rating.replace('/', '_').replace('\\', '_')
        
        if rating:
            filename = f"{ts_code}_{report_date}_{safe_org}_{safe_rating}.pdf"
        else:
            filename = f"{ts_code}_{report_date}_{safe_org}.pdf"
        
        return base / filename
    
    # ==================== 4. 舆情模块 ====================
    
    @staticmethod
    def get_sentiment_path(
        source: str,
        date: str,
        format: str = 'parquet'
    ) -> Path:
        """
        获取舆情数据路径（按月归档）
        
        Args:
            source: 数据源 (xueqiu/guba/interaction)
            date: 日期 (YYYYMMDD 或 YYYY-MM-DD)
            format: 文件格式 (parquet推荐，因数据量大)
            
        Returns:
            data/raw/unstructured/sentiment/{source}/{year}/{YYYYMM}.{format}
            
        Examples:
            >>> get_sentiment_path('xueqiu', '20250115')
            data/raw/unstructured/sentiment/xueqiu/2025/202501.parquet
        """
        date_str = date.replace('-', '')
        year = date_str[:4]
        month = date_str[:6]
        
        base = DATA_ROOT / "sentiment" / source / year
        base.mkdir(parents=True, exist_ok=True)
        
        return base / f"{month}.{format}"
    
    # ==================== 5. 政策模块 ====================
    
    @staticmethod
    def get_policy_rules_path(
        agency: str,
        year: Optional[str] = None,
        format: str = 'jsonl'
    ) -> Path:
        """
        获取政策文本存储路径
        
        Args:
            agency: 机构名称 (csrc/miit/gov_council)
            year: 年份 (可选)
            format: 文件格式 (jsonl/parquet)
            
        Returns:
            data/raw/unstructured/policy/{agency}/rules/{year}.{format}
            
        Examples:
            >>> get_policy_rules_path('csrc', '2025')
            data/raw/unstructured/policy/csrc/rules/2025.jsonl
        """
        base = DATA_ROOT / "policy" / agency / "rules"
        base.mkdir(parents=True, exist_ok=True)
        
        if year:
            filename = f"{year}.{format}"
        else:
            filename = f"all.{format}"
        
        return base / filename
    
    @staticmethod
    def get_policy_file_path(
        agency: str,
        policy_id: str,
        filename: str
    ) -> Path:
        """
        获取政策附件存储路径
        
        Args:
            agency: 机构名称
            policy_id: 政策唯一ID
            filename: 附件文件名
            
        Returns:
            data/raw/unstructured/policy/{agency}/files/{policy_id}/{filename}
            
        Examples:
            >>> get_policy_file_path('csrc', 'csrc_2025_001', '附件1.pdf')
            data/raw/unstructured/policy/csrc/files/csrc_2025_001/附件1.pdf
        """
        base = DATA_ROOT / "policy" / agency / "files" / policy_id
        base.mkdir(parents=True, exist_ok=True)
        
        return base / filename
    
    # ==================== 6. 事件驱动模块 ====================
    
    # 事件类型对应的存储目录（与 base_event.py 保持一致）
    EVENT_TYPE_DIRS = {
        'merger': 'merger_acquisition',
        'penalty': 'penalty',
        'control_change': 'control_change',
        'contract': 'contract',
        'equity_change': 'equity_change',
        'litigation': 'litigation',
        'bankruptcy': 'bankruptcy',
        'suspension': 'suspension',
        'asset_restructure': 'asset_restructure',
        'share_repurchase': 'share_repurchase',
        'major_investment': 'major_investment',
        'other': 'other',
    }
    
    @staticmethod
    def get_event_meta_path(
        event_type: Optional[str] = None,
        year: Optional[str] = None,
        month: Optional[str] = None,
        format: str = 'parquet'
    ) -> Path:
        """
        获取事件元数据路径
        
        Args:
            event_type: 事件类型 (merger/penalty等)
            year: 年份 (可选)
            month: 月份 (可选)
            format: 文件格式
            
        Returns:
            data/raw/unstructured/events/meta/{event_type}/{year}{month}.{format}
            
        Examples:
            >>> get_event_meta_path('penalty', '2025', '01')
            data/raw/unstructured/events/meta/penalty/202501.parquet
        """
        if event_type:
            base = DATA_ROOT / "events" / "meta" / event_type
        else:
            base = DATA_ROOT / "events" / "meta"
        
        base.mkdir(parents=True, exist_ok=True)
        
        if year and month:
            filename = f"{year}{month}.{format}"
        elif year:
            filename = f"{year}.{format}"
        else:
            filename = f"all.{format}"
        
        return base / filename
    
    @staticmethod
    def get_event_pdf_path(
        event_type: str,
        ts_code: str,
        ann_date: str,
        filename: str
    ) -> Path:
        """
        获取事件PDF存储路径
        
        Args:
            event_type: 事件类型（必须是 EVENT_TYPE_DIRS 中的键）
            ts_code: 股票代码
            ann_date: 公告日期 (YYYYMMDD)
            filename: 文件名
            
        Returns:
            data/raw/unstructured/events/{event_dir}/{year}/{filename}
            
        Examples:
            >>> get_event_pdf_path('penalty', '000001.SZ', '20250115', '处罚决定书.pdf')
            data/raw/unstructured/events/penalty/2025/处罚决定书.pdf
        """
        # 获取事件类型对应的目录名
        event_dir = UnstructuredDataPaths.EVENT_TYPE_DIRS.get(event_type, 'other')
        
        # 提取年份
        year = ann_date[:4]
        
        base = DATA_ROOT / "events" / event_dir / year
        base.mkdir(parents=True, exist_ok=True)
        
        return base / filename
    
    # ==================== 工具方法 ====================
    
    @staticmethod
    def parse_date(date_str: str) -> tuple[str, str]:
        """
        解析日期字符串
        
        Args:
            date_str: YYYYMMDD 或 YYYY-MM-DD
            
        Returns:
            (year, month) 如 ('2025', '01')
        """
        clean_date = date_str.replace('-', '')
        return clean_date[:4], clean_date[4:6]
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """
        确保目录存在
        
        Args:
            path: 目录路径
            
        Returns:
            Path对象
        """
        path_obj = Path(path)
        path_obj.mkdir(parents=True, exist_ok=True)
        return path_obj
    
    @staticmethod
    def get_storage_summary() -> dict:
        """
        获取存储空间使用情况
        
        Returns:
            各模块文件数量和大小统计
        """
        summary = {}
        
        for module in ['announcements', 'news', 'reports', 'sentiment', 'policy', 'events']:
            module_path = DATA_ROOT / module
            if not module_path.exists():
                continue
            
            total_size = 0
            file_count = 0
            
            for file_path in module_path.rglob('*'):
                if file_path.is_file():
                    file_count += 1
                    total_size += file_path.stat().st_size
            
            summary[module] = {
                'file_count': file_count,
                'total_size_mb': round(total_size / 1024 / 1024, 2)
            }
        
        return summary


# 全局单例
_paths = UnstructuredDataPaths()


def get_unstructured_paths() -> UnstructuredDataPaths:
    """获取全局路径管理器单例"""
    return _paths


# 便捷函数（直接调用静态方法）
def get_announcement_metadata_path(*args, **kwargs) -> Path:
    """获取公告元数据路径"""
    return UnstructuredDataPaths.get_announcement_metadata_path(*args, **kwargs)


def get_announcement_file_path(*args, **kwargs) -> Path:
    """获取公告文件路径"""
    return UnstructuredDataPaths.get_announcement_file_path(*args, **kwargs)


def get_news_path(*args, **kwargs) -> Path:
    """获取新闻路径"""
    return UnstructuredDataPaths.get_news_path(*args, **kwargs)


def get_report_metadata_path(*args, **kwargs) -> Path:
    """获取研报元数据路径"""
    return UnstructuredDataPaths.get_report_metadata_path(*args, **kwargs)


def get_report_pdf_path(*args, **kwargs) -> Path:
    """获取研报PDF路径"""
    return UnstructuredDataPaths.get_report_pdf_path(*args, **kwargs)


def get_sentiment_path(*args, **kwargs) -> Path:
    """获取舆情路径"""
    return UnstructuredDataPaths.get_sentiment_path(*args, **kwargs)


def get_policy_rules_path(*args, **kwargs) -> Path:
    """获取政策文本路径"""
    return UnstructuredDataPaths.get_policy_rules_path(*args, **kwargs)


def get_policy_file_path(*args, **kwargs) -> Path:
    """获取政策附件路径"""
    return UnstructuredDataPaths.get_policy_file_path(*args, **kwargs)


def get_event_meta_path(*args, **kwargs) -> Path:
    """获取事件元数据路径"""
    return UnstructuredDataPaths.get_event_meta_path(*args, **kwargs)


def get_event_pdf_path(*args, **kwargs) -> Path:
    """获取事件PDF路径"""
    return UnstructuredDataPaths.get_event_pdf_path(*args, **kwargs)


if __name__ == '__main__':
    """测试路径生成"""
    import json
    
    print("=" * 60)
    print("非结构化数据路径管理器 - 测试")
    print("=" * 60)
    
    # 1. 公告路径
    print("\n1. 公告模块")
    print(f"  元数据: {get_announcement_metadata_path('cninfo', '2025', '01')}")
    print(f"  PDF文件: {get_announcement_file_path('000001.SZ', '20250115', '年度报告.pdf', True)}")
    
    # 2. 新闻路径
    print("\n2. 新闻模块")
    print(f"  新浪: {get_news_path('sina', '20250115')}")
    print(f"  东财: {get_news_path('eastmoney', '20250115')}")
    
    # 3. 研报路径
    print("\n3. 研报模块")
    print(f"  元数据: {get_report_metadata_path('2025', '01')}")
    print(f"  PDF: {get_report_pdf_path('600519.SH', '20250115', '中信证券', '买入')}")
    
    # 4. 舆情路径
    print("\n4. 舆情模块")
    print(f"  雪球: {get_sentiment_path('xueqiu', '20250115')}")
    print(f"  股吧: {get_sentiment_path('guba', '20250115')}")
    
    # 5. 政策路径
    print("\n5. 政策模块")
    print(f"  规则: {get_policy_rules_path('csrc', '2025')}")
    print(f"  附件: {get_policy_file_path('csrc', 'csrc_2025_001', '附件1.pdf')}")
    
    # 6. 事件路径
    print("\n6. 事件驱动模块")
    print(f"  元数据: {get_event_meta_path('penalty', '2025', '01')}")
    print(f"  PDF: {get_event_pdf_path('penalty', '000001.SZ', '20250115', '处罚决定书.pdf')}")
    
    # 7. 存储统计
    print("\n7. 存储统计")
    summary = UnstructuredDataPaths.get_storage_summary()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    
    print("\n" + "=" * 60)
