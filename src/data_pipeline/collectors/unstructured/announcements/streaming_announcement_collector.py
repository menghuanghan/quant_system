"""
流式公告采集器 (Streaming Announcement Collector)

基于 StreamingCollector 实现的公告采集器，支持：
- 即时清洗（Extract-on-the-fly）：PDF 下载后立即提取文本
- 智能分流：文本/扫描件/高价值 PDF 三路处理
- 时间清洗：防止未来函数泄露
- 缓冲池落盘：Parquet 压缩存储

用法：
    >>> with StreamingAnnouncementCollector() as collector:
    ...     stats = collector.collect('2024-01-01', '2024-01-31')
    ...     print(stats)
"""

import logging
from typing import Optional, List, Dict, Any, Generator
from datetime import datetime

from ..base import StreamingCollector, ContentType
from ..scraper_base import ScraperBase, get_scraper
from ..request_utils import safe_request, RequestDisguiser
from ..rate_limiter import get_rate_limiter
from .cninfo_crawler import CninfoAnnouncementCrawler

# 尝试导入 PDF 解析器
try:
    from ....clean.unstructured.pdf_parser import PDFParser, ScannedPDFError
    HAS_PDF_PARSER = True
except ImportError:
    HAS_PDF_PARSER = False
    PDFParser = None
    ScannedPDFError = Exception

logger = logging.getLogger(__name__)


class StreamingAnnouncementCollector(StreamingCollector):
    """
    流式公告采集器
    
    从巨潮资讯采集公告，采用流式处理：
    1. 获取元数据列表（标题、URL、日期等）
    2. 逐条下载 PDF 到内存（get_bytes）
    3. 即时清洗：PDFParser.extract_from_bytes()
    4. 分流处理：
       - 正常文本 → 缓冲池 → Parquet
       - 扫描件 → 保存 PDF 文件 → 缓冲池（带 file_path）
       - 高价值公告 → 保存 PDF 文件 + 提取文本 → 缓冲池
    """
    
    # 数据源配置
    SOURCE_NAME = "cninfo"
    DOMAIN = "unstructured"
    SUB_DOMAIN = "announcements"
    
    def __init__(
        self,
        use_proxy: bool = False,
        extract_text: bool = True,
        max_pdf_size_mb: float = 50.0,
        skip_scanned: bool = False,
        **kwargs
    ):
        """
        初始化流式公告采集器
        
        Args:
            use_proxy: 是否使用代理
            extract_text: 是否提取 PDF 文本（False 则仅采集元数据）
            max_pdf_size_mb: 最大 PDF 大小限制（MB）
            skip_scanned: 是否跳过扫描件（不保存）
            **kwargs: StreamingCollector 参数
        """
        super().__init__(**kwargs)
        
        self.use_proxy = use_proxy
        self.extract_text = extract_text
        self.max_pdf_size_mb = max_pdf_size_mb
        self.skip_scanned = skip_scanned
        
        # 元数据采集器（复用现有逻辑）
        self._metadata_crawler = CninfoAnnouncementCrawler(use_proxy=use_proxy)
        
        # 爬虫工具（用于下载 PDF）
        self._scraper = ScraperBase(
            use_proxy=use_proxy,
            rate_limit=True,
            timeout=60  # PDF 下载超时稍长
        )
        
        # PDF 解析器
        if HAS_PDF_PARSER:
            self._pdf_parser = PDFParser(
                backend='auto',  # 自动降级
                check_scanned=True,
                scanned_threshold_kb=500,
                scanned_min_chars=50
            )
        else:
            self._pdf_parser = None
            self.logger.warning(
                "PDFParser 未导入，将仅采集元数据（无文本提取）"
            )
    
    def _collect_items(
        self,
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
        **kwargs
    ) -> Generator[Dict[str, Any], None, None]:
        """
        采集公告数据（生成器）
        
        Args:
            start_date: 开始日期
            end_date: 结束日期
            stock_codes: 股票代码列表
            categories: 公告类型列表
            exchanges: 交易所列表
        
        Yields:
            处理后的数据字典
        """
        # Step 1: 获取元数据列表
        self.logger.info("获取公告元数据...")
        metadata_df = self._metadata_crawler.collect(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            categories=categories,
            exchanges=exchanges
        )
        
        if metadata_df.empty:
            self.logger.warning("未获取到公告元数据")
            return
        
        self.logger.info(f"获取到 {len(metadata_df)} 条公告元数据")
        
        # Step 2: 逐条处理
        for idx, row in metadata_df.iterrows():
            try:
                item = self._process_single_announcement(row)
                if item is not None:
                    yield item
            except Exception as e:
                self.logger.warning(
                    f"处理公告失败 [{row.get('ts_code', 'N/A')}]: {e}"
                )
                continue
    
    def _process_single_announcement(
        self,
        row: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        处理单条公告
        
        分流策略：
        1. 无 URL → 仅元数据
        2. 非 PDF URL → 仅元数据
        3. PDF URL:
           a. 下载到内存
           b. 尝试提取文本
           c. 根据结果分流（文本/扫描件/高价值）
        
        Args:
            row: 元数据行（字典）
        
        Returns:
            处理后的数据字典
        """
        url = row.get('url', '')
        ts_code = row.get('ts_code', '')
        ann_date = row.get('ann_date', '')
        title = row.get('title', '')
        category = row.get('category', '')
        
        # 基础数据
        base_item = {
            'ts_code': ts_code,
            'name': row.get('name', ''),
            'title': title,
            'publish_time': ann_date,  # 原始时间，将被 StreamingCollector 清洗
            'category': category,
            'url': url,
            'original_id': row.get('original_id', ''),
            'is_correction': row.get('is_correction', False),
            'content_type': 'text',
        }
        
        # 无 URL 或无需提取文本 → 仅元数据
        if not url or not self.extract_text:
            return base_item
        
        # 非 PDF URL → 仅元数据
        if not url.lower().endswith('.pdf'):
            return base_item
        
        # PDF 处理
        if self._pdf_parser is None:
            return base_item
        
        # Step 1: 下载 PDF 到内存
        try:
            pdf_bytes = self._scraper.get_bytes(
                url,
                max_size_mb=self.max_pdf_size_mb,
                referer='http://www.cninfo.com.cn/',
                timeout=60
            )
        except ValueError as e:
            # 文件过大
            self.logger.warning(f"PDF 过大，跳过: {url} - {e}")
            return base_item
        except Exception as e:
            self.logger.debug(f"PDF 下载失败: {url} - {e}")
            return base_item
        
        if pdf_bytes is None:
            return base_item
        
        # Step 2: 判断是否高价值公告
        is_high_value = self._is_high_value_category(category)
        
        # Step 3: 尝试提取文本
        content = None
        is_scanned = False
        
        try:
            content = self._pdf_parser.extract_from_bytes(
                pdf_bytes,
                normalize=True
            )
        except ScannedPDFError:
            is_scanned = True
            self.logger.debug(f"检测到扫描件: {title}")
        except Exception as e:
            self.logger.warning(f"PDF 解析失败: {title} - {e}")
            is_scanned = True  # 解析失败按扫描件处理
        
        # Step 4: 分流处理
        file_path = None
        
        # 情况 A: 扫描件
        if is_scanned:
            if self.skip_scanned:
                self.logger.debug(f"跳过扫描件: {title}")
                return None  # 丢弃
            
            # 保存扫描件 PDF
            file_path = self._save_scanned_pdf(
                pdf_bytes, ts_code, ann_date, title
            )
            base_item['content_type'] = 'scanned'
            base_item['file_path'] = file_path
            base_item['content'] = None
            return base_item
        
        # 情况 B: 高价值公告（无论是否提取成功，保留原件）
        if is_high_value:
            file_path = self._save_high_value_pdf(
                pdf_bytes, ts_code, ann_date, title, category
            )
            base_item['content_type'] = 'full_pdf'
            base_item['file_path'] = file_path
        
        # 情况 C: 正常文本（包括高价值的文本部分）
        if content:
            base_item['content'] = content
            base_item['content_type'] = 'full_pdf' if is_high_value else 'text'
        
        return base_item
    
    def close(self):
        """关闭资源"""
        super().close()
        if self._scraper:
            self._scraper.close()


# ============== 便捷函数 ==============

def collect_announcements_streaming(
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    exchanges: Optional[List[str]] = None,
    use_proxy: bool = False,
    extract_text: bool = True,
    buffer_size: int = 500,
    time_mode: str = 'conservative'
) -> Dict[str, Any]:
    """
    流式采集公告数据（推荐接口）
    
    采用即时清洗策略：
    - PDF 下载后立即提取文本
    - 仅保留文本内容，丢弃原始 PDF（扫描件和高价值除外）
    - 时间清洗防止未来函数
    - 批量落盘到 Parquet
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        stock_codes: 股票代码列表
        categories: 公告类型列表
        exchanges: 交易所列表
        use_proxy: 是否使用代理
        extract_text: 是否提取 PDF 文本
        buffer_size: 缓冲区大小
        time_mode: 时间填充模式（conservative/aggressive/ultra_conservative）
    
    Returns:
        采集统计信息
    
    Example:
        >>> stats = collect_announcements_streaming(
        ...     '2024-01-01', '2024-01-31',
        ...     categories=['年报', '业绩预告']
        ... )
        >>> print(f"采集 {stats['cleaned_items']} 条")
    """
    with StreamingAnnouncementCollector(
        use_proxy=use_proxy,
        extract_text=extract_text,
        buffer_size=buffer_size,
        time_mode=time_mode
    ) as collector:
        return collector.collect(
            start_date=start_date,
            end_date=end_date,
            stock_codes=stock_codes,
            categories=categories,
            exchanges=exchanges
        )


def verify_time_cleaning(
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    sample_size: int = 10
) -> List[Dict[str, str]]:
    """
    验证时间清洗效果（防泄露检查）
    
    用于 QA 检查：确认只有日期的公告被正确填充为 17:00:00
    
    Args:
        start_date: 开始日期
        end_date: 结束日期
        stock_codes: 股票代码
        sample_size: 采样数量
    
    Returns:
        采样结果列表
    """
    results = []
    
    # 获取元数据
    crawler = CninfoAnnouncementCrawler()
    df = crawler.collect(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes
    )
    
    if df.empty:
        return results
    
    # 采样检查
    sample_df = df.head(sample_size)
    
    for _, row in sample_df.iterrows():
        raw_time = row.get('ann_date', '')
        
        # 模拟时间清洗
        try:
            from ....clean.unstructured.time_utils import standardize_publish_time
            cleaned_time = standardize_publish_time(raw_time, 'conservative')
        except ImportError:
            cleaned_time = f"{raw_time} 17:00:00" if len(raw_time) == 10 else raw_time
        
        results.append({
            'ts_code': row.get('ts_code', ''),
            'title': row.get('title', '')[:30],
            'raw_time': raw_time,
            'cleaned_time': cleaned_time,
            'is_conservative': '17:00:00' in cleaned_time if len(raw_time) == 10 else True
        })
    
    return results
