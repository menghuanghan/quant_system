"""
巨潮资讯公告爬虫

从巨潮资讯网（Cninfo）采集上市公司公告：
- 支持按股票代码采集
- 支持按日期范围采集
- 支持公告PDF下载
- 内置反爬策略
"""

import os
import re
import json
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd

from ..base import (
    UnstructuredCollector,
    AnnouncementCategory,
    DataSourceType,
    DateRangeIterator,
    CollectionProgress,
    generate_task_id
)
from ..request_utils import (
    safe_request,
    safe_download_file,
    RequestSession,
    RequestDisguiser
)
from ..rate_limiter import get_rate_limiter

logger = logging.getLogger(__name__)


class CninfoAnnouncementCrawler(UnstructuredCollector):
    """
    巨潮资讯公告爬虫
    
    作为核心数据源，从巨潮资讯网采集公告
    """
    
    # API 端点
    BASE_URL = "http://www.cninfo.com.cn"
    SEARCH_API = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    STOCK_LIST_API = "http://www.cninfo.com.cn/new/data/szse_stock.json"
    SSE_STOCK_LIST_API = "http://www.cninfo.com.cn/new/data/sse_stock.json"
    PDF_BASE_URL = "http://static.cninfo.com.cn/"
    
    # 公告类型代码映射
    CATEGORY_CODES = {
        # 定期报告
        "年报": "category_ndbg_szsh",
        "中报": "category_bndbg_szsh",
        "一季报": "category_yjdbg_szsh",
        "三季报": "category_sjdbg_szsh",
        
        # 临时公告
        "并购重组": "category_bcgz_szsh",
        "股权变动": "category_gqbd_szsh",
        "关联交易": "category_gljy_szsh",
        "业绩预告": "category_yjygjxz_szsh",
        
        # IPO相关
        "首发上市": "category_sf_szsh",
        "增发配股": "category_zf_szsh",
        
        # 其他
        "股东大会": "category_gddh_szsh",
        "董事会": "category_dsh_szsh",
        "监事会": "category_jsh_szsh",
        "日常经营": "category_rcjy_szsh",
    }
    
    # 交易所列代码
    EXCHANGE_CODES = {
        'szse': 'szse',   # 深交所
        'sse': 'sse',     # 上交所
        'bse': 'bse',     # 北交所
    }
    
    def __init__(
        self,
        use_proxy: bool = False,
        download_dir: Optional[str] = None
    ):
        """
        Args:
            use_proxy: 是否使用代理
            download_dir: PDF下载目录
        """
        super().__init__()
        self._use_proxy = use_proxy
        self._download_dir = download_dir or os.path.join(
            os.path.expanduser('~'), 
            '.quant_data', 
            'announcements'
        )
        self._stock_cache = {}
        self._disguiser = RequestDisguiser()
    
    def collect(
        self,
        start_date: str,
        end_date: str,
        stock_codes: Optional[List[str]] = None,
        categories: Optional[List[str]] = None,
        exchanges: Optional[List[str]] = None,
        include_delisted: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        采集公告数据
        
        Args:
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD）
            stock_codes: 股票代码列表（纯数字或带后缀）
            categories: 公告类型列表
            exchanges: 交易所列表（szse/sse/bse）
            include_delisted: 是否包含退市公司
        
        Returns:
            标准化的公告数据DataFrame
        """
        all_data = []
        
        # 确定采集范围
        if stock_codes:
            # 按股票采集
            for code in stock_codes:
                df = self._collect_by_stock(
                    code, start_date, end_date, categories
                )
                if not df.empty:
                    all_data.append(df)
        else:
            # 全市场采集（按日期分批）
            exchanges = exchanges or ['szse', 'sse']
            
            for exchange in exchanges:
                df = self._collect_by_exchange(
                    exchange, start_date, end_date, categories
                )
                if not df.empty:
                    all_data.append(df)
        
        if not all_data:
            return pd.DataFrame(columns=self.STANDARD_FIELDS)
        
        result = pd.concat(all_data, ignore_index=True)
        
        # 版本管理：识别更正公告
        result = self._process_corrections(result)
        
        # 去重
        result = self._deduplicate(result)
        
        return self._standardize_dataframe(result, DataSourceType.CNINFO.value)
    
    def _collect_by_stock(
        self,
        stock_code: str,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """按股票代码采集"""
        # 清理代码格式
        if '.' in stock_code:
            stock_code = stock_code.split('.')[0]
        
        stock_code = stock_code.zfill(6)
        
        # 确定交易所
        if stock_code.startswith(('0', '2', '3')):
            exchange = 'szse'
        elif stock_code.startswith(('6', '9')):
            exchange = 'sse'
        else:
            exchange = 'szse'
        
        # 尝试获取orgId以使用精确查询
        org_id = self._get_org_id(stock_code, exchange)
        
        if org_id:
            # 使用精确查询（code,orgId格式）
            params = self._build_search_params(
                stock_code=f"{stock_code},{org_id}",
                start_date=start_date,
                end_date=end_date,
                categories=categories
            )
            df = self._fetch_announcements(params)
            if not df.empty:
                return df
        
        # 降级方案：按交易所采集后过滤
        logger.info(f"使用交易所过滤方式采集 {stock_code}")
        params = self._build_search_params(
            start_date=start_date,
            end_date=end_date,
            categories=categories,
            exchange=exchange
        )
        
        df = self._fetch_announcements(params, max_pages=50)
        
        if not df.empty and 'ts_code' in df.columns:
            # 过滤目标股票
            target_ts_code = self._code_to_tscode(stock_code)
            df = df[df['ts_code'] == target_ts_code]
        
        return df
    
    def _get_org_id(self, stock_code: str, exchange: str) -> Optional[str]:
        """获取股票的orgId"""
        stock_df = self.get_stock_list(exchange)
        if stock_df.empty:
            return None
        
        match = stock_df[stock_df['code'] == stock_code]
        if not match.empty:
            return match.iloc[0].get('orgId')
        return None

    
    def _collect_by_exchange(
        self,
        exchange: str,
        start_date: str,
        end_date: str,
        categories: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """按交易所采集"""
        all_data = []
        
        # 分月采集，避免单次请求过大
        for chunk_start, chunk_end in DateRangeIterator(start_date, end_date, chunk_days=30):
            params = self._build_search_params(
                start_date=chunk_start,
                end_date=chunk_end,
                categories=categories,
                exchange=exchange
            )
            
            df = self._fetch_announcements(params)
            if not df.empty:
                all_data.append(df)
                logger.info(
                    f"采集 {exchange} {chunk_start}~{chunk_end}: {len(df)} 条"
                )
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def _build_search_params(
        self,
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        categories: Optional[List[str]] = None,
        exchange: Optional[str] = None,
        page_num: int = 1,
        page_size: int = 30
    ) -> Dict[str, Any]:
        """构建查询参数"""
        params = {
            'pageNum': page_num,
            'pageSize': page_size,
            'tabName': 'fulltext',
            'isHLtitle': 'true',
        }
        
        # 股票代码
        if stock_code:
            params['stock'] = stock_code
        
        # 日期范围
        if start_date and end_date:
            # 转换日期格式
            start = start_date.replace('-', '')
            end = end_date.replace('-', '')
            params['seDate'] = f"{start[:4]}-{start[4:6]}-{start[6:8]}~{end[:4]}-{end[4:6]}-{end[6:8]}"
        
        # 公告类型
        if categories:
            category_codes = []
            for cat in categories:
                if cat in self.CATEGORY_CODES:
                    category_codes.append(self.CATEGORY_CODES[cat])
            if category_codes:
                params['category'] = ';'.join(category_codes)
        
        # 交易所
        if exchange:
            params['column'] = self.EXCHANGE_CODES.get(exchange, 'szse')
        
        return params
    
    def _fetch_announcements(
        self,
        params: Dict[str, Any],
        max_pages: int = 100
    ) -> pd.DataFrame:
        """获取公告列表"""
        all_records = []
        page_num = 1
        
        while page_num <= max_pages:
            params['pageNum'] = page_num
            
            response = safe_request(
                self.SEARCH_API,
                method='POST',
                data=params,
                headers=self._disguiser.get_json_headers({
                    'Referer': 'http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search',
                    'Origin': 'http://www.cninfo.com.cn',
                }),
                use_proxy=self._use_proxy,
                rate_limit=True
            )
            
            if response is None:
                logger.warning(f"请求失败，页码: {page_num}")
                break
            
            try:
                data = response.json()
            except Exception as e:
                logger.warning(f"解析JSON失败: {e}")
                break
            
            # 提取公告列表
            announcements = data.get('announcements', [])
            
            if not announcements:
                break
            
            for ann in announcements:
                record = self._parse_announcement(ann)
                if record:
                    all_records.append(record)
            
            # 检查是否还有下一页
            has_more = data.get('hasMore', False)
            total_count = data.get('totalRecordNum', 0)
            
            if not has_more or len(all_records) >= total_count:
                break
            
            page_num += 1
        
        if not all_records:
            return pd.DataFrame()
        
        return pd.DataFrame(all_records)
    
    def _parse_announcement(self, ann: Dict) -> Optional[Dict]:
        """解析单条公告数据"""
        try:
            # 提取字段
            title = ann.get('announcementTitle', '')
            ann_date = ann.get('announcementTime', '')
            sec_code = ann.get('secCode', '')
            sec_name = ann.get('secName', '')
            pdf_url = ann.get('adjunctUrl', '')
            ann_id = ann.get('announcementId', '')
            
            # 转换日期（时间戳 -> YYYY-MM-DD）
            if isinstance(ann_date, (int, float)):
                ann_date = datetime.fromtimestamp(ann_date / 1000).strftime('%Y-%m-%d')
            
            # 构建完整PDF URL
            if pdf_url and not pdf_url.startswith('http'):
                pdf_url = urljoin(self.PDF_BASE_URL, pdf_url)
            
            # 推断类型
            category = self._infer_category(title)
            is_correction = self._detect_correction(title)
            
            # 生成ts_code
            ts_code = self._code_to_tscode(sec_code)
            
            return {
                'ts_code': ts_code,
                'name': sec_name,
                'title': title,
                'ann_date': ann_date,
                'category': category,
                'url': pdf_url,
                'original_id': str(ann_id),
                'is_correction': is_correction,
                'list_status': 'L',
            }
            
        except Exception as e:
            logger.debug(f"解析公告失败: {e}")
            return None
    
    def _infer_category(self, title: str) -> str:
        """从标题推断公告类型"""
        if not title:
            return AnnouncementCategory.OTHER.value
        
        title = str(title)
        
        # 定期报告
        if '年度报告' in title or '年报' in title:
            return AnnouncementCategory.PERIODIC_ANNUAL.value
        if '半年度报告' in title or '中报' in title:
            return AnnouncementCategory.PERIODIC_SEMI.value
        if '第一季度' in title or '一季报' in title:
            return AnnouncementCategory.PERIODIC_Q1.value
        if '第三季度' in title or '三季报' in title:
            return AnnouncementCategory.PERIODIC_Q3.value
        
        # 临时公告
        if any(kw in title for kw in ['并购', '重组', '收购', '合并']):
            return AnnouncementCategory.MERGER_ACQUISITION.value
        if any(kw in title for kw in ['增持', '买入']):
            return AnnouncementCategory.EQUITY_INCREASE.value
        if any(kw in title for kw in ['减持', '卖出']):
            return AnnouncementCategory.EQUITY_DECREASE.value
        if '重大合同' in title:
            return AnnouncementCategory.MAJOR_CONTRACT.value
        if any(kw in title for kw in ['诉讼', '仲裁', '起诉']):
            return AnnouncementCategory.LITIGATION.value
        if any(kw in title for kw in ['处罚', '警示', '监管']):
            return AnnouncementCategory.REGULATORY_PENALTY.value
        if '业绩预告' in title:
            return AnnouncementCategory.EARNINGS_FORECAST.value
        if '业绩快报' in title:
            return AnnouncementCategory.EARNINGS_EXPRESS.value
        
        # 更正类
        if any(kw in title for kw in ['更正', '补充', '修订']):
            return AnnouncementCategory.CORRECTION.value
        
        return AnnouncementCategory.OTHER.value
    
    def _process_corrections(self, df: pd.DataFrame) -> pd.DataFrame:
        """处理更正公告，建立版本关联"""
        if df.empty or 'is_correction' not in df.columns:
            return df
        
        correction_mask = df['is_correction'] == True
        if not correction_mask.any():
            return df
        
        # 尝试为更正公告找到原始公告
        for idx in df[correction_mask].index:
            title = df.loc[idx, 'title']
            ts_code = df.loc[idx, 'ts_code']
            
            # 提取原公告引用
            original_ref = self._extract_original_ref(title)
            if not original_ref:
                continue
            
            # 在同一股票的公告中查找匹配
            same_stock = df[df['ts_code'] == ts_code]
            for orig_idx in same_stock.index:
                if orig_idx == idx:
                    continue
                orig_title = df.loc[orig_idx, 'title']
                if original_ref in orig_title:
                    # 找到原始公告
                    df.loc[idx, 'correction_of'] = df.loc[orig_idx, 'original_id']
                    break
        
        return df
    
    @staticmethod
    def _code_to_tscode(code: str) -> str:
        """股票代码转换为ts_code格式"""
        code = str(code).zfill(6)
        if code.startswith(('0', '2', '3')):
            return f"{code}.SZ"
        elif code.startswith(('6', '9')):
            return f"{code}.SH"
        elif code.startswith(('4', '8')):
            return f"{code}.BJ"
        return f"{code}.SZ"
    
    def download_announcement(
        self,
        url: str,
        save_dir: Optional[str] = None,
        filename: Optional[str] = None
    ) -> Optional[str]:
        """
        下载公告PDF
        
        Args:
            url: 公告URL
            save_dir: 保存目录（默认使用配置的目录）
            filename: 文件名（默认从URL提取）
        
        Returns:
            保存的文件路径，失败返回None
        """
        save_dir = save_dir or self._download_dir
        
        if not filename:
            # 从URL提取文件名
            filename = url.split('/')[-1]
            if not filename.endswith('.pdf'):
                filename += '.pdf'
        
        save_path = os.path.join(save_dir, filename)
        
        success = safe_download_file(
            url,
            save_path,
            use_proxy=self._use_proxy
        )
        
        return save_path if success else None
    
    def batch_download(
        self,
        df: pd.DataFrame,
        save_dir: Optional[str] = None,
        max_concurrent: int = 3
    ) -> pd.DataFrame:
        """
        批量下载公告PDF
        
        Args:
            df: 包含url列的公告DataFrame
            save_dir: 保存目录
            max_concurrent: 最大并发数
        
        Returns:
            添加了file_path列的DataFrame
        """
        if df.empty or 'url' not in df.columns:
            return df
        
        save_dir = save_dir or self._download_dir
        os.makedirs(save_dir, exist_ok=True)
        
        file_paths = []
        for _, row in df.iterrows():
            url = row.get('url', '')
            if not url:
                file_paths.append(None)
                continue
            
            # 生成文件名
            ts_code = row.get('ts_code', 'unknown')
            ann_date = row.get('ann_date', 'unknown')
            filename = f"{ts_code}_{ann_date}_{row.name}.pdf"
            
            path = self.download_announcement(url, save_dir, filename)
            file_paths.append(path)
        
        df['file_path'] = file_paths
        return df
    
    def get_stock_list(self, exchange: str = 'szse') -> pd.DataFrame:
        """
        获取股票列表
        
        Args:
            exchange: 交易所（szse/sse）
        
        Returns:
            股票列表DataFrame
        """
        cache_key = f"stock_list_{exchange}"
        if cache_key in self._stock_cache:
            return self._stock_cache[cache_key]
        
        url = self.STOCK_LIST_API if exchange == 'szse' else self.SSE_STOCK_LIST_API
        
        response = safe_request(
            url,
            use_proxy=self._use_proxy,
            rate_limit=True
        )
        
        if response is None:
            return pd.DataFrame()
        
        try:
            data = response.json()
            stocks = data.get('stockList', [])
            
            df = pd.DataFrame([
                {
                    'code': s.get('code', ''),
                    'name': s.get('zwjc', ''),
                    'orgId': s.get('orgId', ''),
                }
                for s in stocks
            ])
            
            self._stock_cache[cache_key] = df
            return df
            
        except Exception as e:
            logger.error(f"解析股票列表失败: {e}")
            return pd.DataFrame()


# 便捷函数

def get_cninfo_announcements(
    start_date: str,
    end_date: str,
    stock_codes: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    exchanges: Optional[List[str]] = None,
    use_proxy: bool = False
) -> pd.DataFrame:
    """
    从巨潮资讯获取公告数据
    
    Args:
        start_date: 开始日期（YYYY-MM-DD）
        end_date: 结束日期（YYYY-MM-DD）
        stock_codes: 股票代码列表
        categories: 公告类型列表
        exchanges: 交易所列表
        use_proxy: 是否使用代理
    
    Returns:
        公告数据DataFrame
    
    Example:
        >>> df = get_cninfo_announcements('2024-01-01', '2024-01-31')
        >>> df = get_cninfo_announcements(
        ...     '2024-01-01', '2024-12-31',
        ...     stock_codes=['000001'],
        ...     categories=['年报', '业绩预告']
        ... )
    """
    crawler = CninfoAnnouncementCrawler(use_proxy=use_proxy)
    return crawler.collect(
        start_date=start_date,
        end_date=end_date,
        stock_codes=stock_codes,
        categories=categories,
        exchanges=exchanges
    )
